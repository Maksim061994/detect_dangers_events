import os, glob, time
from ultralytics import YOLO
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as albu

import numpy as np
import pandas as pd

from app.config.settings import get_settings


def get_preprocessing(preprocessing_fn):
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    _transform = []
    if preprocessing_fn:
        _transform.append(albu.Lambda(image=preprocessing_fn))
    _transform.append(albu.Lambda(name='to_tenzor', image=to_tensor, mask=to_tensor))
    return albu.Compose(_transform)


settings = get_settings()

# TOOD: с 24 по 39 строчку привести в чувства
IMG_HEIGHT = 704
IMG_WIDTH = 704
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODEL_DETECTION = YOLO(settings.detection_model_checkpoint_dir)
MODEL_SEGMENTATION = torch.load(settings.segmentation_model_checkpoint_dir, DEVICE)
MODEL_SEGMENTATION.eval()
AUGUMENTATOR =  albu.Compose([
    albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])
PREPROCESSOR = get_preprocessing(
    smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
)


class DetectorDangerEvent():

    def __init__(self):
        # TODO: добавить наследование от BaseModel (super().__init__())
        self.class_list = [0, 1, 2, 3, 5, 7]
        self.conf_yolo = 0.6
        self.coeff_corner_contours_danger = 0.5
        self.coeff_corner_contours_warning = 0.85

    @staticmethod
    def __corner_contours(shape, coeff, type_viol=None):
        height = int(shape[0] / 2)
        weight = int(shape[1] / 2)

        if type_viol == 'warning':
            left_sight = 0
            right_sight = int(shape[1])
        else:
            left_sight = int(weight - int(0.45 * weight))
            right_sight = int(weight + int(0.45 * weight))

        corner = int(weight * coeff)
        left_corner = left_sight + corner
        right_corner = right_sight - corner

        point1 = [left_corner, height]
        point2 = [left_sight, 2 * height]
        point3 = [right_sight, 2 * height]
        point4 = [right_corner, height]

        contours = np.array([point1, point2, point3, point4], dtype=np.int32)

        return contours

    def __danger_area(self, predict_frame):
        shape_image = predict_frame.shape
        pts = self.__corner_contours(shape_image, coeff=self.coeff_corner_contours_danger)
        mask = np.zeros(shape_image, dtype=np.uint8)
        ignore_mask_color = (255)
        cv2.fillConvexPoly(mask, pts, ignore_mask_color)
        masked_image = cv2.bitwise_and(predict_frame, mask)
        return masked_image

    def __warning_area(self, image_frame):
        shape_image = image_frame.shape[:2]
        pts = self.__corner_contours(shape_image, coeff=self.coeff_corner_contours_warning, type_viol='warning')

        mask = np.zeros(image_frame.shape, dtype=np.uint8)
        channel_count = image_frame.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        cv2.fillConvexPoly(mask, pts, ignore_mask_color)
        masked_image = cv2.bitwise_and(image_frame, mask)
        binary_mask = (masked_image[:, :, 0] > 0).astype(int)

        return binary_mask

    def __predict_rails_way(self, image, shape):
        """
        Выход модели, возвращает тензоры маски рельс(1) или пути(0).
        :param image:
        :param shape:
        :return:
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = AUGUMENTATOR(image=image)['image']
        img = PREPROCESSOR(image=img)['image']

        x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
        pr_mask = MODEL_SEGMENTATION.predict(x_tensor)

        pr_mask = pr_mask.cpu().squeeze(0).numpy().round()
        answer = (np.sum(pr_mask, axis=0) > 0).astype('uint8')
        answer_res = cv2.resize(answer, shape)

        return answer_res

    def __compute_predict_by_frame(
            self, image, df,
            count_status, count_alarm, count_frame,
            status, alarm_status
    ):
        """
        Функция прогнозирования по фрейму
        # TODO: рефакторинг
        :param image:
        :param df:
        :param count_status:
        :param count_alarm:
        :param count_frame:
        :param status:
        :param alarm_status:
        :return:
        """

        shape_image = image.shape[1], image.shape[0]

        if count_status == 50:
            status = False

        if (count_alarm == 50) & (alarm_status == True):
            alarm_status = False
            label = 3
            df.loc[len(df)] = label, count_frame

        predict_yolo = MODEL_DETECTION.predict(
            source=image, classes=self.class_list, conf=self.conf_yolo, verbose=False
        )
        bbox = predict_yolo[0].boxes.data.detach().cpu().numpy()
        bbox_data = bbox[bbox[:, -1] != 7][:, :-2].astype(int)

        if len(bbox_data) != 0:

            predict_segment = self.__predict_rails_way(image, (shape_image))
            binary_danger_area = self.__danger_area(predict_segment)

            for i in bbox_data:

                test_area = binary_danger_area[i[1]:i[3], i[0]:i[2]]
                summary_check = test_area.sum()

                if (summary_check > 20) & (status == False):

                    label = 1
                    status = True
                    count_status = -1
                    count_alarm -= 1
                    df.loc[len(df)] = label, count_frame
                    break

                elif summary_check > 10:
                    count_status = -1
                    count_alarm -= 1
                    break

                else:

                    binary_test_area = self.__warning_area(image)
                    test_area = binary_test_area[i[1]:i[3], i[0]:i[2]]
                    summary_check = test_area.sum()

                    thresholder = int(0.2 * test_area.shape[0] * test_area.shape[1])

                    if (summary_check > thresholder) & (alarm_status == False):
                        label = 2
                        alarm_status = True
                        count_alarm = -1
                        count_status -= 1
                        df.loc[len(df)] = label, count_frame
                        break

                    elif summary_check > thresholder:
                        count_alarm = -1
                        count_status -= 1
                        break
                    else:
                        continue

        count_alarm += 1
        count_status += 1
        return df, count_status, count_frame, count_alarm, status, alarm_status

    def predict(self, video):
        df = pd.DataFrame(
            columns=['type_violation', 'number_frame']
        )
        count_status = 0
        count_frame = 0
        count_alarm = 0
        status = False
        alarm_status = False
        ret = True
        fps = video.get(cv2.CAP_PROP_FPS)
        while ret:
            ret, frame = video.read()
            if ret:
                df, count_status, count_frame, count_alarm, status, alarm_status = self.__compute_predict_by_frame(
                    image=frame, df=df,
                    count_status=count_status, count_alarm=count_alarm, count_frame=count_frame,
                    status=status, alarm_status=alarm_status
                )
                count_frame += 1
        df['seconds'] = (df['number_frame'] / fps).astype(int)
        print(df)
        return df

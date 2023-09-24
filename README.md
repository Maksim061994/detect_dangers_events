# Кейс "Безопасный маршрут"

## 1. Навигация

 - Модели: `app/handlers/cv_models/models/*`
 - Submit на тестовой выборке: `submit`
 - Пользовательский интерфейс: https://board.vniizht.ru
 - Ноутбук: `notebooks`
 - Ссылка на API: https://ml.api.vniizht.ru
 - Код АПИ: `app`
 - Документация и презентация: `docs`
 - Примеры работы системы: `examples`



## 2. Запуск и работа с системой

**Для успешного запуска, на сервере необходимо наличие GPU NVidia с установленными драйверами**  
**В настройках Docker должна быть включена поддержка GPU**  

В конфигурационном файлу Docker `/etc/docker/daemon.json` должны быть добавлены следующие параметры:  

```
  "default-runtime": "nvidia",  
  "runtimes": {  
    "nvidia": {  
      "path": "nvidia-container-runtime",  
      "runtimeArgs": []  
    }  
  }  
```
 
1. Отредактировать файлы `.env` и `.env.dev` под свои нужды
2. Отредактировать конфигурационные файлы в каталоге `configs`
3. Запустить команду `docker-compose up -d` дождаться запуска всех контейнеров
4. Ввести отредактировав под свои нужды команду:
docker exec -it data_science_ui superset fab create-admin \
			   --username admin \
			   --firstname Superset \
			   --lastname Admin \
			   --email admin@admin.com \
			   --password hf#d,mIDN5dhI*C539JF; \
docker exec -it data_science_ui superset db upgrade; \
docker exec -it data_science_ui superset init;

5. Ссылки для работы в системе:
 - ip_host:8089 - Superset (логин и пароль в пункте 4 текущего документа)
 - ip_host:3025/docs - API
 - ip_host:5555/dashboard - Flower (отслеживание работы Celery)\
 
6. Авторизоваться в Superset используя имя и пароль заданные выше
 - Загрузить Dashboard из сохраненного шаблона: dashboards/dashboard_export_20230923T205330.zip

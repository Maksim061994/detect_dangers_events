FROM python:3.10.9-slim as builder

# install system dependencies
RUN apt-get update
RUN apt-get install build-essential -y

# set Work Directory
WORKDIR /usr/src/app

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

FROM python:3.10.9-slim

#Timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir -p /opt/
WORKDIR /opt/

# set environment variable
#Prevents Python from writing pyc files to disc (equivalent python -B)
ENV PYTHONDONTWRITEBYTECODE 1
#Prevents Python from buffering stdout and stderr (equivalent python -u)
ENV PYTHONUNBUFFERED 1

# add app
COPY . /opt/

RUN apt update && \
    apt -y install apt-utils tzdata locales nano libgl1 ffmpeg libsm6 libxext6 --fix-missing && \
    apt clean && apt autoclean && apt autoremove && rm -rf /var/lib/apt/lists/* && \
    ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

# install python dependenciesыы
COPY --from=builder /usr/src/app/wheels /wheels
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*

FROM python:3.11-slim
LABEL maintainer="otzslayer@gmail.com"

ARG USER_HOME=/home/mlops
ARG UTIL_PATH=utils
ARG PREPROCESSING_PATH=pipelines/continuous_training/data_preprocessing
ARG REQUIREMENTS_PATH=pipelines/continuous_training/docker

RUN groupadd --gid 1000 mlops \
    && useradd --uid 1000 --gid mlops --shell /bin/bash --create-home mlops

COPY --chown=mlops:mlops ${UTIL_PATH}/ ${USER_HOME}/utils
COPY --chown=mlops:mlops ${PREPROCESSING_PATH}/preprocessor.py \
    ${USER_HOME}/data_preprocessing/
COPY --chown=mlops:mlops ${REQUIREMENTS_PATH}/requirements.txt \
    ${USER_HOME}/

USER mlops

RUN mkdir -p ${USER_HOME}/artifacts \
    && pip install --no-cache-dir \
    -r ${USER_HOME}/requirements.txt

WORKDIR ${USER_HOME}

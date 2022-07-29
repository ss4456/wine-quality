FROM ubuntu:latest

COPY predict.py /predict.py
COPY models/ /models/

COPY requirements.txt ./requirements.txt
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip \
    && apt-get install openjdk-8-jdk-headless -y
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

COPY ValidationDataset.csv /data/ValidationDataset.csv

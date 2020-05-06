FROM ubuntu:18.04

MAINTAINER Tyler Lozano "tyler.m.lozano@gmail.com"

# We copy just the requirements.txt first to leverage Docker cache
COPY ./app/requirements.txt /app/requirements.txt

ENV STATIC_PATH /app/static

WORKDIR /app

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.7
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-setuptools \
    python3.7-dev \
    build-essential \
    python3-pip
#RUN easy_install pip==19.0.3
RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN pip3 install --no-cache-dir -r requirements.txt
#RUN pip install git+https://github.com/Maluuba/nlg-eval.git@master
#RUN nlg-eval --setup

COPY ./app /app

ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]
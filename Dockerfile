FROM ubuntu:18.04

MAINTAINER Tyler Lozano "tyler.m.lozano@gmail.com"

# copy just the requirements.txt first to leverage Docker cache
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
    python3-pip \
    git

# latest pip breaks psutils dependency, but 19 works
RUN pip3 install --upgrade pip==19.0.3
RUN pip3 install wheel
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install git+https://github.com/Maluuba/nlg-eval.git@master

# Set the locale so nlg-eval doesn't crash
RUN apt-get -y install locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

RUN nlg-eval --setup

COPY ./app /app

ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]
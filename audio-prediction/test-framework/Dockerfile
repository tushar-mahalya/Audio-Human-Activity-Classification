FROM centos:latest

RUN yum update -y
RUN yum install -y python3-pip
RUN yum install gcc openssl-devel bzip2-devel libffi-devel wget make xz-devel -y
RUN curl -O https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
RUN tar -xzf Python-3.8.1.tgz
RUN cd Python-3.8.1/ && ./configure --enable-optimizations && make install
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py --user
RUN yum install -y libsndfile

ENV FLASK_DEBUG=1 WERKZEUG_DEBUG_PIN=off

COPY . /app
WORKDIR /app

RUN pip3 install grpcio~=1.25.0 numpy tensorflow keras flask pandas mysql-connector-python librosa

EXPOSE 5000

ENTRYPOINT ["python3.8"]
CMD ["api_server.py"]

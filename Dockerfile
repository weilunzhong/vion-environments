FROM docker-registry.service.vionel:5000/tensorflow-opencv:0.7.1-gpu

RUN mkdir ~/.pip
RUN echo "[global]\ntimeout=1\nextra-index-url=http://192.168.1.41:64473\ntrusted-host=192.168.1.41" >> ~/.pip/pip.conf
RUN pip install vionel-auxiliary
RUN pip install vionel-sdk

COPY . /source_code
WORKDIR /source_code

CMD python environment_classifier.py

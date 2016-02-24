FROM b.gcr.io/tensorflow/tensorflow-devel-gpu

RUN mkdir ~/.pip
RUN echo "[global]\ntimeout=1\nextra-index-url=http://192.168.1.41:64473\ntrusted-host=192.168.1.41" >> ~/.pip/pip.conf
RUN pip install vionel-auxiliary
RUN pip install vionel-sdk

RUN apt-get update
RUN apt-get install git \
                    cmake -y
Run git clone http://github.com/Itseez/opencv.git
RUN cd opencv && mkdir build && cd build \
    && cmake .. && make -j4 && make install \
    && ldconfig

COPY . /source_code
WORKDIR /source_code

CMD python environment_classifier.py

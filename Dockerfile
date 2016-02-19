FROM b.gcr.io/tensorflow/tensorflow-devel-gpu

COPY . /source_code

WORKDIR /source_code
CMD python environment_classifier.py

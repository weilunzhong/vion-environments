FROM tensorflow/tensorflow 

COPY . /source_code

WORKDIR /source_code
CMD python environment_classifier.py

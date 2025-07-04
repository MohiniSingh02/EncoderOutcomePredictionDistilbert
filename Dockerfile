FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
FROM registry.datexis.com/tsteffek/encoderoutcomeprediction:distilbert-v1

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install transformers


WORKDIR /src
COPY . .

ENV PYTHONPATH=/src
RUN sed -i 's/from src\.model/from model/g' /src/classification_main.py


CMD ["/usr/sbin/sshd", "-D"]

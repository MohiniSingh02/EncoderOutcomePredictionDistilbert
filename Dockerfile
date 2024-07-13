FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /src
COPY src .
CMD ["/usr/sbin/sshd", "-D"]

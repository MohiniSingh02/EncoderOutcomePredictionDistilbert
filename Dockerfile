FROM registry.datexis.com/pgrundmann/pytorch-ngc:24.05

RUN git clone https://github.com/pytorch/vision.git /vision
WORKDIR /vision
RUN git checkout v0.18.1
RUN MAX_JOBS=16 pip install . --no-dependencies

WORKDIR ..
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /src
COPY src .
CMD ["/usr/sbin/sshd", "-D"]

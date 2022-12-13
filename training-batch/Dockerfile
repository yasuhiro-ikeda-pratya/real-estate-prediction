ARG s3_bucket_name=''
ARG aws_access_key_id=''
ARG aws_secret_access_key=''

FROM --platform=linux/x86_64 python:3

ARG s3_bucket_name
ARG aws_access_key_id
ARG aws_secret_access_key

COPY ./requirements.txt .

RUN pip install --upgrade pip && pip install --user awscli
RUN pip install --user -r requirements.txt

RUN mkdir /mnt/app
RUN mkdir /mnt/app/data
RUN mkdir /mnt/app/models
WORKDIR /mnt/app

ENV PATH="/root/.local/bin:${PATH}"
ENV PATH="/mnt/app:${PATH}"

ENV S3_BUCKET_NAME=${s3_bucket_name}
ENV AWS_ACCESS_KEY_ID=${aws_access_key_id}
ENV AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}

COPY ./module ./module
COPY ./scripts/train.py .
COPY ./training-batch/train .
RUN chmod +x train

CMD ["train"]
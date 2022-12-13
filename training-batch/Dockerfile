# ビルド時に引数で環境変数を与えるための設定
ARG s3_bucket_name=''
ARG aws_access_key_id=''
ARG aws_secret_access_key=''

# Macなど異なる環境でビルドする際にはplatformを明示的に指定
FROM --platform=linux/x86_64 python:3

# ビルド時に引数で環境変数を与えるための設定
ARG s3_bucket_name
ARG aws_access_key_id
ARG aws_secret_access_key

# pipでインストールするパッケージのrequirements.txtをコピー
COPY ./requirements.txt .

# pipによるインストール（AWS CLIによる操作も含まれるためインストール）
RUN pip install --upgrade pip && pip install --user awscli
RUN pip install --user -r requirements.txt

# アプリケーション実行に必要なディレクトリの作成
RUN mkdir /mnt/app
RUN mkdir /mnt/app/data
RUN mkdir /mnt/app/models
WORKDIR /mnt/app

# パスの設定
ENV PATH="/root/.local/bin:${PATH}"
ENV PATH="/mnt/app:${PATH}"

# 環境変数を引数で与えた値で設定
ENV S3_BUCKET_NAME=${s3_bucket_name}
ENV AWS_ACCESS_KEY_ID=${aws_access_key_id}
ENV AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}

# モジュール、学習スクリプト、学習スクリプト実行のシェルスクリプトをコピー
COPY ./module ./module
COPY ./scripts/train.py .
COPY ./training-batch/train .
RUN chmod +x train

CMD ["train"]
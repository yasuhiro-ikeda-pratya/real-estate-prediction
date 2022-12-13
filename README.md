# real-estate-prediction
国土交通省データを用いた地価価格の予測

## Dockerイメージのビルド
```
docker build -t training -f training-batch/Dockerfile —build-arg s3_bucket_name=$S3_BUCKET_NAME —build-arg aws_access_key_id=$AWS_ACCESS_KEY_ID —build-arg aws_secret_access_key=$AWS_SECRET_ACCESS_KEY .
docker build -t prediction -f prediction-batch/Dockerfile —build-arg s3_bucket_name=$S3_BUCKET_NAME —build-arg aws_access_key_id=$AWS_ACCESS_KEY_ID —build-arg aws_secret_access_key=$AWS_SECRET_ACCESS_KEY .
```
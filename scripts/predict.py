import os
import logging
import pickle
import click
import pandas as pd
from module.preprocessor import PreProcessor
from module.trainer import LgbTrainer

logger = logging.getLogger(__name__)


@click.command()
@click.option("--data_dir", default="./data/", type=str)
@click.option("--model_dir", default="./models/", type=str)
@click.option("--result_dir", default="./results/", type=str)
def main(data_dir, model_dir, result_dir):
    # 学習データ前処理
    logger.info("Preprocess test data")
    df_test = pd.read_csv(os.path.join(data_dir, "df_test.csv"))
    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    X_test = preprocessor.preprocess_transform(df_test)

    # モデルによる予測
    logger.info("Predict model")
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        classifier = pickle.load(f)
    y_pred = classifier.lgb_model.predict(X_test)

    df_pred = pd.DataFrame(y_pred)
    df_pred.index = X_test.index
    df_pred.columns = ["予測値"]
    df_pred.to_csv(os.path.join(result_dir, "df_pred.csv"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

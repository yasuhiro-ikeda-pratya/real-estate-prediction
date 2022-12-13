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
def main(data_dir, model_dir):
    # 学習データ前処理
    logger.info("Preprocess data")
    df_train = pd.read_csv(os.path.join(data_dir, "df_train.csv"))
    preprocessor = PreProcessor()
    X_train, y_train = preprocessor.preprocess_fit(df_train)
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    # モデル学習
    logger.info("Train model")
    classifier = LgbTrainer()
    classifier.train(X_train, y_train)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(classifier, f)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

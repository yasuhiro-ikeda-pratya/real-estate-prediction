import os
import pickle
import lightgbm as lgb


class LgbTrainer(object):
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        # lightgbm用データセットの作成
        lgb_train = lgb.Dataset(X_train, label=y_train)

        # lightgbm用パラメータセット
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
        }

        self.lgb_model = lgb.train(
            lgb_params,
            lgb_train,
        )

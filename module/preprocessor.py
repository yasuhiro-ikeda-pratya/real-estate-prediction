import os
import pandas as pd
import numpy as np
import pickle

from .params import DROP_COLS, FEATURE_COLS, TARGET_COL, CATEGORICAL_COLS


class PreProcessor(object):
    def __init__(self):
        self.d_mean_values = {}


    # 建築年
    def wareki_to_seireki(self, x):
        if x != x:  # NaNの判定
            return np.nan
        elif x[:2] == "戦前":
            return 1925  # 大正14年。昭和元年の一年前
        elif x[:2] == "昭和":
            base_year = 1925  # 昭和元年の一年前
            yy = float(x.strip("昭和 年"))  # stripで文字列から文字を削除。スペース区切りで複数の文字を指定可能
            return base_year + yy
        elif x[:2] == "平成":
            base_year = 1988  # 平成元年の一年前
            yy = float(x.strip("平成 年"))
            return base_year + yy
        elif x[:2] == "令和":
            base_year = 2018  # 令和元年の一年前
            yy = float(x.strip("令和 年"))
            return base_year + yy

    def preprocess_fit(self, df_train):
        df_train[TARGET_COL] = (
            df_train["取引価格（総額）"] >= df_train["取引価格（総額）"].median()
        ).astype(int)

        self.torihiki_kakaku_median = df_train["取引価格（総額）"].median()

        df_train = df_train.drop(DROP_COLS, axis=1).copy()

        # 最寄駅：距離（分）
        df_train["最寄駅：距離（分）"] = (
            df_train["最寄駅：距離（分）"]
            .replace({"30分?60分": "45", "1H?1H30": "75", "1H30?2H": "105", "2H?": "120"})
            .astype(float)
        )
        df_train["最寄駅：距離（分）"] = df_train["最寄駅：距離（分）"].fillna(
            df_train["最寄駅：距離（分）"].mean()
        )
        self.d_mean_values["最寄駅：距離（分）"] = df_train["最寄駅：距離（分）"].mean()

        # 坪単価
        df_train["坪単価"] = df_train["坪単価"].fillna(df_train["坪単価"].mean())
        self.d_mean_values["坪単価"] = df_train["坪単価"].mean()

        # 面積（㎡）
        df_train["面積（㎡）"] = (
            df_train["面積（㎡）"]
            .replace({"2000㎡以上": "2000", "5000㎡以上": "5000"})
            .astype(float)
        )
        df_train["面積（㎡）"] = df_train["面積（㎡）"].fillna(df_train["面積（㎡）"].mean())
        self.d_mean_values["面積（㎡）"] = df_train["面積（㎡）"].mean()

        # 取引価格（㎡単価）
        df_train["取引価格（㎡単価）"] = df_train["取引価格（㎡単価）"].fillna(
            df_train["取引価格（㎡単価）"].mean()
        )
        self.d_mean_values["取引価格（㎡単価）"] = df_train["取引価格（㎡単価）"].mean()

        # 間口
        df_train["間口"] = df_train["間口"].replace("50.0m以上", "50.0").astype(float).copy()
        df_train["間口"] = df_train["間口"].fillna(df_train["間口"].mean())
        self.d_mean_values["間口"] = df_train["間口"].mean()

        # 延床面積（㎡）
        df_train["延床面積（㎡）"] = (
            df_train["延床面積（㎡）"].replace("2000㎡以上", "2000").astype(float).copy()
        )
        df_train["延床面積（㎡）"] = df_train["延床面積（㎡）"].fillna(df_train["延床面積（㎡）"].mean())
        self.d_mean_values["延床面積（㎡）"] = df_train["延床面積（㎡）"].mean()

        # 建築年
        df_train["建築年"] = df_train["建築年"].apply(self.wareki_to_seireki)
        df_train["建築年"] = df_train["建築年"].fillna(df_train["建築年"].mean())
        self.d_mean_values["建築年"] = df_train["建築年"].mean()

        # 前面道路：幅員（ｍ）
        df_train["前面道路：幅員（ｍ）"] = df_train["前面道路：幅員（ｍ）"].fillna(
            df_train["前面道路：幅員（ｍ）"].mean()
        )
        self.d_mean_values["前面道路：幅員（ｍ）"] = df_train["前面道路：幅員（ｍ）"].mean()

        # 建ぺい率（％）
        df_train["建ぺい率（％）"] = df_train["建ぺい率（％）"].fillna(df_train["建ぺい率（％）"].mean())
        self.d_mean_values["建ぺい率（％）"] = df_train["建ぺい率（％）"].mean()

        # 容積率（％）
        df_train["容積率（％）"] = df_train["容積率（％）"].fillna(df_train["容積率（％）"].mean())
        self.d_mean_values["容積率（％）"] = df_train["容積率（％）"].mean()

        X_train = df_train[FEATURE_COLS].copy()
        y_train = df_train[TARGET_COL].copy()

        for col in CATEGORICAL_COLS:
            X_train[col] = X_train[col].astype("category")

        return X_train, y_train

    def preprocess_transform(self, df_test):
        df_test = df_test.drop(DROP_COLS, axis=1).copy()

        # 最寄駅：距離（分）
        df_test["最寄駅：距離（分）"] = (
            df_test["最寄駅：距離（分）"]
            .replace({"30分?60分": "45", "1H?1H30": "75", "1H30?2H": "105", "2H?": "120"})
            .astype(float)
        )
        df_test["最寄駅：距離（分）"] = df_test["最寄駅：距離（分）"].fillna(
            self.d_mean_values["最寄駅：距離（分）"]
        )

        # 坪単価
        df_test["坪単価"] = df_test["坪単価"].fillna(self.d_mean_values["坪単価"].mean())

        # 面積（㎡）
        df_test["面積（㎡）"] = (
            df_test["面積（㎡）"]
            .replace({"2000㎡以上": "2000", "5000㎡以上": "5000"})
            .astype(float)
        )
        df_test["面積（㎡）"] = df_test["面積（㎡）"].fillna(self.d_mean_values["面積（㎡）"])

        # 取引価格（㎡単価）
        df_test["取引価格（㎡単価）"] = df_test["取引価格（㎡単価）"].fillna(
            self.d_mean_values["取引価格（㎡単価）"]
        )

        # 間口
        df_test["間口"] = df_test["間口"].replace("50.0m以上", "50.0").astype(float).copy()
        df_test["間口"] = df_test["間口"].fillna(self.d_mean_values["間口"])

        # 延床面積（㎡）
        df_test["間口"] = df_test["間口"].replace("50.0m以上", "50.0").astype(float).copy()
        df_test["間口"] = df_test["間口"].fillna(self.d_mean_values["間口"].mean())

        # 延床面積（㎡）
        df_test["延床面積（㎡）"] = (
            df_test["延床面積（㎡）"].replace("2000㎡以上", "2000").astype(float).copy()
        )
        df_test["延床面積（㎡）"] = df_test["延床面積（㎡）"].fillna(
            self.d_mean_values["延床面積（㎡）"].mean()
        )

        df_test["建築年"] = df_test["建築年"].apply(self.wareki_to_seireki)
        df_test["建築年"] = df_test["建築年"].fillna(self.d_mean_values["建築年"])

        # 前面道路：幅員（ｍ）
        df_test["前面道路：幅員（ｍ）"] = df_test["前面道路：幅員（ｍ）"].fillna(
            self.d_mean_values["前面道路：幅員（ｍ）"]
        )

        # 建ぺい率（％）
        df_test["建ぺい率（％）"] = df_test["建ぺい率（％）"].fillna(
            self.d_mean_values["建ぺい率（％）"].mean()
        )

        # 容積率（％）
        df_test["容積率（％）"] = df_test["容積率（％）"].fillna(
            self.d_mean_values["容積率（％）"].mean()
        )

        # 容積率（％）
        df_test["容積率（％）"] = df_test["容積率（％）"].fillna(self.d_mean_values["容積率（％）"])

        X_test = df_test[FEATURE_COLS]

        for col in CATEGORICAL_COLS:
            X_test[col] = X_test[col].astype("category")

        return X_test

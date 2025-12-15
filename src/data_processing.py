from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TransactionDatetimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col: str = "TransactionStartTime", drop_original: bool = True):
        self.datetime_col = datetime_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TransactionDatetimeFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.datetime_col in X_out.columns:
            dt = pd.to_datetime(X_out[self.datetime_col], errors="coerce", utc=True)
            X_out["tx_hour"] = dt.dt.hour.astype("Int64")
            X_out["tx_day"] = dt.dt.day.astype("Int64")
            X_out["tx_month"] = dt.dt.month.astype("Int64")
            X_out["tx_year"] = dt.dt.year.astype("Int64")
            X_out["tx_dayofweek"] = dt.dt.dayofweek.astype("Int64")
            X_out["tx_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            if self.drop_original:
                X_out = X_out.drop(columns=[self.datetime_col])
        return X_out


class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        customer_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
        provider_col: str = "ProviderId",
        product_col: str = "ProductId",
        product_category_col: str = "ProductCategory",
        channel_col: str = "ChannelId",
    ):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.provider_col = provider_col
        self.product_col = product_col
        self.product_category_col = product_category_col
        self.channel_col = channel_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CustomerAggregateFeatures":
        df = X.copy()

        if self.amount_col in df.columns:
            df[self.amount_col] = pd.to_numeric(df[self.amount_col], errors="coerce")
        if self.value_col in df.columns:
            df[self.value_col] = pd.to_numeric(df[self.value_col], errors="coerce")

        agg_parts: list[pd.DataFrame] = []

        if self.customer_col in df.columns:
            if self.amount_col in df.columns:
                amount = df[[self.customer_col, self.amount_col]].copy()
                amount["amount_pos"] = amount[self.amount_col].where(amount[self.amount_col] > 0, 0)
                amount["amount_neg"] = (-amount[self.amount_col]).where(amount[self.amount_col] < 0, 0)
                g = amount.groupby(self.customer_col, dropna=False)
                a = g[self.amount_col].agg(["sum", "mean", "std", "min", "max", "count"]).rename(
                    columns={
                        "sum": "cust_amount_sum",
                        "mean": "cust_amount_mean",
                        "std": "cust_amount_std",
                        "min": "cust_amount_min",
                        "max": "cust_amount_max",
                        "count": "cust_txn_count",
                    }
                )
                ap = g["amount_pos"].sum().rename("cust_amount_pos_sum")
                an = g["amount_neg"].sum().rename("cust_amount_neg_sum")
                agg_parts.append(pd.concat([a, ap, an], axis=1))

            if self.value_col in df.columns:
                g = df.groupby(self.customer_col, dropna=False)[self.value_col]
                v = g.agg(["sum", "mean", "std", "min", "max"]).rename(
                    columns={
                        "sum": "cust_value_sum",
                        "mean": "cust_value_mean",
                        "std": "cust_value_std",
                        "min": "cust_value_min",
                        "max": "cust_value_max",
                    }
                )
                agg_parts.append(v)

            for col, out_name in [
                (self.provider_col, "cust_provider_nunique"),
                (self.product_col, "cust_product_nunique"),
                (self.product_category_col, "cust_product_category_nunique"),
                (self.channel_col, "cust_channel_nunique"),
            ]:
                if col in df.columns:
                    n = df.groupby(self.customer_col, dropna=False)[col].nunique().rename(out_name)
                    agg_parts.append(n.to_frame())

        if agg_parts:
            agg = pd.concat(agg_parts, axis=1).reset_index()
        else:
            agg = pd.DataFrame(columns=[self.customer_col])

        self.customer_agg_ = agg
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        if self.customer_col in X_out.columns and hasattr(self, "customer_agg_"):
            X_out = X_out.merge(self.customer_agg_, on=self.customer_col, how="left")

            agg_cols = [c for c in self.customer_agg_.columns if c != self.customer_col]
            for c in agg_cols:
                if c in X_out.columns:
                    X_out[c] = pd.to_numeric(X_out[c], errors="coerce")
            if agg_cols:
                X_out[agg_cols] = X_out[agg_cols].fillna(0.0)

        return X_out


class WOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, include: list[str] | None = None, exclude: list[str] | None = None):
        self.include = include
        self.exclude = exclude

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEEncoder":
        try:
            from xverse.transformer import WOE
        except Exception as e:
            raise ImportError("xverse is required for WOE. Install it via requirements.txt") from e

        df = X.copy()
        obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]

        cols = obj_cols
        if self.include is not None:
            cols = [c for c in cols if c in self.include]
        if self.exclude is not None:
            cols = [c for c in cols if c not in self.exclude]

        self.woe_cols_ = cols
        self.woe_ = WOE()
        self.woe_.fit(df[self.woe_cols_], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not hasattr(self, "woe_cols_") or not self.woe_cols_:
            return df
        woe_df = self.woe_.transform(df[self.woe_cols_])
        df = df.drop(columns=self.woe_cols_)
        return pd.concat([df, woe_df], axis=1)


def make_preprocess_pipeline(
    customer_col: str = "CustomerId",
    amount_col: str = "Amount",
    value_col: str = "Value",
    datetime_col: str = "TransactionStartTime",
    use_woe: bool = False,
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    col_xf = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps: list[tuple[str, object]] = [
        ("tx_time", TransactionDatetimeFeatures(datetime_col=datetime_col, drop_original=True)),
        (
            "cust_agg",
            CustomerAggregateFeatures(
                customer_col=customer_col,
                amount_col=amount_col,
                value_col=value_col,
            ),
        ),
    ]

    if use_woe:
        steps.append(("woe", WOEEncoder(exclude=[customer_col])))

    steps.append(("encode_scale", col_xf))
    return Pipeline(steps=steps)


def build_features(df: pd.DataFrame) -> np.ndarray:
    pipe = make_preprocess_pipeline(use_woe=False)
    return pipe.fit_transform(df)


def build_proxy_target(df: pd.DataFrame, delinquency_column: str) -> pd.Series:
    y = pd.to_numeric(df[delinquency_column], errors="coerce").fillna(0)
    return (y >= 90).astype(int)

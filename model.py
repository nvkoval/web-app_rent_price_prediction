import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

import joblib


data = pd.read_csv('data/real_estate_last.csv')
black_list = ['id', 'price']
features = [col for col in data.columns if col not in black_list]

X = data[features]
y = np.log1p(data['price']).values

transformer = ColumnTransformer(
    [('OneHot', OneHotEncoder(sparse_output=True), ['district']),
     ('labelEnc', OrdinalEncoder(handle_unknown='use_encoded_value',
                                 encoded_missing_value=-1, unknown_value=-1),
                                 ['subway'])],
    remainder='passthrough',
    verbose_feature_names_out=False
)

parameters_lgb = {
    'num_leaves': 3,
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 805,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'random_state': 24,
    'force_col_wise': True,
    'min_child_samples': 100,
    'objective': 'regression',
}

model_1 = lgb.LGBMRegressor(**parameters_lgb)

lgb_model = Pipeline([
    ('transform', transformer),
    ('regr', model_1)
])

lgb_model.fit(X, y)
joblib.dump(lgb_model, 'data/lgb_model.sav')

parameters_xgb = {
    "objective": "reg:squarederror",
    "eval_metric": "rmsle",
    "eta": 0.3,
    "verbosity": 1,
    "seed": 24,
    "tree_method": "hist",
    "grow_policy": "lossguide",

    "max_depth": 5,
    "max_leaves": 3,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    'n_estimators': 1000
}

model_2 = xgb.XGBRegressor(**parameters_xgb)

xgb_model = Pipeline([
    ('transform', transformer),
    ('regr', model_2)
])

xgb_model.fit(X, y)
joblib.dump(xgb_model, 'data/xgb_model.sav')


parameters_ctb = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "random_seed": 24,
    "od_wait": 30,
    "od_type": "Iter",
    "thread_count": 4,
    "verbose": False
}

model_3 = ctb.CatBoostRegressor(**parameters_ctb)

ctb_model = Pipeline([
    ('transform', transformer),
    ('regr', model_3)
])

ctb_model.fit(X, y)
joblib.dump(ctb_model, 'data/ctb_model.sav')

model_4 = LinearRegression()

lr_model = Pipeline([
    ('transform', transformer),
    ('regr', model_4)
])

lr_model.fit(X.fillna('unknown'), y)
joblib.dump(lr_model, 'data/lr_model.sav')


model_5 = ExtraTreesRegressor(
    max_depth=10, max_features=0.6,
    bootstrap=True, 
    random_state=24, max_samples=0.8)

etr_model = Pipeline([
    ('transform', transformer),
    ('regr', model_5)
])

etr_model.fit(X.fillna('unknown'), y)
joblib.dump(etr_model, 'data/etr_model.sav')

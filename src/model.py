import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from mapie.regression import MapieRegressor

import joblib


data = pd.read_csv('data/real_estate_last.csv')
black_list = ['id', 'price']
features = [col for col in data.columns if col not in black_list]

X = data[features]
y = np.log1p(data['price']).values

transformer = ColumnTransformer(
    [('labelEnc', OrdinalEncoder(handle_unknown='use_encoded_value',
                                 encoded_missing_value=-1,
                                 unknown_value=-1), ['subway', 'district'])],
    remainder='passthrough',
    verbose_feature_names_out=False
)

parameters_lgb = {
    'num_leaves': 3,
    'max_depth': 5,
    'learning_rate': 0.01,
    'n_estimators': 4925,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'random_state': 24,
    'force_col_wise': True,
    'min_child_samples': 10,
    'objective': 'regression',
}

regr_lgb = lgb.LGBMRegressor(**parameters_lgb)

lgb_model = Pipeline([
    ('transform', transformer),
    ('regr', regr_lgb)
])

lgb_model.fit(X, y)
joblib.dump(lgb_model, 'data/lgb_model.sav')


mapie_reg_lgb = MapieRegressor(estimator=lgb_model, cv=3, agg_function='median')
mapie_reg_lgb.fit(X, y)
joblib.dump(mapie_reg_lgb, 'data/mapie_reg_lgb.sav')

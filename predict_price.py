import joblib
from matplotlib import colormaps
from eli5 import formatters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mapie.regression import MapieRegressor

from model import X, y


def predict(X_test, model_file: str):
    model = joblib.load(f"data/{model_file}")
    y_pred = np.expm1(model.predict(X_test))
    y_pred[y_pred < 0] = 0
    return y_pred


def get_explain(X_test, features):
    model = joblib.load('lgb_model.sav')
    cm = colormaps['cividis']
    test = pd.DataFrame(model[0].transform(X_test),
                        columns= model[0].get_feature_names_out())
    df = formatters.as_dataframe.explain_prediction_df(
        model[1], test.astype(float),
        top=(10, 10),
        feature_names=model[0].get_feature_names_out())
    df.loc[ 1:, 'weight'] = np.expm1(df['weight'])*np.expm1(df.loc[0, 'weight'])
    df = (df
            .loc[1:, ['weight', 'feature']]
            .rename(columns={'weight': 'вплив на ціну',
                             'feature': 'характеристика'}))
    df_formatted = (df
            .style.background_gradient(cmap=cm)
            .format({'вплив на ціну': '{0:+.2f}'}))
    
    fig, ax = plt.subplots(figsize=(10,6))
    plt.yticks(ticks=np.arange(df.shape[0]), labels=df['характеристика'], fontsize=10)
    ax.barh(df['характеристика'], df['вплив на ціну'], color='DarkCyan')

    return df_formatted, fig


def conf_interval(models, X_test):
    #res = []
    estimators = []
    predictions = pd.DataFrame(index=[0])
    for i, name_model in enumerate(models):
        model = joblib.load(f"{name_model}.sav")
        
        estimators.append(model)
        mapie_reg = MapieRegressor(estimator=model, cv=3, agg_function='median')
        mapie_reg.fit(X, y)

        y_pred_mapie, y_pis = mapie_reg.predict(X_test, ensemble=True, alpha=0.3)

        predictions[f"point prediction_{i}"] = y_pred_mapie
        predictions[f"lower_{i}"] = y_pis.reshape(-1,2)[:,0]
        predictions[f"upper_{i}"] = y_pis.reshape(-1,2)[:,1]
    predictions['prediction'] = np.expm1(predictions.filter(regex='point').mean(axis=1))
    predictions['lower'] = np.expm1(predictions.filter(regex='lower').mean(axis=1))
    predictions['upper'] = np.expm1(predictions.filter(regex='upper').mean(axis=1))
    predictions['error_lower'] = predictions['prediction'] - predictions['lower']
    predictions['error_upper'] = predictions['upper'] - predictions['prediction']
    predictions['error'] = predictions.filter(regex='error').mean(axis=1)

    return predictions['error']
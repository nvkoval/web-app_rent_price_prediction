import joblib
from matplotlib import colormaps
from eli5 import formatters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

columns_dict = {
    'rooms': 'кількість кімнат',
    'floor': 'поверх',
    'full_area': 'загальна площа',
    'living_area': 'житлова площа',
    'kitchen_area': 'площа кухні',
    'num_storeys': 'кількість поверхів у будинку',
    'floor_other': 'поверх інші',
    'full_area_log': 'логарифм загальнаої площі',
    'living_area_log': 'логарифм житлової площі',
    'kitchen_area_log': 'логарифм площі кухні',
    'num_storeys_other': 'кількість поверхів інші',
    'first_rental': 'перша здача',
    'last_floor': 'останній поверх',
    'first_floor': 'перший поверх',
    'district': 'район',
    'typical_panel': 'типова панель',
    'ukrainian_panel': 'українська панель',
    'old_panel': 'стара панель',
    'concrete_monolithic': 'бетонно-монолітний',
    'old_brick': 'стара цегла',
    'ukrainian_brick': 'українська цегла',
    'gas_block': 'газоблок',
    'Stalinka': 'сталінка',
    'pre_revolutionary': 'дореволюційний',
    'adjacent_separate': 'суміжно-роздільна',
    'multilevel': 'багаторівнева',
    'kitchen_living_room': 'кухня-вітальня',
    'penthouse': 'пентхаус',
    'studio': 'студія',
    'free_planning': 'вільне планування',
    'adjacent': 'суміжна',
    'separate': 'роздільна',
    'eurorenovation': 'євроремонт',
    'repair_in_progress': 'незавершений ремонт',
    'designer_renovation': 'дизайнерський ремонт',
    'needs_repairs': 'потрібен ремонт',
    'satisfactory_condition': 'задовільний стан',
    'good_condition': 'хороший стан',
    'excellent_condition': 'чудовий стан',
    'safe': 'сейф',
    'shower_cabin': 'душова кабіна',
    'wardrobe': 'шафа',
    'TV': 'телевізор',
    'hair_dryer': 'фен',
    'dishes': 'посуд',
    'satellite_TV': 'супутникове ТБ',
    'DVD_player': 'DVD програвач',
    'washing_machine': 'пральна машина',
    'fireplace': 'камін',
    'dishwashers': 'посудомийна машина',
    'alarms': 'сигналізація',
    'bed': 'ліжко',
    'counters': 'лічильники',
    'air_conditioning': 'кондиціонер',
    'refrigerator': 'холодильник',
    'jacuzzi': 'джакузі',
    'microwave': 'мікрохвильовка',
    'iron': 'праска',
    'cable_TV': 'кабельне ТБ',
    'district_lat': 'широта району',
    'district_lon': 'довгота району',
    'lat': 'широта',
    'lon': 'довгота',
    'subway': 'метро',
    'min_dist_to_subway': 'найменша відстань до метро',
    'dist_to_center': 'відстань до центру',
    'dist_to_center|full_area_log': 'відстань до центру | логарифм повної площі',
    'dist_to_center|full_area_log|kitchen_area': 'відстань до центру | логарифм повної площі | площа кухні',
    'full_area|kitchen_area_log': 'загальна площа | логарифм площі кухні'
}


def predict(X_test, model_name: str):
    model = joblib.load(f"data/{model_name}.sav")
    y_pred = np.expm1(model.predict(X_test))
    y_pred[y_pred < 0] = 0
    return y_pred


def get_explain(model_name, X_test):
    model = joblib.load(f"data/{model_name}.sav")
    cm = colormaps['cividis']
    test = pd.DataFrame(model[0].transform(X_test),
                        columns=model[0].get_feature_names_out())
    df = formatters.as_dataframe.explain_prediction_df(
        model[1], test.astype(float),
        top=(10, 10),
        feature_names=[columns_dict[feat] if feat in columns_dict.keys() else feat
                       for feat in model[0].get_feature_names_out()]
        )
    df.loc[1:, 'weight'] = np.expm1(df['weight'])*np.expm1(df.loc[0, 'weight'])
    df = (df
          .loc[1:, ['weight', 'feature']]
          .rename(columns={'weight': 'вплив на ціну',
                           'feature': 'характеристика'}))
    df_formatted = (df
                    .style.background_gradient(cmap=cm)
                    .format({'вплив на ціну': '{0:+.2f}'}))

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.yticks(ticks=np.arange(df.shape[0]),
               labels=df['характеристика'], fontsize=10)
    ax.barh(df['характеристика'], df['вплив на ціну'], color='DarkCyan')

    return df_formatted, fig


def conf_interval(mapie_reg_name, X_test):
    predictions = pd.DataFrame(index=[0])

    mapie_reg = joblib.load(f"data/{mapie_reg_name}.sav")

    y_pred_mapie, y_pis = mapie_reg.predict(X_test, ensemble=True, alpha=0.3)

    predictions['prediction'] = np.expm1(y_pred_mapie)
    predictions['lower'] = np.expm1(y_pis.reshape(-1, 2)[:, 0])
    predictions['upper'] = np.expm1(y_pis.reshape(-1, 2)[:, 1])
    predictions['error_lower'] = (predictions['prediction']
                                  - predictions['lower'])
    predictions['error_upper'] = (predictions['upper']
                                  - predictions['prediction'])
    predictions['error'] = predictions.filter(regex='error').mean(axis=1)

    return predictions['error']

import streamlit as st

st.set_page_config(
   page_title="Predict Rent Price",
   layout="wide",
   initial_sidebar_state="expanded",
)

import pandas as pd

from src.predict_price import get_explain, predict, conf_interval
from src.utils import fe

st.title('Оцінка вартості оренди квартири у Києві')

st.write('---')

st.sidebar.title("Заповніть данні по квартирі")

address = st.sidebar.text_input('Адреса')

district = st.sidebar.selectbox(
    'Район',
    options=['Голосіївський', 'Дарницький', 'Деснянський',
             'Дніпровський', 'Оболонський', 'Печерський',
             'Подільський', 'Святошинський', 'Солом\'янський',
             'Шевченківський'])

full_area = st.sidebar.number_input('Загальна площа', step=0.1)

living_area = st.sidebar.number_input('Житлова площа', step=0.1)

kitchen_area = st.sidebar.number_input('Площа кухні', step=0.1)

rooms = st.sidebar.number_input('Кількість кімнат', min_value=1, step=1)

floor = st.sidebar.number_input('Поверх', min_value=1, step=1)

num_storeys = st.sidebar.number_input(
    'Кількість поверхів у будинку',
    min_value=1, step=1)

first_rental = st.sidebar.checkbox('Перша здача')

building_details = st.sidebar.selectbox(
    'Характеристики будівлі',
    options=['типова панель', 'українська панель', 'стара панель',
             'стара цегла', 'українська цегла', 'газоблок',
             'бетонно-монолітний', 'сталінка', 'дореволюційний'])

features_planning = st.sidebar.multiselect(
    'Особливості планування',
    options=['суміжна', 'суміжно-роздільна', 'роздільна',
             'кухня-вітальня', 'студія', 'вільне планування',
             'багаторівнева', 'пентхаус'])

repair_state = st.sidebar.selectbox(
   'Стан ремонту',
   options=['євроремонт', 'дизайнерський ремонт',
            'незавершений ремонт', 'потрібен ремонт'])

condition = st.sidebar.selectbox(
   'загальний стан квартири',
   options=['задовільний стан', 'хороший стан', 'чудовий стан'])

facilities = st.sidebar.multiselect(
    'В квартирі є',
    options=['ліжко', 'шафа', 'посуд', 'холодильник', 'мікрохвильовка',
             'посудомийна машина', 'пральна машина', 'праска', 'фен',
             'телевізор', 'кабельне ТБ', 'DVD програвач', 'супутникове ТБ',
             'душова кабіна', 'джакузі', 'кондиціонер', 'лічильники',
             'камін', 'сейф', 'сигналізація'])

input_data = {
    'address': address,
    'rooms': rooms,
    'floor': floor,
    'district': district,
    'full_area': full_area,
    'living_area': living_area,
    'kitchen_area': kitchen_area,
    'num_storeys': num_storeys,
    'first_rental': first_rental,
    'building_details': building_details,
    'features_planning': str([x for x in features_planning]),
    'repair_state': repair_state,
    'condition': condition,
    'facilities': str([x for x in facilities])
}


if st.sidebar.button('Оцінити'):
    df = pd.DataFrame(input_data, index=[0])
    X_test = fe(df)

    cost = predict(X_test, 'lgb_model')[0]

    error = conf_interval('mapie_reg_lgb', X_test)[0]
    st.subheader(f"{int(cost)} грн. ± {int(error)}")

    st.text('Що вплинуло на формування ціни:')

    df_formatted, fig = get_explain('lgb_model', X_test)
    st.dataframe(df_formatted, width=400)

    st.text('')

    st.pyplot(fig)

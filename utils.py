import pandas as pd
import numpy as np
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


@st.cache_data
def read_data(file):
    return pd.read_csv(f'data/{file}')


df_subway_location = read_data('subway_location.csv')

df_district_location = read_data('district_location.csv')

geolocator = Nominatim(user_agent="my_request")


def get_location(address):
    loc = address + ', Київ'
    location = geolocator.geocode(loc)
    if location:
        lat = location.latitude
        lon = location.longitude
    else:
        lat = None
        lon = None
    return (lat, lon)


def get_min_dist_to_subway(location):
    if location:
        list_dist = []
        for station in df_subway_location['subway']:
            idx = (df_subway_location
                   .index[df_subway_location['subway'] == station]
                   .values[0])
            subway_loc = (df_subway_location.at[idx, 'subway_lat'],
                          df_subway_location.at[idx, 'subway_lon'])
            list_dist.append(geodesic(location, subway_loc).kilometers)
        return min(list_dist)
    else:
        return None


centre_loc = (50.450555, 30.5206892)


def get_dist_to_center(location):
    if location:
        dist = geodesic(location, centre_loc).kilometers
        return dist
    else:
        return None

def fe(df):
    test = pd.DataFrame(index=[0])

    test['rooms'] = df['rooms']
    test['floor'] = df['floor']
    test['full_area'] = df['full_area']
    test['living_area'] = df['living_area']
    test['kitchen_area'] = df['kitchen_area']
    test['num_storeys'] = df['num_storeys']

    test['floor_other'] = test['floor'].apply(lambda x: 30 if x > 26 else x)

    test['full_area_log'] = np.log1p(test['full_area'])

    test['living_area_log'] = np.log1p(test['living_area'])

    test['kitchen_area_log'] = np.log1p(df['kitchen_area'])

    test['num_storeys_other'] = (test['num_storeys']
                                 .apply(lambda x: 40 if x > 35 else x))
    test['first_rental'] = df['first_rental']
    test['last_floor'] = (test['floor'] == test['num_storeys']).astype(int)
    test['first_floor'] = (test['floor'] == 1).astype(int)

    df['location'] = df['address'].apply(get_location)
    test['lat'] = df['address'].apply(lambda x: get_location(x)[0])
    test['lon'] = df['address'].apply(lambda x: get_location(x)[1])
    
    test['min_dist_to_subway'] = df['location'].apply(get_min_dist_to_subway)
    test['dist_to_center'] = df['location'].apply(get_dist_to_center)

    test['typical_panel'] = (df['building_details']
                             .apply(lambda x: 'типова панель' in x))
    test['ukrainian_panel'] = (df['building_details']
                               .apply(lambda x: 'українська панель' in x))
    test['old_panel'] = (df['building_details']
                         .apply(lambda x: 'стара панель' in x))
    test['concrete_monolithic'] = (df['building_details']
                                   .apply(lambda x: 'бетонно-монолітний' in x))
    test['old_brick'] = (df['building_details']
                         .apply(lambda x: 'стара цегла' in x))
    test['ukrainian_brick'] = (df['building_details']
                               .apply(lambda x: 'українська цегла' in x))
    test['gas_block'] = (df['building_details']
                         .apply(lambda x: 'газоблок' in x))
    test['Stalinka'] = (df['building_details']
                        .apply(lambda x: 'сталінка' in x))
    test['pre_revolutionary'] = (df['building_details']
                                 .apply(lambda x: 'дореволюційний' in x))

    test['adjacent_separate'] = (df['features_planning']
                                 .apply(lambda x: 'суміжно-роздільна' in x))
    test['multilevel'] = (df['features_planning']
                          .apply(lambda x: 'багаторівнева' in x))
    test['kitchen_living_room'] = (df['features_planning']
                                   .apply(lambda x: 'кухня-вітальня' in x))
    test['penthouse'] = (df['features_planning']
                         .apply(lambda x: 'пентхаус' in x))
    test['studio'] = (df['features_planning']
                      .apply(lambda x: 'студія' in x))
    test['free_planning'] = (df['features_planning']
                             .apply(lambda x: 'вільне планування' in x))
    test['adjacent'] = (df['features_planning']
                        .apply(lambda x: 'суміжна' in x))
    test['separate'] = (df['features_planning']
                        .apply(lambda x: 'роздільна' in x))

    test['eurorenovation'] = (df['repair_state']
                              .apply(lambda x: 'євроремонт' in x))
    test['repair_in_progress'] = (df['repair_state']
                                  .apply(lambda x: 'незавершений ремонт' in x))
    test['designer_renovation'] = (df['repair_state']
                                   .apply(lambda x: 'дизайнерський ремонт' in x)
)
    test['satisfactory_condition'] = (df['condition']
                                      .apply(lambda x: 'задовільний стан' in x))
    test['good_condition'] = df['condition'].apply(lambda x: 'хороший стан' in x)
    test['excellent_condition'] = (df['condition']
                                   .apply(lambda x: 'чудовий стан' in x))

    test['safe'] = df['facilities'].apply(lambda x: 'сейф' in x)
    test['shower_cabin'] = df['facilities'].apply(lambda x: 'душова кабіна' in x)
    test['wardrobe'] = df['facilities'].apply(lambda x: 'шафа' in x)
    test['TV'] = df['facilities'].apply(lambda x: 'телевізор' in x)
    test['hair_dryer'] = df['facilities'].apply(lambda x: 'фен' in x)
    test['dishes'] = df['facilities'].apply(lambda x: 'посуд' in x)
    test['satellite_TV'] = df['facilities'].apply(lambda x: 'супутникове ТБ' in x)
    test['DVD_player'] = df['facilities'].apply(lambda x: 'DVD програвач' in x)
    test['washing_machine'] = df['facilities'].apply(lambda x: 'пральна машина' in x)
    test['fireplace'] = df['facilities'].apply(lambda x: 'камін' in x)
    test['dishwashers'] = df['facilities'].apply(lambda x: 'посудомийна машина' in x)
    test['alarms'] = df['facilities'].apply(lambda x: 'сигналізація' in x)
    test['bed'] = df['facilities'].apply(lambda x: 'ліжко' in x)
    test['counters'] = df['facilities'].apply(lambda x: 'лічильники' in x)
    test['air_conditioning'] = df['facilities'].apply(lambda x: 'кондиціонер' in x)
    test['refrigerator'] = df['facilities'].apply(lambda x: 'холодильник' in x)
    test['jacuzzi'] = df['facilities'].apply(lambda x: 'джакузі' in x)
    test['microwave'] = df['facilities'].apply(lambda x: 'мікрохвильовка' in x)
    test['iron'] = df['facilities'].apply(lambda x: 'праска' in x)
    test['cable_TV'] = df['facilities'].apply(lambda x: 'кабельне ТБ' in x)

    test['district'] = df['district']
    test['subway'] = df['subway'].apply(lambda x: x.lower())

    test = test.merge(df_district_location, how='left', on='district')

    test['dist_to_center|full_area_log'] = (test['dist_to_center']
                                            * test['full_area_log'])

    test['dist_to_center|full_area_log|kitchen_area'] = (test['dist_to_center'] 
                                                         * test['full_area']
                                                         * test['kitchen_area'])
    
    test['full_area|kitchen_area_log'] = (test['full_area']
                                          / test['kitchen_area_log'])
    
    return test

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


df_subway_location = pd.read_csv('data/subway_location.csv')

df_district_location = pd.read_csv('data/district_location.csv')

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

    test['typical_panel'] = df['building_details'].str.contains('типова панель')
    test['ukrainian_panel'] = df['building_details'].str.contains('українська панель')
    test['old_panel'] = df['building_details'].str.contains('стара панель')
    test['concrete_monolithic'] = df['building_details'].str.contains('бетонно-монолітний')
    test['old_brick'] = df['building_details'].str.contains('стара цегла')
    test['ukrainian_brick'] = df['building_details'].str.contains('українська цегла')
    test['gas_block'] = df['building_details'].str.contains('газоблок')
    test['Stalinka'] = df['building_details'].str.contains('сталінка')
    test['pre_revolutionary'] = df['building_details'].str.contains('дореволюційний')

    test['adjacent_separate'] = df['features_planning'].str.contains('суміжно-роздільна')
    test['multilevel'] = df['features_planning'].str.contains('багаторівнева')
    test['kitchen_living_room'] = df['features_planning'].str.contains('кухня-вітальня')
    test['penthouse'] = df['features_planning'].str.contains('пентхаус')
    test['studio'] = df['features_planning'].str.contains('студія')
    test['free_planning'] = df['features_planning'].str.contains('вільне планування')
    test['adjacent'] = df['features_planning'].str.contains('суміжна')
    test['separate'] = df['features_planning'].str.contains('роздільна')

    test['eurorenovation'] = df['repair_state'].str.contains('євроремонт')
    test['repair_in_progress'] = df['repair_state'].str.contains('незавершений ремонт')
    test['designer_renovation'] = df['repair_state'].str.contains('дизайнерський ремонт')
    test['needs_repairs'] = df['repair_state'].str.contains('потрібен ремонт')

    test['satisfactory_condition'] = df['condition'].str.contains('задовільний стан')


    test['good_condition'] = df['condition'].str.contains('хороший стан')
    test['excellent_condition'] = df['condition'].str.contains('чудовий стан')

    test['safe'] = df['facilities'].str.contains('сейф')
    test['shower_cabin'] = df['facilities'].str.contains('душова кабіна')
    test['wardrobe'] = df['facilities'].str.contains('шафа')
    test['TV'] = df['facilities'].str.contains('телевізор')
    test['hair_dryer'] = df['facilities'].str.contains('фен')
    test['dishes'] = df['facilities'].str.contains('посуд')
    test['satellite_TV'] = df['facilities'].str.contains('супутникове ТБ')
    test['DVD_player'] = df['facilities'].str.contains('DVD програвач')
    test['washing_machine'] = df['facilities'].str.contains('пральна машина')
    test['fireplace'] = df['facilities'].str.contains('камін')
    test['dishwashers'] = df['facilities'].str.contains('посудомийна машина')
    test['alarms'] = df['facilities'].str.contains('сигналізація')
    test['bed'] = df['facilities'].str.contains('ліжко')
    test['counters'] = df['facilities'].str.contains('лічильники')
    test['air_conditioning'] = df['facilities'].str.contains('кондиціонер')
    test['refrigerator'] = df['facilities'].str.contains('холодильник')
    test['jacuzzi'] = df['facilities'].str.contains('джакузі')
    test['microwave'] = df['facilities'].str.contains('мікрохвильовка')
    test['iron'] = df['facilities'].str.contains('праска')
    test['cable_TV'] = df['facilities'].str.contains('кабельне ТБ')

    test['district'] = df['district']
    test['subway'] = df['subway'].str.lower()

    test = test.merge(df_district_location, how='left', on='district')

    test['dist_to_center|full_area_log'] = (test['dist_to_center']
                                            * test['full_area_log'])

    test['dist_to_center|full_area_log|kitchen_area'] = (test['dist_to_center']
                                                         * test['full_area']
                                                         * test['kitchen_area'])

    test['full_area|kitchen_area_log'] = (test['full_area']
                                          / test['kitchen_area_log'])

    return test

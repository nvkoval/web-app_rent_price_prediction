import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


df_subway_location = pd.read_csv('data/subway_location.csv')

df_district_location = pd.read_csv('data/district_location.csv')
dict_district = df_district_location.set_index('district').to_dict()

geolocator = Nominatim(user_agent='my_request')


def get_location(address):
    lat, lon = None, None
    loc = address + ', Київ'
    location = geolocator.geocode(loc)
    if location:
        if 'Київ,' in str(location).split():
            lat = location.latitude
            lon = location.longitude
    return lat, lon


def get_min_dist_to_subway(row):
    loc = row['lat'], row['lon']
    dict_dist = {}
    for station in df_subway_location['subway']:
        idx = (df_subway_location
               .index[df_subway_location['subway'] == station]
               .values[0])
        subway_loc = (df_subway_location.at[idx, 'subway_lat'],
                      df_subway_location.at[idx, 'subway_lon'])
        dict_dist[station] = geodesic(loc, subway_loc).kilometers
    station = min(dict_dist, key=dict_dist.get)
    min_dist = dict_dist[station]
    return station, min_dist


centre_loc = (50.450555, 30.5206892)


def get_dist_to_center(row):
    loc = row['lat'], row['lon']
    dist = geodesic(loc, centre_loc).kilometers
    return dist


def fe(df):
    test = pd.DataFrame(index=[0])
    df['location'] = df['address'].apply(get_location)

    test = (test
            .assign(rooms=df['rooms'],
                    floor=df['floor'],
                    full_area=df['full_area'],
                    living_area=df['living_area'],
                    kitchen_area=df['kitchen_area'],
                    num_storeys=df['num_storeys'],
                    floor_other=df['floor'].where(df['floor'] < 27, 30),
                    full_area_log=np.log1p(df['full_area']),
                    living_area_log=np.log1p(df['living_area']),
                    kitchen_area_log=np.log1p(df['kitchen_area']),
                    num_storeys_other=(df['num_storeys']
                                       .where(df['num_storeys'] < 36, 40)),
                    first_rental=df['first_rental'],
                    last_floor=(df['floor'] == df['num_storeys']).astype(int),
                    first_floor=(df['floor'] == 1).astype(int),
                    district=df['district'],

                    typical_panel=df['building_details'].str.contains('типова панель'),
                    ukrainian_panel=df['building_details'].str.contains('українська панель'),
                    old_panel=df['building_details'].str.contains('стара панель'),
                    concrete_monolithic=df['building_details'].str.contains('бетонно-монолітний'),
                    old_brick=df['building_details'].str.contains('стара цегла'),
                    ukrainian_brick=df['building_details'].str.contains('українська цегла'),
                    gas_block=df['building_details'].str.contains('газоблок'),
                    Stalinka=df['building_details'].str.contains('сталінка'),
                    pre_revolutionary=df['building_details'].str.contains('дореволюційний'),

                    adjacent_separate=df['features_planning'].str.contains('суміжно-роздільна'),
                    multilevel=df['features_planning'].str.contains('багаторівнева'),
                    kitchen_living_room=df['features_planning'].str.contains('кухня-вітальня'),
                    penthouse=df['features_planning'].str.contains('пентхаус'),
                    studio=df['features_planning'].str.contains('студія'),
                    free_planning=df['features_planning'].str.contains('вільне планування'),
                    adjacent=df['features_planning'].str.contains('суміжна'),
                    separate=df['features_planning'].str.contains('роздільна'),

                    eurorenovation=df['repair_state'].str.contains('євроремонт'),
                    repair_in_progress=df['repair_state'].str.contains('незавершений ремонт'),
                    designer_renovation=df['repair_state'].str.contains('дизайнерський ремонт'),
                    needs_repairs=df['repair_state'].str.contains('потрібен ремонт'),

                    satisfactory_condition=df['condition'].str.contains('задовільний стан'),
                    good_condition=df['condition'].str.contains('хороший стан'),
                    excellent_condition=df['condition'].str.contains('чудовий стан'),

                    safe=df['facilities'].str.contains('сейф'),
                    shower_cabin=df['facilities'].str.contains('душова кабіна'),
                    wardrobe=df['facilities'].str.contains('шафа'),
                    TV=df['facilities'].str.contains('телевізор'),
                    hair_dryer=df['facilities'].str.contains('фен'),
                    dishes=df['facilities'].str.contains('посуд'),
                    satellite_TV=df['facilities'].str.contains('супутникове ТБ'),
                    DVD_player=df['facilities'].str.contains('DVD програвач'),
                    washing_machine=df['facilities'].str.contains('пральна машина'),
                    fireplace=df['facilities'].str.contains('камін'),
                    dishwashers=df['facilities'].str.contains('посудомийна машина'),
                    alarms=df['facilities'].str.contains('сигналізація'),
                    bed=df['facilities'].str.contains('ліжко'),
                    counters=df['facilities'].str.contains('лічильники'),
                    air_conditioning=df['facilities'].str.contains('кондиціонер'),
                    refrigerator=df['facilities'].str.contains('холодильник'),
                    jacuzzi=df['facilities'].str.contains('джакузі'),
                    microwave=df['facilities'].str.contains('мікрохвильовка'),
                    iron=df['facilities'].str.contains('праска'),
                    cable_TV=df['facilities'].str.contains('кабельне ТБ'),

                    district_lat=lambda df: dict_district['district_lat'][df['district'][0]],
                    district_lon=lambda df: dict_district['district_lon'][df['district'][0]],
                    )
            )

    test['lat'], test['lon'] = zip(*df['location'])
    test['lat'] = test['lat'].fillna(test['district_lat'])

    test['lon'] = test['lon'].fillna(test['district_lon'])

    test['subway'], test['min_dist_to_subway'] = zip(*test.apply(get_min_dist_to_subway, axis=1))
    test['dist_to_center'] = test.apply(get_dist_to_center, axis=1)

    test['dist_to_center|full_area_log'] = (test['dist_to_center']
                                            * test['full_area_log'])

    test['dist_to_center|full_area_log|kitchen_area'] = (test['dist_to_center']
                                                         * test['full_area']
                                                         * test['kitchen_area'])

    test['full_area|kitchen_area_log'] = (test['full_area']
                                          / test['kitchen_area_log'])

    return test

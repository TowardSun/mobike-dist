# -*- coding: utf-8 -*-

import math
from enum import IntEnum


class ModelChoice(IntEnum):
    cnn = 0
    dense_cnn = 1


class ReducerChoice(IntEnum):
    fa = 0
    pca = 1
    tca = 2


class FeatureChoice(IntEnum):
    all = 0
    poi = 1
    street = 2
    engineer = 3


class Location:
    def __init__(self, lat, lng):
        self._lat = lat
        self._lng = lng

    @property
    def lat(self):
        return self._lat

    @property
    def lng(self):
        return self._lng


class Block:
    def __init__(self, left_lower, right_upper, width, height):
        """
        定义城市矩形范围，以及网格的大小
        :param left_lower: 城市左下角坐标
        :param right_upper: 城市右上角坐标
        :param width: 网格宽
        :param height: 网格高
        """
        self.left_lower = left_lower
        self.right_upper = right_upper
        self.width = width
        self.height = height


def get_grid_steps(block):
    """
    获取网格步长
    """
    left_lower = block.left_lower
    right_upper = block.right_upper

    lat_steps = int(math.ceil((right_upper.lat - left_lower.lat) / block.height))
    lng_steps = int(math.ceil((right_upper.lng - left_lower.lng) / block.width))
    return lat_steps, lng_steps


def get_sh_range(width=0.01, height=0.01):
    # 上海市大范围：31.495189, 120.914373, 30.700175, 121.967546
    sh_left_lower = Location(lat=30.705000, lng=120.915000)
    sh_right_upper = Location(lat=31.495000, lng=121.965000)

    sh_block = Block(sh_left_lower, sh_right_upper, width=width, height=height)
    lat_steps, lng_steps = get_grid_steps(sh_block)
    return lat_steps, lng_steps, sh_block


def get_bj_range(width=0.01, height=0.01):
    # mobike spider 40.2912, 116.0796, 39.7126, 116.7086
    bj_lower_left = Location(lat=39.720, lng=116.090)
    bj_upper_right = Location(lat=40.280, lng=116.700)

    bj_block = Block(bj_lower_left, bj_upper_right, width=width, height=height)
    lat_steps, lng_steps = get_grid_steps(bj_block)
    return lat_steps, lng_steps, bj_block


def get_nb_range(width=0.01, height=0.01):
    # mobike range 29.99, 121.45, 29.80, 121.91
    nb_lower_left = Location(lat=29.80, lng=121.45)
    nb_upper_right = Location(lat=29.99, lng=121.90)

    nb_block = Block(nb_lower_left, nb_upper_right, width=width, height=height)
    lat_steps, lng_steps = get_grid_steps(nb_block)
    return lat_steps, lng_steps, nb_block


PATH_PATTERN = './data/%s_mobike_grid_week_0717_0721_street_std_mean_1000_1000.csv'

POI_FEATURES = ['food', 'hotel', 'shopping', 'life_service', 'beauty', 'tourist', 'entertainment', 'sports',
                'education', 'culture_media', 'medical', 'car_service', 'transportation', 'finance', 'estate',
                'company', 'government', 'num_pois', 'poi_entropy']

ENGINEER_FEATURES = ['light', 'light_dis',
                     'subway_dis', 'shop_center_dis', 'shop_center_level']

STREET_FEATURES = ['highway-bridleway', 'highway-cycleway', 'highway-footway', 'highway-living_street',
                   'highway-motorway', 'highway-motorway_link', 'highway-path',
                   'highway-pedestrian', 'highway-primary', 'highway-primary_link', 'highway-raceway',
                   'highway-residential', 'highway-road', 'highway-secondary', 'highway-secondary_link',
                   'highway-service', 'highway-steps', 'highway-tertiary', 'highway-tertiary_link',
                   'highway-track', 'highway-trunk', 'highway-trunk_link', 'highway-unclassified',
                   'man_made-pier',
                   'railway-disused', 'railway-funicular', 'railway-light_rail',
                   'railway-monorail', 'railway-narrow_gauge',
                   'railway-preserved', 'railway-rail',
                   'railway-subway', 'railway-tram', 'num_highway', 'num_railway', 'highway_entropy']

FEATURES = POI_FEATURES + ENGINEER_FEATURES + STREET_FEATURES
TARGET = 'mobike'


CITY_BLOCK_DICT = {
    'sh': get_sh_range(),
    'bj': get_bj_range(),
    'nb': get_nb_range()
}


FEATURE_DICT = {
    FeatureChoice.all: FEATURES,
    FeatureChoice.poi: POI_FEATURES,
    FeatureChoice.street: STREET_FEATURES,
    FeatureChoice.engineer: ENGINEER_FEATURES
}

LOG_DIR = './logs'

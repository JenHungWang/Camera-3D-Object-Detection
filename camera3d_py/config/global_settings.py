# Copyright 2023 Jen-Hung Wang, Lehrstuhl für Fahrzeugtechnik, TUM School of Engineering and Design, TU München
"""System module."""
import configparser
import json


def create_config_dict(config):
    """Create config dict"""
    dict = {}
    for section in config.sections():
        dict[section] = {}
        for key, val in config.items(section):
            dict[section][key] = val
    return dict


def import_config_dict():
    """Import config dict"""
    config = configparser.ConfigParser()
    config.read('config/config_camera.ini')
    config_dict = create_config_dict(config)

    config.read('config/config_bbox.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_obj_points.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_inference.ini')
    config_dict.update(create_config_dict(config))

    # camera parameters
    config_dict['PARAMETER']['camera_matrix'] = \
        json.loads(config_dict['PARAMETER']['camera_matrix'])
    config_dict['PARAMETER']['distortion'] = \
        json.loads(config_dict['PARAMETER']['distortion'])

    # bbox coordinates
    config_dict['COORDINATE']['axis'] = \
        json.loads(config_dict['COORDINATE']['axis'])
    config_dict['COORDINATE']['cube'] = \
        json.loads(config_dict['COORDINATE']['cube'])
    config_dict['COORDINATE']['rect'] = \
        json.loads(config_dict['COORDINATE']['rect'])

    # 3D object points
    config_dict['COORDINATE']['wheel_rear_left'] = \
        json.loads(config_dict['COORDINATE']['wheel_rear_left'])
    config_dict['COORDINATE']['wheel_rear_right'] = \
        json.loads(config_dict['COORDINATE']['wheel_rear_right'])
    config_dict['COORDINATE']['wheel_front_left'] = \
        json.loads(config_dict['COORDINATE']['wheel_front_left'])
    config_dict['COORDINATE']['wheel_front_right'] = \
        json.loads(config_dict['COORDINATE']['wheel_front_right'])

    # Inference Settings
    config_dict['PARAMETER']['resolution'] = \
        int(config_dict['PARAMETER']['resolution'])
    config_dict['PARAMETER']['conf_threshold'] = \
        float(config_dict['PARAMETER']['conf_threshold'])
    config_dict['PARAMETER']['iou_threshold'] = \
        float(config_dict['PARAMETER']['iou_threshold'])
    config_dict['PARAMETER']['line_thickness'] = \
        int(config_dict['PARAMETER']['line_thickness'])
    config_dict['PARAMETER']['hide_conf'] = \
        str2bool(config_dict['PARAMETER']['hide_conf'])

    return config_dict


def str2bool(string):
    """string to boolean"""
    return string.lower() in ("yes", "true", "t", "1")

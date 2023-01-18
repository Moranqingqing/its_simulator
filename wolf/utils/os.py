import os
import logging
import json
import yaml

LOGGER = logging.getLogger(__name__)


def makedirs(path):
    # path = os.path.dirname(path)
    if not os.path.exists(path):
        LOGGER.info("Creating folder(s) \"{}\"".format(path))
        os.makedirs(path)
    else:
        LOGGER.warning("Can't create \"{}\", folder exists".format(path))


def empty_directory(path_directory):
    os.system("rm -rf {}/*".format(str(path_directory)))


def load_config_file(config):
    if type(config) == type(''):
        with open(config, 'r') as infile:
            if 'json' in config:
                config = json.load(infile)
            elif 'yaml' in config:
                config = yaml.full_load(infile)
            else:
                raise Exception('Unsupported format, yaml or json accepted')
    elif type(config) != type({}):
        raise TypeError('Wrong type for configuration, must be a path or a dict')

    return config

# -*- coding: utf-8 -*-

from os.path import (expanduser, join, split, isdir, isfile, splitext)
from os import (makedirs, rename)
from random import choice
from string import ascii_lowercase
from time import strftime
try:
    from configparser import (ConfigParser, Error)
except ImportError:
    from ConfigParser import (ConfigParser, Error)
from niftynet.utilities.decorators import singleton


@singleton
class NiftyNetGlobalConfig(object):
    """Global configuration settings"""

    global_section = 'global'
    home_key = 'home'

    def __init__(self):
        self._download_server_url = \
            'https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer'
        self._config_home = join(expanduser('~'), '.niftynet')
        self._config_file = join(self._config_home, 'config.ini')

        config_opts = self.__load_or_create(self._config_file)

        self._niftynet_home = expanduser(
            config_opts[NiftyNetGlobalConfig.global_section][NiftyNetGlobalConfig.home_key])
        if not isdir(self._niftynet_home):
            makedirs(self._niftynet_home)

    def __load_or_create(self, config_file):
        """Load passed configuration file, if it exists; create a default
        otherwise. If this method finds an incorrect config file, it
        backs the file up with a human-readable timestamp suffix and
        creates a default one.

        :param config_file: no sanity checks are performed, as this
        method is for internal use only
        :type config_file: `os.path`
        :returns: a dictionary of parsed configuration options
        :rtype: `dict`
        """
        required_sections = [NiftyNetGlobalConfig.global_section]
        required_keys = {
            required_sections[0]: [NiftyNetGlobalConfig.home_key]
        }
        default_values = {
            required_sections[0]: {
                NiftyNetGlobalConfig.home_key: '~/niftynet'
            }
        }

        backup = False
        if isfile(config_file):
            try:
                config = ConfigParser()
                config.read(config_file)

                # check all required sections and keys present
                for required_section in required_sections:
                    if required_section not in config:
                        backup = True
                        break

                    for required_key in required_keys[required_section]:
                        if required_key not in config[required_section]:
                            backup = True
                            break

                    if backup:
                        break

            except Error:
                backup = True

            if not backup:  # loaded file contains all required
                            # config options: so return
                return dict(config)

        config_dir, config_filename = split(config_file)
        if not isdir(config_dir):
            makedirs(config_dir)

        if backup:  # config file exists, but does not contain all required
                    # config opts: so backup not to override
            timestamp = strftime('%Y-%m-%d-%H-%M-%S')
            random_str = ''.join(choice(ascii_lowercase) for _ in range(3))
            backup_suffix = '-'.join(['backup', timestamp, random_str])

            filename, extension = splitext(config_filename)
            backup_filename = ''.join([filename, '-', backup_suffix, extension])
            backup_file = join(config_dir, backup_filename)
            rename(config_file, backup_file)

        # create a new default global config file
        config = ConfigParser(default_values)
        for required_section in required_sections:
            for required_key in required_keys[required_section]:
                config.add_section(required_section)
                config[required_section][required_key] = \
                    default_values[required_section][required_key]
        with open(config_file, 'w') as new_config_file:
            config.write(new_config_file)
        return dict(config)

    def get_niftynet_home_folder(self):
        """Return the folder containing NiftyNet models and data"""
        return self._niftynet_home

    def get_niftynet_config_folder(self):
        """Return the folder containing NiftyNet global configuration"""
        return self._config_home

    def get_default_examples_folder(self):
        """Return the default folder containing NiftyNet examples"""
        return join(self._niftynet_home, 'examples')

    def get_download_server_url(self):
        """Return the URL to the NiftyNet examples server"""
        return self._download_server_url

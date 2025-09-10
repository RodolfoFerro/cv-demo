"""Configuration utilities module."""

from configparser import ConfigParser


def load_config(config_file):
    """Load the configuration file.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.

    Returns
    -------
    config : ConfigParser
        A configuration object loaded from file.
    """

    config = ConfigParser()
    config.read(config_file)

    return config

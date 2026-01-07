# Define the version of the template module.
__version__ = "1.1.1"
__minimal_miner_version__ = "1.1.1"
__minimal_validator_version__ = "1.1.1"

version_split = __version__.split(".")
__version_as_int__ = (100 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))

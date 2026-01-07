def get_local_version():
    try:
        # loading version from __init__.py
        here = path.abspath(path.dirname(__file__))
        parent = here.rsplit("/", 1)[0]
        with codecs.open(os.path.join(parent, "__init__.py"), encoding="utf-8") as init_file:
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
            version_string = version_match.group(1)
        return version_string
    except Exception as e:
        bt.logging.error(f"Error getting local version. : {e}")
        return ""
    
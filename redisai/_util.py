import six
import importlib


def to_string(s):
    if isinstance(s, six.string_types):
        return s
    elif isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s  # Not a string we care about


def is_installed(packages):
    if not isinstance(packages, list):
        packages = [packages]
    for p in packages:
        if importlib.util.find_spec(p) is None:
            return False
    return True

from .client import Client  # noqa
import pkg_resources

__version__ = pkg_resources.get_distribution('redisai').version

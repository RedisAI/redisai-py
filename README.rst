==========
redisai-py
==========

.. image:: https://img.shields.io/github/license/RedisAI/redisai-py.svg
        :target: https://github.com/RedisAI/redisai-py

.. image:: https://badge.fury.io/py/redisai.svg
        :target: https://badge.fury.io/py/redisai

.. image:: https://circleci.com/gh/RedisAI/redisai-py/tree/master.svg?style=svg
        :target: https://circleci.com/gh/RedisAI/redisai-py/tree/master

.. image:: https://img.shields.io/github/release/RedisAI/redisai-py.svg
        :target: https://github.com/RedisAI/redisai-py/releases/latest

.. image:: https://codecov.io/gh/RedisAI/redisai-py/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/RedisAI/redisai-py

.. image:: https://readthedocs.org/projects/redisai-py/badge/?version=latest
        :target: https://redisai-py.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Forum-RedisAI-blue
        :target: https://forum.redis.com/c/modules/redisai

.. image:: https://img.shields.io/discord/697882427875393627?style=flat-square
        :target: https://discord.gg/rTQm7UZ

.. image:: https://snyk.io/test/github/RedisAI/redisai-py/badge.svg?targetFile=pyproject.toml
        :target: https://snyk.io/test/github/RedisAI/redisai-py?targetFile=pyproject.toml

redisai-py is the Python client for RedisAI. Checkout the
`documentation <https://redisai-py.readthedocs.io/en/latest/>`_ for API details and examples

Installation
------------

1. Install Redis 5.0 or above
2. Install `RedisAI <http://redisai.io>`_
3. Install the Python client

.. code-block:: bash

    $ pip install redisai


4. Install serialization-deserialization utility (optional)

.. code-block:: bash

    $ pip install ml2rt

Development
-----------

1. Assuming you have virtualenv installed, create a virtualenv to manage your python dependencies, and activate it.
   ```virtualenv -v venv; source venv/bin/activate```
2. Install [pypoetry](https://python-poetry.org/) to manage your dependencies.
   ```pip install poetry```
3. Install dependencies.
   ```poetry install --no-root```

[tox](https://tox.readthedocs.io/en/latest/) runs all tests as its default target. Running *tox* by itself will run unit tests. Ensure you have a running redis, with the module loaded.

**Contributing**

Prior to submitting a pull request, please ensure you've built and installed poetry as above. Then:

1. Run the linter.
   ```tox -e linters.```
2. Run the unit tests. This assumes you have a redis server running, with the [RedisAI module](https://redisai.io) already loaded.  If you don't, you may want to install a [docker build](https://hub.docker.com/r/redislabs/redisai/tags).
   ```tox -e tests```

`RedisAI example repo <https://github.com/RedisAI/redisai-examples>`_ shows few examples
made using redisai-py under `python_client` folder. Also, checkout
`ml2rt <https://github.com/hhsecond/ml2rt>`_ for convenient functions those might help in
converting models (sparkml, sklearn, xgboost to ONNX), serializing models to disk, loading
it back to redisai-py etc.

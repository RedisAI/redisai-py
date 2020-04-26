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


`RedisAI example repo <https://github.com/RedisAI/redisai-examples>`_ shows few examples
made using redisai-py under `python_client` folder. Also, checkout
`ml2rt <https://github.com/hhsecond/ml2rt>`_ for convenient functions those might help in
converting models (sparkml, sklearn, xgboost to ONNX), serializing models to disk, loading
it back to redisai-py etc.

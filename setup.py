#!/usr/bin/env python
from setuptools import setup, find_packages


with open("README.rst") as f:
    long_description = f.read()

setup(
    name="redisai",
    version="1.0.2",
    description="RedisAI Python Client",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://github.com/RedisAI/redisai-py",
    author="RedisLabs",
    author_email="oss@redislabs.com",
    packages=find_packages(),
    install_requires=["redis", "hiredis", "numpy"],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Database",
    ],
)

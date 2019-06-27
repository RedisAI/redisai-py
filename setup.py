
#!/usr/bin/env python
from setuptools import setup, find_packages

exec(open('redisai/version.py').read())

setup(
    name='redisai',
    version=__version__,  # comes from redisai/version.py

    description='RedisAI Python Client',
    url='http://github.com/RedisAI/redisai-py',
    author='RedisLabs',
    author_email='oss@redislabs.com',
    packages=find_packages(),
    install_requires=['redis', 'hiredis', 'rmtest', 'onnxmltools'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Database',
        'Topic :: Software Development :: Testing'
    ]
)

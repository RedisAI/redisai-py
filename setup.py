
#!/usr/bin/env python
from setuptools import setup, find_packages

try:
	exec(open('redisai/version.py', encoding='utf-8').read())
except TypeError:
	exec(open('redisai/version.py').read())

with open('README.md') as f:
	long_description = f.read()


setup(
	name='redisai',
	version=__version__,  # comes from redisai/version.py
	description='RedisAI Python Client',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='http://github.com/RedisAI/redisai-py',
	author='RedisLabs',
	author_email='oss@redislabs.com',
	packages=find_packages(),
	install_requires=['redis', 'hiredis', 'rmtest', 'six', 'numpy'],
	python_requires='>=3.2',
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

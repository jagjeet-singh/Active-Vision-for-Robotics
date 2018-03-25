"""
Setup for kuka environment.
Environment has a kuka bot with mounted camera, table and objects.
"""
from setuptools import setup

setup(name='kuka',
	version='0.0.1',
	install_requires=['gym','pybullet'])
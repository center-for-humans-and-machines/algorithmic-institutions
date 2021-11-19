from setuptools import setup, find_packages
import sys
import os.path


def load_requirements(filename='requirements.txt'):
    with open(filename) as f:
        lines = f.readlines()
    return lines


setup(
    name='aimanage',
    version="0.0.1",
    description='',
    url='',
    author='MPIB - Human and Machines',
    author_email='',
    license='',
    packages=[package for package in find_packages()
              if package.startswith('aimanage')],
    zip_safe=False,
    install_requires=load_requirements(),
    extras_require={'dev': load_requirements('requirements_dev.txt')},
    scripts=[
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)

#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


def read_requirements():
    with open('requirements.txt') as req:
        return req.read().strip().split('\n')


extras_require = {
    'full': [
        'nemo_toolkit[asr]'
    ]
}

test_requirements = ['pytest>=3', ]

setup(
    author="Ara Yeroyan",
    author_email='ar23yeroyan@gmail.com',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Aligning Very Long audio and text pairs through VAC pipeline",
    install_requires=read_requirements(),
    extras_require=extras_require,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='vac_aligner',
    name='vac_aligner',
    packages=find_packages(include=['vac_aligner', 'vac_aligner.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Ara-Yeroyan/vac_aligner',
    download_url='https://github.com/Ara-Yeroyan/vac_aligner/archive/refs/tags/v0.0.2.tar.gz',
    version='0.2.0',
    zip_safe=False,
)

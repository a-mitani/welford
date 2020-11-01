from setuptools import setup

kwargs = {
    "name": "welford",
    "version": "0.1.3",
    "description": "Python (numpy) implementation of Welford's algorithm.",
    "author": "Akira Mitani",
    "author_email": "amitani.public@gmail.com",
    "url": "https://github.com/a-mitani/welford",
    "license": "MIT",
    "keywords": ["statistics", "online", "welford"],
    "install_requires": ["numpy"],
    "packages": ["welford"],
}

setup(**kwargs)

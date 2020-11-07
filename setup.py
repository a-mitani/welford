from setuptools import setup

with open("README.rst") as f:
    readme = f.read()

kwargs = {
    "name": "welford",
    "version": "0.2.4",
    "description": "Python (numpy) implementation of Welford's algorithm.",
    "author": "Akira Mitani",
    "author_email": "amitani.public@gmail.com",
    "url": "https://github.com/a-mitani/welford",
    "license": "MIT",
    "keywords": ["statistics", "online", "welford"],
    "install_requires": ["numpy"],
    "packages": ["welford"],
    "long_description": readme,
}

setup(**kwargs)

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["numpy~=1.19.4",
"pandas~=1.1.5",
"matplotlib~=3.3.3",
"tensorflow~=2.4.0",
"pyDOE~=0.3.8",
"scipy~=1.5.4",
"sklearn~=0.0",
"scikit-learn~=0.24.0",
"setuptools~=51.1.1"]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    descripion='DEANN',
)
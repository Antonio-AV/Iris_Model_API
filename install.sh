#!/bin/bash
apt-get update
apt-get install -y python3-distutils
pip install --upgrade setuptools
pip install -r requirements.txt

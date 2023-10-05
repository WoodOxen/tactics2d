#! /bin/bash

apt-get update
apt-get -y install python3 python3-pip
cd tactics2d
if [ -f requirements.txt ];
    then pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple;
fi
if [ -f tests/requirements.txt ];
    then pip install -r tests/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple;
fi
mkdir ./tests/runtime
pytest tests --cov=tactics2d --cov-report=xml
#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -y python3-distutils unzip

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo apt-get install python3-distutils -y
sudo apt-get install pipx -y
pipx ensurepath

export PATH=$PATH:~/.local/bin

sudo apt-get install unzip

sleep 5

pipx install kaggle

mkdir -p ~/.kaggle

gcloud secrets versions access latest --secret="kaggle-secret" > ~/.kaggle/kaggle.json

gcloud secrets versions access latest --secret="service-account-secret" > service-account-key.json

kaggle datasets download -d aeeeeeep/apple-single-object-detection

unzip apple-single-object-detection.zip

gcloud auth activate-service-account --key-file=./service-account-key.json

gsutil -m cp -r JPEGImages gs://apple-bucket-data
gsutil -m cp -r Annotations gs://apple-bucket-data

echo "Dataset was successfully uploaded to Google Cloud Storage (bucket)"

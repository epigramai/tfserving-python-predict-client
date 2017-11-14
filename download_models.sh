#!/usr/bin/env bash

rm -rf models/ &&
mkdir -p models &&
cd models/ &&
gsutil -m cp -r gs://epigram-models/incv3_2048/ . &&
gsutil -m cp -r gs://epigram-models/incv4_1536/ . &&
gsutil -m cp -r gs://epigram-models/nina_rcnn_inception_resnet_v2/ . &&
gsutil -m cp -r gs://epigram-models/door_omfang/ .

#!/usr/bin/env bash

rm -rf models/ &&
mkdir -p models &&
cd models/ &&
mkdir -p incv4_1536/1 && curl https://storage.googleapis.com/epigram-models/incv4_1536/1/saved_model.pb -o incv4_1536/1/savedmodel.pb

#!/usr/bin/env bash

rm -rf test_data/* &&
gsutil -m cp -r gs://epigram-test-images/* test_data/
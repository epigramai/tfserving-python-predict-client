#!/usr/bin/env bash

docker pull epigramai/model-server:light &&
docker run --name incv3 -p 9000:9000 -d -v $(pwd)/models/:/models/ epigramai/model-server:light --port=9000 --model_name=incv3 --model_base_path=/models/incv3_2048 &&
docker run --name incv4 -p 9001:9000 -d -v $(pwd)/models/:/models/ epigramai/model-server:light --port=9000 --model_name=incv4 --model_base_path=/models/incv4_1536 &&
docker run --name door -p 9002:9000 -d -v $(pwd)/models/:/models/ epigramai/model-server:light --port=9000 --model_name=door --model_base_path=/models/door_omfang &&
docker run --name nina -p 9003:9000 -d -v $(pwd)/models/:/models/ epigramai/model-server:light --port=9000 --model_name=nina --model_base_path=/models/nina_rcnn_inception_resnet_v2

# To stop and remove containers
# docker stop incv3 incv4 door nina && docker rm incv3 incv4 door nina
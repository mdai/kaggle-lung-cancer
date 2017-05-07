#!/usr/bin/env bash

# start redis
sudo docker run -d \
  --name kaggle-lung-cancer-redis \
  -v $(pwd)/redis_data:/data \
  -p 6379:6379 \
  --restart=always \
  redis:3.2.8

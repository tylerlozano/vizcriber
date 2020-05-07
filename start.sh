#!/bin/bash
docker build -t viz .
docker run -d --tmpfs /app/static -m=4g -p 5001:5001 viz
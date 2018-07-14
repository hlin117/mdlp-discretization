#!/usr/bin/env bash

DOCKER_IMAGE="quay.io/pypa/manylinux1_x86_64"

docker run -v `pwd`:/code $DOCKER_IMAGE /code/scripts/docker_build.sh
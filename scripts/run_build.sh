#!/usr/bin/env bash

DOCKER_IMAGE="quay.io/pypa/manylinux1_x86_64"
DOCKER_IMAGE2="quay.io/pypa/manylinux1_i686"

docker run -v `pwd`:/io $DOCKER_IMAGE /io/scripts/build_wheels.sh

docker run -v `pwd`:/io $DOCKER_IMAGE2 linux32 /io/scripts/build_wheels.sh
#!/usr/bin/env bash

PACKAGE_NAME="mdlp_discretization"

DOCKER_IMAGE="quay.io/pypa/manylinux1_x86_64"
PLAT="manylinux1_x86_64"

DOCKER_IMAGE2="quay.io/pypa/manylinux1_i686"
PLAT2="manylinux1_i686"

docker run --rm -e PLAT=$PLAT -e PACKAGE_NAME=$PACKAGE_NAME -v `pwd`:/io $DOCKER_IMAGE /io/scripts/build_wheels.sh

docker run --rm -e PLAT=$PLAT2 -e PACKAGE_NAME=$PACKAGE_NAME -v `pwd`:/io $DOCKER_IMAGE2 linux32 /io/scripts/build_wheels.sh

# Show and copy wheels
ls -lh wheelhouse/
mkdir -p dist
cp wheelhouse/$PACKAGE_NAME*.whl dist/.
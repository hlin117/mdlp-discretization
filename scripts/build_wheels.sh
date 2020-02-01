#!/bin/bash
set -e -x

echo ===================
echo "Building wheels for $PLAT"
echo ===================

# Install a system package required by our library

# Collect the pythons
pys=(/opt/python/*/bin)

# Print list of Python's available
echo "All Pythons: ${pys[@]}"

# Filter out Python 3.4
pys=(${pys[@]//*34*/})

# Compile wheels
# for PYBIN in /opt/python/*/bin; do
#     if [[ "$PYBIN" =~ (cp33|cp37) ]]; then
#         continue
#     fi
for PYBIN in "${pys[@]}"; do
    ${PYBIN}/pip install -r /io/dev-requirements.txt
    # ${PYBIN}/pip wheel /io/ -w wheelhouse/
    # (cd /io; ${PYBIN}/python setup.py bdist_wheel --dist-dir /wheelhouse)
    ${PYBIN}/pip wheel /io/ --no-deps -w wheelhouse/
done

ls -lh wheelhouse/

# Bundle external shared libraries into the wheels
for whl in wheelhouse/mdlp_discretization-*.whl; do
    auditwheel repair $whl --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
# for PYBIN in /opt/python/*/bin/; do
#     if [[ "$PYBIN" =~ (cp33|cp37) ]]; then
#         continue
#     fi
for PYBIN in "${pys[@]}"; do
    ${PYBIN}/pip install ${PACKAGE_NAME} --no-index -f /io/wheelhouse
    (cd /io; ${PYBIN}/pytest)
done
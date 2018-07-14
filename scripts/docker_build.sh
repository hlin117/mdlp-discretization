#!/bin/bash
set -e -x

# Install a system package required by our library
# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" =~ (cp33|cp37) ]]; then
        continue
    fi
    ${PYBIN}/pip install -r /code/dev-requirements.txt
    ${PYBIN}/pip wheel /code/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w /code/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ "$PYBIN" =~ (cp33|cp37) ]]; then
        continue
    fi
    ${PYBIN}/pip install mdlp-discretization --no-index -f /code/wheelhouse
    (cd /code; ${PYBIN}/python -m pytest)
done
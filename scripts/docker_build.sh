#!/bin/bash
set -e -x

# Install a system package required by our library
# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" =~ (cp33|cp37) ]]; then
        continue
    fi
    ${PYBIN}/pip install -r /code/dev-requirements.txt
    ${PYBIN}/pip wheel /code/ --no-deps -w dist/
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair $whl -w /code/dist/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ "$PYBIN" =~ (cp33|cp37) ]]; then
        continue
    fi
    ${PYBIN}/pip install mdlp-discretization --no-index -f /code/dist
    (cd /code; ${PYBIN}/python -m pytest)
done
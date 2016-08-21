all:
	python setup.py build_ext --inplace
test:
	nosetests
clean:
	rm -rf *.c *.so build

all:
	python setup.py build_ext --inplace
clean:
	rm -rf *.c *.so build

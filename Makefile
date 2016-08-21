all:
	python setup.py build_ext --inplace
test:
	@$(MAKE) all
	nosetests
clean:
	rm -rf *.c *.so build

all:
	python setup.py build_ext

clean:
	rm -rf .benchmarks
	rm -rf .pytest_cache
	rm -rf build
	rm -rf __pycache__
	rm src/pocketfft.cpp

test:
	pytest

install:
	pip install -e .
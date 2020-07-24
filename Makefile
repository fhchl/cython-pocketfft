all:
	python setup.py build_ext
	pip install -e .[dev]

clean:
	pip uninstall pocketfft
	rm -rf .benchmarks
	rm -rf .pytest_cache
	rm -rf build
	rm -rf __pycache__
	rm src/pocketfft.cpp

test:
	pytest
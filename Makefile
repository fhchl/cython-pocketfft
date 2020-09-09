all:
	#python setup.py build_ext
	pip install .[dev]

clean:
	pip uninstall -y pocketfft
	rm -rf .benchmarks
	rm -rf .pytest_cache
	rm -rf build
	rm -rf __pycache__
	rm -rf pocketfft.egg-info
	rm src/pocketfft.cpp

test:
	pytest
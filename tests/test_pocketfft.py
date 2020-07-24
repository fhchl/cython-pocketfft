import pocketfft
import numpy as np
import numpy.testing as npt
import pytest

sizes = [2**4, 2**8, 2**12, 2**16]


@pytest.mark.parametrize("size", sizes)
def test_pocket_fft(benchmark, size):
    x = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    result = benchmark(lambda: pocketfft.fft(x))
    npt.assert_allclose(np.asarray(result), np.fft.fft(x))


@pytest.mark.parametrize("size", sizes)
def test_numpy_fft(benchmark, size):
    x = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    benchmark(lambda: np.fft.fft(x))


@pytest.mark.parametrize("size", sizes)
def test_pocket_ifft(size):
    x = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    npt.assert_allclose(np.asarray(pocketfft.ifft(x)), np.fft.ifft(x))


@pytest.mark.parametrize("size", sizes)
def test_unitary(size):
    x = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    npt.assert_allclose(np.asarray(pocketfft.ifft(pocketfft.fft(x))), x)


@pytest.mark.parametrize("size", sizes)
def test_pocket_fft_2d(benchmark, size):
    x = np.random.normal(size=(size, 16)) + 1j * np.random.normal(size=(size, 16))
    out = np.empty(x.shape, dtype=complex)
    benchmark(lambda: pocketfft.fft(x, out=out, nthreads=2))
    npt.assert_allclose(out, np.fft.fft(x, axis=0))


@pytest.mark.parametrize("size", sizes)
def test_numpy_fft_2d(benchmark, size):
    x = np.random.normal(size=(size, 16)) + 1j * np.random.normal(size=(size, 16))
    benchmark(lambda: np.fft.fft(x, axis=0))

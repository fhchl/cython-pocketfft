import pocketfft
import numpy as np
import numpy.testing as npt
import pytest

sizes = [2 ** 4, 2 ** 8, 2 ** 12, 2 ** 16]


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="1D")
def test_pocket_fft(benchmark, size):
    x = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    result = benchmark(lambda: pocketfft.fft(x))
    npt.assert_allclose(np.asarray(result), np.fft.fft(x))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="1D")
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
@pytest.mark.benchmark(group="2D")
def test_pocket_fft_2d(benchmark, size):
    x = np.random.normal(size=(size, 16)) + 1j * np.random.normal(size=(size, 16))
    out = np.empty(x.shape, dtype=complex)
    benchmark(lambda: pocketfft.fft(x, out=out, nthreads=1))
    npt.assert_allclose(out, np.fft.fft(x, axis=0))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="2D")
def test_numpy_fft_2d(benchmark, size):
    x = np.random.normal(size=(size, 16)) + 1j * np.random.normal(size=(size, 16))
    benchmark(lambda: np.fft.fft(x, axis=0))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="3D")
def test_pocket_fft_3d(benchmark, size):
    x = np.random.normal(size=(size, 16, 16)) + 1j * np.random.normal(
        size=(size, 16, 16)
    )
    out = np.empty(x.shape, dtype=complex)
    benchmark(lambda: pocketfft.fft(x, out=out, nthreads=1))
    npt.assert_allclose(out, np.fft.fft(x, axis=0))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="3D")
def test_numpy_fft_3d(benchmark, size):
    x = np.random.normal(size=(size, 16, 16)) + 1j * np.random.normal(
        size=(size, 16, 16)
    )
    benchmark(lambda: np.fft.fft(x, axis=0))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="1D")
def test_ComplexFFT_1d(benchmark, size):
    arr_in = np.random.normal(size=(size)) + 1j * np.random.normal(size=(size))
    arr_out = np.zeros(arr_in.shape, dtype=complex)

    fft = pocketfft.ComplexFFT(arr_in, arr_out)

    benchmark(fft.forward)
    npt.assert_allclose(np.fft.fft(arr_in), arr_out)

    fft.backward()
    npt.assert_allclose(np.fft.ifft(arr_in), arr_out)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="2D")
def test_ComplexFFT_2d(benchmark, size):
    arr_in = np.random.normal(size=(16, size)) + 1j * np.random.normal(size=(16, size))
    arr_out = np.zeros(arr_in.shape, dtype=complex)

    fft = pocketfft.ComplexFFT(arr_in, arr_out)

    benchmark(fft.forward)
    npt.assert_allclose(np.fft.fft(arr_in), arr_out)

    fft.backward()
    npt.assert_allclose(np.fft.ifft(arr_in), arr_out)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="3D")
def test_ComplexFFT_3d(benchmark, size):
    arr_in = np.random.normal(size=(16, 16, size)) + 1j * np.random.normal(
        size=(16, 16, size)
    )
    arr_out = np.zeros(arr_in.shape, dtype=complex)

    fft = pocketfft.ComplexFFT(arr_in, arr_out)

    benchmark(fft.forward)
    npt.assert_allclose(np.fft.fft(arr_in), arr_out)

    fft.backward()
    npt.assert_allclose(np.fft.ifft(arr_in), arr_out)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="1D")
def test_realFFT_1d(benchmark, size):
    arr_real = np.random.normal(size=(size))
    arr_complex = np.zeros((size // 2 + 1), dtype=complex)

    fft = pocketfft.realFFT(arr_real, arr_complex)

    benchmark(fft.forward)  # writes arr_complex
    npt.assert_allclose(np.fft.rfft(arr_real), arr_complex)

    arr_real.fill(0)

    fft.backward()  # writes arr_real
    npt.assert_allclose(arr_real, np.fft.irfft(arr_complex))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="2D")
def test_realFFT_2d(benchmark, size):
    arr_real = np.random.normal(size=(16, size))
    arr_complex = np.zeros((16, size // 2 + 1), dtype=complex)

    fft = pocketfft.realFFT(arr_real, arr_complex)

    benchmark(fft.forward)  # writes arr_complex
    npt.assert_allclose(np.fft.rfft(arr_real), arr_complex)

    arr_real.fill(0)

    fft.backward()  # writes arr_real
    npt.assert_allclose(arr_real, np.fft.irfft(arr_complex))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.benchmark(group="3D")
def test_realFFT_3d(benchmark, size):
    arr_real = np.random.normal(size=(16, 16, size))
    arr_complex = np.zeros((16, 16, size // 2 + 1), dtype=complex)

    fft = pocketfft.realFFT(arr_real, arr_complex)

    benchmark(fft.forward)  # writes arr_complex
    npt.assert_allclose(np.fft.rfft(arr_real), arr_complex)

    arr_real.fill(0)

    fft.backward()  # writes arr_real
    npt.assert_allclose(arr_real, np.fft.irfft(arr_complex))

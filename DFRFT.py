import numpy as np

def fast_fractional_fourier_transform(vec, exponent):
    # Compute Fourier transform powers of vec.
    f0 = np.array(vec)
    f1 = np.fft.fft(f0)
    f2 = negate_permutation(f0)
    f3 = negate_permutation(f1)

    # Derive eigenbasis vectors from vec's Fourier transform powers.
    b0 = f0 + f1 + f2 + f3
    b1 = f0 + 1j*f1 - f2 - 1j*f3
    b2 = f0 - f1 + f2 - f3
    b3 = f0 - 1j*f1 - f2 + 1j*f3
    # Note: vec == (b0 + b1 + b2 + b3) / 4

    # Phase eigenbasis vectors by their eigenvalues raised to the exponent.
    b1 *= 1j**exponent
    b2 *= 1j**(2*exponent)
    b3 *= 1j**(3*exponent)

    # Recombine transformed basis vectors to get transformed vec.
    return (b0 + b1 + b2 + b3) / 4


def negate_permutation(vec):
    """Returns the result of applying an FFT to the given vector twice."""
    head, tail = vec[:1], vec[1:]
    return np.concatenate((head, tail[::-1]))
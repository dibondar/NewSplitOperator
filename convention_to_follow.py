"""
This is the convention we should follow for real 2 complex FAFT
"""
import cufft
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt

#################################################################################
#
# Lets generate some function
#
#################################################################################
X_gridDIM = 2024
X_amplitude = 5.

dX = 2. * X_amplitude / X_gridDIM
X = np.linspace(-X_amplitude, X_amplitude - dX, X_gridDIM)
Y = np.exp(-X**2)

#################################################################################
#
# Initialize GPU arrays
#
#################################################################################

# Allocate the largest array first for storing the FFT image of the real function
fft_Y_gpu = gpuarray.zeros((1, X_gridDIM // 2 + 1), dtype=np.complex128)

# Next allocate a smaler array containing the original real data that reuses data from fft_Y_gpu
Y_gpu = gpuarray.GPUArray((1, X_gridDIM), dtype=np.float64, gpudata=fft_Y_gpu.gpudata)

# Check that both arrays point to the same region
assert fft_Y_gpu.gpudata is Y_gpu.gpudata

#################################################################################
#
# Do FFTs
#
#################################################################################

# Initialize plans
cufft_D2Z = cufft.Plan2DAxis1(Y_gpu.shape, cufft.CUFFT_D2Z)
cufft_Z2D = cufft.Plan2DAxis1(Y_gpu.shape, cufft.CUFFT_Z2D)

# Transfer the data
Y_gpu[:] = Y[np.newaxis, :]

# Do FFT
cufft.cu_fft_D2Z(Y_gpu, fft_Y_gpu, cufft_D2Z)

print "Difference between CPU FFT and GPU FFT =", np.linalg.norm(np.fft.rfft(Y) - fft_Y_gpu.get())

# Do iFFT
cufft.cu_ifft_Z2D(fft_Y_gpu, Y_gpu, cufft_Z2D)
Y_gpu /= X_gridDIM

print "Difference between iFFT(FFT(y)) and y =", np.linalg.norm(Y - Y_gpu.get())


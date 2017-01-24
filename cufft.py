import ctypes
import numpy as np

__doc__ = "ctypes wrapper of CUDA FFT (cuFFT)"

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

CUFFT_R2C = 0x2a # Real to complex (interleaved)
CUFFT_C2R = 0x2c # Complex (interleaved) to real
CUFFT_C2C = 0x29 # Complex to complex (interleaved)
CUFFT_D2Z = 0x6a # Double to double-complex (interleaved)
CUFFT_Z2D = 0x6c # Double-complex (interleaved) to double
CUFFT_Z2Z = 0x69 # Double-complex to double-complex (interleaved)

_libcufft = ctypes.cdll.LoadLibrary('libcufft.so')

_libcufft.cufftPlanMany.argtypes = [ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]


# General CUFFT error:
class cufftError(Exception):
    """CUFFT error"""
    pass


# Exceptions corresponding to different CUFFT errors:
class cufftInvalidPlan(cufftError):
    """CUFFT was passed an invalid plan handle."""
    pass


class cufftAllocFailed(cufftError):
    """CUFFT failed to allocate GPU memory."""
    pass


class cufftInvalidType(cufftError):
    """The user requested an unsupported type."""
    pass


class cufftInvalidValue(cufftError):
    """The user specified a bad memory pointer."""
    pass


class cufftInternalError(cufftError):
    """Internal driver error."""
    pass


class cufftExecFailed(cufftError):
    """CUFFT failed to execute an FFT on the GPU."""
    pass


class cufftSetupFailed(cufftError):
    """The CUFFT library failed to initialize."""
    pass


class cufftInvalidSize(cufftError):
    """The user specified an unsupported FFT size."""
    pass


class cufftUnalignedData(cufftError):
    """Input or output does not satisfy texture alignment requirements."""
    pass

cufftExceptions = {
    0x1: cufftInvalidPlan,
    0x2: cufftAllocFailed,
    0x3: cufftInvalidType,
    0x4: cufftInvalidValue,
    0x5: cufftInternalError,
    0x6: cufftExecFailed,
    0x7: cufftSetupFailed,
    0x8: cufftInvalidSize,
    0x9: cufftUnalignedData
}


def cufftCheckStatus(status):
    """Raise an exception if the specified CUBLAS status is an error."""

    if status != 0:
        try:
            raise cufftExceptions[status]
        except KeyError:
            raise cufftError


def cufftPlanMany(rank, n, inembed, istride, idist, onembed, ostride, odist, fft_type, batch):
    """
    Create batched FFT plan configuration.
    """
    plan = ctypes.c_uint()
    status = _libcufft.cufftPlanMany(
                ctypes.byref(plan), rank, n, inembed, istride, idist, onembed, ostride, odist, fft_type, batch
    )
    cufftCheckStatus(status)
    return plan


# Destroy Plan
_libcufft.cufftDestroy.restype = int
_libcufft.cufftDestroy.argtypes = [ctypes.c_uint]


def cufftDestroy(plan):
    """Destroy FFT plan."""
    status = _libcufft.cufftDestroy(plan)
    cufftCheckStatus(status)

# double2complex FFT

_libcufft.cufftExecD2Z.restype = int
_libcufft.cufftExecD2Z.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cu_fft_D2Z(x_input_gpu, y_output_gpu, plan):
    _libcufft.cufftExecD2Z(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_FORWARD)


def cu_ifft_D2Z(x_input_gpu, y_output_gpu, plan):
    _libcufft.cufftExecD2Z(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_INVERSE)

##################################################################################
#
# complex2double FFT
#
##################################################################################

_libcufft.cufftExecZ2D.restype = int
_libcufft.cufftExecZ2D.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cu_fft_Z2D(x_input_gpu, y_output_gpu, plan):
    status = _libcufft.cufftExecZ2D(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_FORWARD)
    cufftCheckStatus(status)


def cu_ifft_Z2D(x_input_gpu, y_output_gpu, plan):
    status = _libcufft.cufftExecZ2D(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_INVERSE)
    cufftCheckStatus(status)


class Plan2DAxis0:
    """
    Create cuFFT plat to perform FFT of 2D array over axis = 0
    """
    def __init__(self, shape, cuFFT_type):
        self.batch = shape[1]
        self.fft_type = cuFFT_type

        if len(shape) == 2:
            n = np.array([shape[0]])
            stride = shape[1]  # distance jump between two elements in the same series
            idist = 1  # distance jump between two consecutive batches

            inembed = np.array([shape[0], stride])
            onembed = np.array([shape[0], stride])

            rank = 1
            self.handle = cufftPlanMany(
                rank, n.ctypes.data, inembed.ctypes.data, stride, idist,
                onembed.ctypes.data, stride, idist, self.fft_type, self.batch
            )
        else:
            raise ValueError('invalid transform dimension')

    def __del__(self):
        try:
            cufftDestroy(self.handle)
        except:
            pass


class Plan2DAxis1:
    """
    Create cuFFT plat to perform FFT of 2D array over axis = 1
    """
    def __init__(self, shape, cuFFT_type):
        self.batch = shape[0]
        self.fft_type = cuFFT_type

        if len(shape) == 2:
            n = np.array([shape[1]])

            rank = 1
            self.handle = cufftPlanMany(
                rank, n.ctypes.data, None, 1, 0, None, 1, 0, self.fft_type, self.batch
            )
        else:
            raise ValueError('invalid transform dimension')

    def __del__(self):
        try:
            cufftDestroy(self.handle)
        except:
            pass

##################################################################################
#
# complex 2 complex fft
#
##################################################################################

# Execution
_libcufft.cufftExecZ2Z.restype = int
_libcufft.cufftExecZ2Z.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cufftExecZ2Z(plan, idata, odata, direction):
    """Execute double precision complex-to-complex transform plan as
    specified by `direction`."""
    status = _libcufft.cufftExecZ2Z(plan, idata, odata, direction)
    cufftCheckStatus(status)


def fft_Z2Z(x_input_gpu, y_output_gpu, plan):
    cufftExecZ2Z(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_FORWARD)


def ifft_Z2Z(x_input_gpu, y_output_gpu, plan):
    cufftExecZ2Z(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_INVERSE)


class PlanZ2Z:
    """
    CUFFT plan class.
    This class represents an FFT  plan for CUFFT for complex double precission Z2Z
    Parameters
    ----------
    shape : ntuple
    batch : int
        Number of FFTs to configure in parallel (default is 1).
    """
    def __init__(self, shape, batch=1):
        self.shape = shape
        self.batch = batch
        self.fft_type = CUFFT_Z2Z

        if len(self.shape) > 0:
            n = np.asarray(shape, np.int32)
            rank = len(shape)
            self.handle = cufftPlanMany(
                rank, n.ctypes.data, None, 1, 0, None, 1, 0, self.fft_type, self.batch
            )
        else:
            raise ValueError('invalid transform size')

    def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
        try:
            cufftDestroy(self.handle)
        except:
            pass


class Plan_Z2Z_2D_Axis0:
    """
    fft in axis 0 for a 2D array
    Parameters
    ----------
    shape : ntuple of four elements
    """
    def __init__(self, shape):
        self.batch = shape[1]
        self.fft_type = CUFFT_Z2Z

        if len(shape) == 2:
            n = np.array([ shape[0] ])
            stride = shape[1]           # distance jump between two elements in the same series
            idist  = 1                  # distance jump between two consecutive batches

            inembed = np.array( [shape[0],stride] )
            onembed = np.array( [shape[0],stride] )

            rank = 1
            self.handle = cufftPlanMany(
                rank, n.ctypes.data, inembed.ctypes.data, stride,
                idist, onembed.ctypes.data, stride, idist, self.fft_type, self.batch
            )
        else:
            raise ValueError('invalid transform dimension')

    def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
        try:
            cufftDestroy(self.handle)
        except:
            pass


class Plan_Z2Z_2D_Axis1:
    """
    fft in axis 1 for a 2D array
    Parameters
    ----------
    shape : ntuple of four elements
    """
    def __init__(self, shape):
        self.batch = shape[0]
        self.fft_type = CUFFT_Z2Z

        if len(shape) == 2:
            n = np.array([ shape[1] ])
            stride = 1                  # distance jump between two elements in the same series
            idist = shape[1]	    # distance jump between two consecutive batches

            inembed = np.array( shape )
            onembed = np.array( shape )

            rank = 1
            self.handle = cufftPlanMany(
                rank, n.ctypes.data, inembed.ctypes.data, stride, idist,
                onembed.ctypes.data, stride, idist, self.fft_type, self.batch
            )
        else:
            raise ValueError('invalid transform dimension')

    def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
        try:
            cufftDestroy(self.handle)
        except:
            pass
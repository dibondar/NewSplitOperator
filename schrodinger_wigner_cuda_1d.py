import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.tools import dtype_to_ctype
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np
import os
import ctypes
from types import MethodType, FunctionType


class SchrodingerWignerCUDA1D:
    """
    The second-order split-operator propagator for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t) using CUDA.

    The Wigner function is obtained by Wigner transforming the wave function,
    which is propagated via the Schrodinger equation.

    This implementation stores the Wigner function as a 2D real gpu array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters are to be specified
            X_gridDIM - the coordinate grid size

            X_amplitude - coordinate maximum value
            P_amplitude - momentum's maximum value

            t (optional) - initial value of time (default t = 0)
            consts (optional) - a string of the C code declaring the constants

            functions (optional) -  a string of the C code declaring auxiliary functions

            V - a string of the C code specifying potential energy. Coordinate (X) and time (t) variables are declared.
            K - a string of the C code specifying kinetic energy. Momentum (P) and time (t) variables are declared.

            diff_V (optional) - a string of the C code specifying the potential energy derivative w.r.t. X
                                    for the Ehrenfest theorem calculations
            diff_K (optional) - a string of the C code specifying the kinetic energy derivative w.r.t. P
                                    for the Ehrenfest theorem calculations

            dt - time step
            abs_boundary_p (optional) - a string of the C code specifying function of PP and PP_prime,
                                    which will be applied to the density matrix at each propagation step

            abs_boundary_x (optional) - a string of the C code specifying function of XX and XX_prime,
                                    which will be applied to the density matrix at each propagation step
        """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert self.X_gridDIM == 2048, "Currently, only values of X_gridDIM = 2048 are permitted"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.P_amplitude
        except AttributeError:
            raise AttributeError("Momentum grid range (P_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
            del kwargs['t']
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        # Save the current value of t as the initial time
        kwargs.update(t_initial=self.t)

        self.t = np.float64(self.t)

        ##########################################################################################
        #
        # Generating grids
        #
        ##########################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.dP = 2. * self.P_amplitude / self.X_gridDIM

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)

        # momentum grid
        self.P = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.X_gridDIM)

        ##########################################################################################
        #
        # Allocate memory for the wigner function and the wave function
        #
        ##########################################################################################

        self.wavefunction = gpuarray.zeros(self.X_gridDIM, np.complex64)

        # Wave function copy (wavefunction in the momentum representation)
        self._tmp = gpuarray.empty_like(self.wavefunction)

        # Types for CUDA compilation
        self.cuda_types = dict(
            cuda_complex=dtype_to_ctype(self.wavefunction.dtype),
            cuda_real=dtype_to_ctype(self.wavefunction.real.dtype),
        )

        ##########################################################################################
        #
        # Absorbing boundaries in different representations
        #
        ##########################################################################################

        try:
            self.abs_boundary_x
        except AttributeError:
            self.abs_boundary_x = "{cuda_real}(1.)".format(**self.cuda_types)

        try:
            self.abs_boundary_p
        except AttributeError:
            self.abs_boundary_p = "{cuda_real}(1.)".format(**self.cuda_types)

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        kwargs.update(dX=self.dX, dP=self.dP,)

        # Declear datatypes
        self.preamble = "#include <math.h>\n\n#define _USE_MATH_DEFINES\n"

        # Convert real constants into CUDA code
        for name, value in kwargs.items():
            if isinstance(value, int):
                self.preamble += "#define %s %d\n" % (name, value)
            elif isinstance(value, float):
                self.preamble += "#define {name} {cuda_real}({:.20})\n".format(value, name=name, **self.cuda_types)

        # Append user defined constants, if specified
        try:
            self.preamble += self.consts
        except AttributeError:
            pass

        # Declare potential and kinetic energy functions
        self.preamble += \
            "\n// Potential energy" \
            "\n__device__ {cuda_real} V(const {cuda_real} X, const {cuda_real} t)" \
            "\n{{\n\treturn ({V}); \n}}\n" \
            "\n// Kinetic energy" \
            "\n__device__ {cuda_real} K(const {cuda_real} P, const {cuda_real} t)" \
            "\n{{\n\treturn ({K}); \n}}\n".format(V=self.V, K=self.K, **self.cuda_types)

        # Append all other user defined functions, if specified
        try:
            self.preamble += self.functions
        except AttributeError:
            pass

        ##########################################################################################
        #
        # Load FAFT
        #
        ##########################################################################################

        faft_dir = os.getcwd() + "/FAFT/FAFT_%d-points_C2C/FAFT%d_1D_C2C.so" % (self.X_gridDIM, 2 * self.X_gridDIM)

        _faft_1D = ctypes.cdll.LoadLibrary(faft_dir)

        self.cuda_faft = getattr(_faft_1D, "FAFT%d_1D_C2C" % (2 * self.X_gridDIM))

        self.cuda_faft.restype = int
        self.cuda_faft.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int]

        # Parameters
        self.segment = 0
        self.delta = self.dX * self.dP / (2 * np.pi)

        ##########################################################################################
        #
        # Set CUDA functions for propagation
        #
        ##########################################################################################

        self.expV = ElementwiseKernel(
            arguments="{cuda_complex} *psi, const {cuda_real} t".format(**self.cuda_types),

            operation="""
                    const {cuda_real} X = dX * (i - 0.5 * X_gridDIM);
                    const {cuda_real} phase = -0.5 * dt * V(X, t);

                    psi[i] *= {cuda_complex}(cos(phase), sin(phase)) * ({abs_boundary_x});
                    """.format(abs_boundary_x=self.abs_boundary_x, **self.cuda_types),

            preamble=self.preamble
        )

        self.expK = ElementwiseKernel(
            arguments="{cuda_complex} *psi, const {cuda_real} t".format(**self.cuda_types),

            operation="""
                    const {cuda_real} P = dP * (i - 0.5 * X_gridDIM);
                    const {cuda_real} phase = -dt * K(P, t);

                    psi[i] *= {cuda_complex}(cos(phase), sin(phase)) * ({abs_boundary_p});
                    """.format(abs_boundary_p=self.abs_boundary_p, **self.cuda_types),

            preamble=self.preamble
        )

        self.norm_square = ReductionKernel(
            self.wavefunction.real.dtype,
            arguments="const {cuda_complex} *psi".format(**self.cuda_types),
            neutral="0.",
            reduce_expr="a + b",
            map_expr="pow(abs(psi[i]), 2)",
        )

        ##########################################################################################
        #
        #   Ehrenfest theorems (optional)
        #
        ##########################################################################################

        # hash table of cuda compiled functions that calculate an average of specified observable
        self.compiled_observable = dict()

        try:
            # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
            self.diff_K
            self.diff_V

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def get_average(self, observable):
        """
        Return the expectation value of the observable with respect to the current wave function
        :param observable: (str)
        :return: float
        """
        # Compile the corresponding cuda functions, if it has not been done
        try:
            func = self.compiled_observable[observable]
        except KeyError:
            # Due to the way ReductionKernel is implemented,
            # we need to declare coordinate and momenta as macros
            preamble = self.preamble \
                        + "\n#define X (dX * (i - 0.5 * X_gridDIM))" \
                        + "\n#define P (dP * (i - 0.5 * X_gridDIM))"

            func = self.compiled_observable[observable] = ReductionKernel(
                self.wavefunction.real.dtype,
                arguments="const {cuda_complex} *psi".format(**self.cuda_types),
                neutral="0.",
                reduce_expr="a + b",
                map_expr="pow(abs(psi[i]), 2) * (%s)" % observable,
                preamble=preamble,
            )
        return func(self._tmp).get()

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time
        :param t: current time
        :return: coordinate and momentum densities, if the Ehrenfest theorems were calculated; otherwise, return None
        """
        if self.isEhrenfest:

            # Make a copy of the wave function
            gpuarray._memcpy_discontig(self._tmp, self.wavefunction)

            #########################################################################
            #
            #   Working in the coordinate representation
            #
            #########################################################################

            # Normalize
            self._tmp /= np.sqrt(self.norm_square(self._tmp).get())

            # save the current value of <X>
            self.X_average.append(self.get_average("X"))

            # save the current value of <-diff_V>
            self.P_average_RHS.append(-self.get_average(self.diff_V))

            # save the potential energy
            self.hamiltonian_average.append(self.get_average(self.V))

            #########################################################################
            #
            #   Working in the momentum representation
            #
            #########################################################################

            # Go to the momentum representation
            self.FAFT(self._tmp)

            # Normalize
            self._tmp /= np.sqrt(self.norm_square(self._tmp).get())

            # save the current value of <diff_K>
            self.X_average_RHS.append(self.get_average(self.diff_K))

            # save the current value of <P>
            self.P_average.append(self.get_average("P"))

            # add the expectation value for the kinetic energy
            self.hamiltonian_average[-1] += self.get_average(self.K)

    def set_wavefunction(self, new_wave_func):
        """
        Set the wave function
        :param new_wave_func: CUDA C string
        :return: self
        """

        # Initialize the function for assigning values
        init_func = ElementwiseKernel(
            arguments="{cuda_complex} *psi".format(**self.cuda_types),

            operation=("""
                const {cuda_real} X = dX * (i - 0.5 * X_gridDIM);
                psi[i] = %s; """ % new_wave_func
            ).format(**self.cuda_types),

            preamble=self.preamble
        )

        # set the values
        init_func(self.wavefunction)

        # normalize
        self.wavefunction /= np.sqrt(self.norm_square(self.wavefunction).get() * self.dX)

        return self

    @classmethod
    def print_memory_info(cls):
        """
        Print the CUDA memory info
        :return:
        """
        print(
            "\n\n\t\tGPU memory Total %.2f GB\n\t\tGPU memory Free %.2f GB\n" % \
            tuple(np.array(pycuda.driver.mem_get_info()) / 2. ** 30)
        )

    def propagate(self, steps=1):
        """
        Time propagate the density matrix saved in self.rho
        :param steps: number of self.dt time increments to make
        :return: self
        """
        for _ in xrange(steps):
            # increment current time
            self.t += self.dt

            # advance by one time step
            self.single_step_propagation()

            # normalize
            self.wavefunction /= np.sqrt(self.norm_square(self.wavefunction).get() * self.dX)

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

        return self

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final wave function is not normalized.
        :return: self.wavefunction
        """
        self.expV(self.wavefunction, self.t)

        self.FAFT(self.wavefunction)
        self.expK(self.wavefunction, self.t)
        self.iFAFT(self.wavefunction)

        self.expV(self.wavefunction, self.t)

        return self.wavefunction

    def FAFT(self, f):
        """
        Calculate the direct Fast Accurate Fourier Transform (FAFT)
        :param f: gpu.array (the source and destination of FAFT)
        :return: self
        """
        self.cuda_faft(int(f.gpudata), self.dX, self.delta, self.segment)
        return self

    def iFAFT(self, f):
        """
        Calculate the inverse Fast Accurate Fourier Transform (iFAFT)
        :param f: gpu.array (the source and destination of iFAFT)
        :return: self
        """
        self.cuda_faft(int(f.gpudata), self.dX, -self.delta, self.segment)
        return self

##########################################################################################
#
# Example
#
##########################################################################################

if __name__=='__main__':

    import matplotlib.pyplot as plt

    np.random.seed(7402164840)

    # Create propagator
    quant_sys = SchrodingerWignerCUDA1D(
        t=0.,
        dt=0.01,

        X_gridDIM=2048,

        X_amplitude=10.,
        P_amplitude=15.,

        # randomized parameter
        omega_square=np.random.uniform(2., 6.),

        # randomized parameters for initial condition
        sigma=np.random.uniform(0.5, 4.),
        p0=np.random.uniform(-1., 1.),
        x0=np.random.uniform(-1., 1.),

        # kinetic energy part of the hamiltonian
        K="0.5 * P * P",

        # potential energy part of the hamiltonian
        V="0.5 * omega_square * X * X",

        # these functions are used for evaluating the Ehrenfest theorems
        diff_K="P",
        diff_V="omega_square * X"
    )

    # set randomised initial condition
    quant_sys.set_wavefunction(
        "exp({cuda_complex}(-sigma * pow(X - x0, 2), p0 * X))"
    )

    quant_sys.propagate(1000)
    #evolution = [quant_sys.propagate(5).wavefunction.get() for _ in xrange(500)]
    #plt.imshow(np.abs(evolution)**2)
    #plt.show()

    # extract the reference to quantum system
    #quant_sys = visualizer.quant_sys

    # Analyze how well the energy was preserved
    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min() / h.max()) * 100)
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = dt * np.arange(len(quant_sys.X_average)) + dt

    plt.subplot(131)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial V/\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, h)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()
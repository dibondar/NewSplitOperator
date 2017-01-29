

from schrodinger_wigner_cuda_1d import SchrodingerWignerCUDA1D
#from test import SchrodingerWignerCUDA1D
import numpy as np

# load tools for creating animation
import sys
import matplotlib
import h5py

if sys.platform == 'darwin':
    # only for MacOS
    matplotlib.use('TKAgg')

import matplotlib.animation
import matplotlib.pyplot as plt

##########################################################################################
#
#   Parameters of quantum systems
#
##########################################################################################

# Soft-core Coulomb potential
coulomb_sys_params = dict(
    t=0.,
    dt=0.01,

    X_gridDIM=2048,

    # the lattice constant is 2 * X_amplitude
    X_amplitude=80.,

    P_amplitude=20.,

    # frequency of laser field (800nm)
    omega=0.05698,

    # field strength
    F=0.04,

    # ionization potential
    Ip=0.59,

    functions="""
    // # The laser field (the field will be on for 8 periods of laser field)
    __device__ {cuda_real} E(const {cuda_real} t)
    {{
        return -F * sin(omega * t) * pow(sin(omega * t / 16.), 2);
    }}
    """,

    abs_boundary_x="pow(abs(sin(0.5 * M_PI * (X + X_amplitude) / X_amplitude)), dt * 0.02)",

    # The same as C code
    E=lambda self, t: -self.F * np.sin(self.omega * t) * np.sin(self.omega * t / 16.)**2,

    ##########################################################################################
    #
    # Specify system's hamiltonian
    #
    ##########################################################################################

    # the kinetic energy
    K="0.5 * P * P",

    # derivative of the kinetic energy to calculate Ehrenfest
    diff_K="P",

    # the soft core Coulomb potential for Ar
    V = "-1. / sqrt(X * X + 1.37) + X * E(t)",

    # the derivative of the potential to calculate Ehrenfest
    diff_V="X / pow(X * X + 1.37, 1.5) + E(t)",
)

# Soft-core short range potential
short_sys_params = coulomb_sys_params.copy()
short_sys_params.update(
    # the soft core short range potential
    V="-1. / sqrt(1.37) * exp(-0.2246 * X * X) + X * E(t)",

    # the derivative of the potential to calculate Ehrenfest
    #diff_V="2. * 1.02 * sqrt(1.37) * exp(-1.02 * X * X) * X + E(t)",
)

##########################################################################################
#
#   Visualization
#
##########################################################################################

class VisualizeDynamics:
    """
    Class to visualize dynamics in phase space.
    """

    def __init__(self, fig, coulomb_sys_params, short_sys_params, file_results):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        :param coulomb_sys_params: dictionary of parameters to initialize Coulomb potential
        :param short_sys_params: dictionary of parameters to initialize short range potential
        :param file_results: HDF5 file to save results
        """
        self.fig = fig
        self.coulomb_sys_params = coulomb_sys_params
        self.short_sys_params = short_sys_params
        self.file_results = file_results

        #  Initialize systems
        self.set_quantum_sys()

        #################################################################
        #
        # Save quantum system's parameters into the HDF5 file
        #
        #################################################################

        self.settings_grp = self.file_results.create_group("settings")

        for key, val in coulomb_sys_params.items():
            try:
                self.settings_grp[key] = val
            except TypeError:
                pass

        # Create group where each frame will be saved
        self.frames_grp = self.file_results.create_group("frames")

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        ax = fig.add_subplot(311)

        ax.set_title("Coordinate probability density")
        self.x_rho_plot, = ax.semilogy([self.coulomb_sys.X.min(), self.coulomb_sys.X.max()], [1e-8, 2e-1])
        ax.set_ylabel('$\\left|\\Psi(x,t)\\right|^2$ (a.u.)')
        ax.set_xlabel('$x$ (a.u.)')

        ax = fig.add_subplot(312)

        ax.set_title("Momentum probability density")
        self.p_rho_plot, = ax.plot([self.coulomb_sys.P.min(), self.coulomb_sys.P.max()], [1e-8, 2e-1])
        ax.set_ylabel('$\\left|\\Psi(p,t)\\right|^2$ (a.u.)')
        ax.set_xlabel('$p$ (a.u.)')

        ax = fig.add_subplot(313)
        F = self.coulomb_sys.F
        self.laser_filed_plot, = ax.plot([0., self.T_final], [-F, F])
        ax.set_xlabel('time (a.u.)')
        ax.set_ylabel('Laser field $E(t)$ (a.u.)')

    def set_quantum_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """
        # Create propagator
        self.coulomb_sys = SchrodingerWignerCUDA1D(**self.coulomb_sys_params)

        print "\nGround state energy for the Coulomb potential %f (a.u.)\n" % \
              self.coulomb_sys.set_ground_state().get_energy()

        #self.short_sys = SchrodingerWignerCUDA1D(**self.short_sys_params)
        self.short_sys = None
        #print "\nGround state energy for the short-range potential %f (a.u.) \n" % \
        #      self.short_sys.set_ground_state().get_energy()

        # Constant specifying the duration of simulation

        # final propagation time
        self.T_final = 8 * 2 * np.pi / self.coulomb_sys.omega

        # Number of steps before plotting
        self.num_iteration = 200

        # Number of frames
        self.num_frames = int(np.ceil(self.T_final / self.coulomb_sys.dt / self.num_iteration))

        self.current_frame_num = 0

        # List to save times
        self.times = [self.coulomb_sys.t]

    def empty_frame(self):
        """
        Make empty frame and reinitialize quantum system
        :param self:
        :return: image object
        """
        self.x_rho_plot.set_data([], [])
        self.p_rho_plot.set_data([], [])
        self.laser_filed_plot.set_data([], [])

        return self.x_rho_plot, self.p_rho_plot, self.laser_filed_plot

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        # propagate
        self.coulomb_sys.propagate(self.num_iteration)
        #self.short_sys.propagate(self.num_iteration)

        self.x_rho_plot.set_data(
            self.coulomb_sys.X,
            np.abs(self.coulomb_sys.wavefunction.get()) ** 2
        )

        self.p_rho_plot.set_data(
            self.coulomb_sys.P,
            np.abs(self.coulomb_sys._tmp.get()) ** 2
        )

        # prepare goup where simulations for the current frame will be saved
        # frame_grp = self.frames_grp.create_group(str(self.current_frame_num))

        #frame_grp["wigner"] = wigner
        #frame_grp["t"] = self.coulomb_sys.t

        # Extract the diagonal of the density matrix
        #frame_grp["prob"] = np.abs(self.coulomb_sys.wavefunction.get())**2

        print("Frame : %d / %d" % (self.current_frame_num, self.num_frames))
        self.current_frame_num += 1

        self.times.append(self.coulomb_sys.t)

        t = np.array(self.times)
        self.laser_filed_plot.set_data(t, self.coulomb_sys.E(t))

        return self.x_rho_plot, self.p_rho_plot, self.laser_filed_plot

with h5py.File('strong_field_physics.hdf5', 'w') as file_results:
    fig = plt.gcf()
    visualizer = VisualizeDynamics(fig, coulomb_sys_params, short_sys_params, file_results)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=min(881, visualizer.num_frames),
        init_func=visualizer.empty_frame, blit=True, repeat=False
    )

    plt.show()

    # Set up formatting for the movie files
    # writer = matplotlib.animation.writers['mencoder'](fps=10, metadata=dict(artist='Denys Bondar'), bitrate=-1)

    # Save animation into the file
    # animation.save('strong_field_physics.mp4', writer=writer)

    # extract the reference to quantum system
    coulomb_sys = visualizer.coulomb_sys

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################
    # Analyze how well the energy was preserved
    h = np.array(coulomb_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min() / h.max()) * 100)
    )

    # generate time step grid
    dt = coulomb_sys.dt
    times = dt * np.arange(len(coulomb_sys.X_average)) + dt

    plt.subplot(131)
    plt.title("Ehrenfest 1")
    plt.plot(times, np.gradient(coulomb_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, coulomb_sys.X_average_RHS, 'b--', label='$\\langle p + \\gamma x \\rangle$')

    plt.legend(loc='upper left')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("Ehrenfest 2")

    plt.plot(times, np.gradient(coulomb_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(
        times, coulomb_sys.P_average_RHS, 'b--',
        label='$\\langle -\\partial V/\\partial x  + \\gamma p \\rangle$'
    )

    plt.legend(loc='upper left')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, coulomb_sys.hamiltonian_average)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    # #################################################################
    # #
    # # Plot HHG spectra as FFT(<P>)
    # #
    # #################################################################
    #
    # N = len(coulomb_sys.P_average)
    #
    # # the windowed fft of the evolution
    # # to remove the spectral leaking. For details see
    # # rhttp://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    # from scipy import fftpack
    # from scipy.signal import blackman
    #
    # # obtain the dipole
    # J = np.array(coulomb_sys.P_average)
    #
    # fft_J = fftpack.fft(blackman(N) * J)
    # #fft_J = fftpack.fft(J)
    # spectrum = np.abs(fftpack.fftshift(fft_J))**2
    # omegas = fftpack.fftshift(fftpack.fftfreq(N, coulomb_sys.dt/(2*np.pi))) / coulomb_sys.omega
    #
    #
    # spectrum /= spectrum.max()
    #
    # plt.semilogy(omegas, spectrum)
    # plt.ylabel('spectrum FFT($\\langle p \\rangle$)')
    # plt.xlabel('frequency / $\\omega$')
    # plt.xlim([0, 100.])
    # plt.ylim([1e-20, 1.])
    #
    # plt.show()
    #
    # #################################################################
    # #
    # # Saving Ehrenfest theorem results into HDF5 file
    # #
    # #################################################################
    #
    # ehrenfest_grp = file_results.create_group("ehrenfest")
    # ehrenfest_grp["X_average"] = coulomb_sys.X_average
    # ehrenfest_grp["P_average"] = coulomb_sys.P_average
    # ehrenfest_grp["X_average_RHS"] = coulomb_sys.X_average_RHS
    # ehrenfest_grp["P_average_RHS"] = coulomb_sys.P_average_RHS
    # ehrenfest_grp["hamiltonian_average"] = coulomb_sys.hamiltonian_average
    # #ehrenfest_grp["wigner_time"] = coulomb_sys.wigner_time
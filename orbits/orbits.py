"""
orbits.py

Module for simulating and analyzing two-body orbits (classical and relativistic)
with various umerical integration methods.
"""


# Importing libraries

import os
import argparse

import scipy.constants as const
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import pandas as pd


# Defining the constants
G = const.G # [m^3 kg^-1 s^-2]
c = const.c # [m s^-1]
AU = const.au # [m]
MASS_SUN = 1.988416e30 # [kg]
YEAR = 365.25 * 24 * 3600 # [s]


# Getting G and c in AU
G_AU = G / (AU**3 / (MASS_SUN * YEAR**2)) # [AU^3 solar_mass^-1 year^-2]
c_AU = c / (AU / YEAR) # [AU year^-1]



# Defining the main class

class TwoBodyProblem:
    """
    Simulates the motion of a planet around a black hole using the two-body problem.
    """


    def __init__(self, ecc = 0., mass_bh = 5.e6, sm_axis = 1., orb_period = 1.,
                 method = 'Trapezoidal', relativity = False, dt = None):
        """
        Initialize the TwoBodyProblem class.

        Parameters
        ----------
        ecc : float
            Eccentricity of the orbit.
        mass_bh : float
            Mass of the black hole in solar masses.
        sm_axis : float
            Semi-major axis of the orbit in AU.
        orb_period : float
            Number of orbital periods to simulate.
        method : str
            Numerical method for solving the differential equation ('Trapezoidal', 
            'RK3', or 'Scipy').
        relativity : bool
            Whether to include relativistic effects.
        dt : float or None
            Time step for the simulation in years. If None, a default is used.

        Returns
        -------
        None
        """

        # Defining valid methods
        valid_methods = ['Trapezoidal', 'RK3', 'Scipy']
        if method not in valid_methods:
            raise ValueError(
                f"Method not recognized: '{method}'."
                f"Choose one of: {', '.join(valid_methods)}"
            )

        # Setting the parameters
        self.ecc = ecc # [-]
        self.mass_bh = mass_bh # [solar masses]
        self.sm_axis = sm_axis # [AU]
        self.orb_period = orb_period # [-]
        self.method = method # [str]
        self.relativity = relativity
        self.dt = dt # [years]

        # Initial conditions
        self.r0 = np.array([0, self.sm_axis * (1 - self.ecc)]) # [AU]
        self.v0 = np.array([np.sqrt(G_AU * self.mass_bh / self.sm_axis * (1 + self.ecc) /
                                    (1 - self.ecc)), 0]) # [AU/year]

        # Initial state vector
        self.s0 = np.array([self.r0[0], self.r0[1], self.v0[0], self.v0[1]])

        # Schwarzschild radius
        self.r_s = 2 * G_AU * self.mass_bh / c_AU**2 # [AU]

        # Kepler's third law
        self.period = 2 * np.pi * np.sqrt(self.sm_axis**3 / (G_AU * self.mass_bh)) # [years]


    def grid_generator(self, save_map = False):
        """
        Generate a 2D grid for orbit visualization, masking out the Schwarzschild radius.

        Parameters
        ----------
        save_map : bool, optional
            Whether to save the grid as a .npy file (default is False).

        Returns
        -------
        x_2d : np.ndarray
            X coordinates of the grid.
        y_2d : np.ndarray
            Y coordinates of the grid.

        Raises
        ------
        ValueError
            If the semi-major axis is not positive.
        """

        # Checking for correct semi-major axis input
        if self.sm_axis <= 0:
            raise ValueError('Semi-major axis must be positive.')

        # Generating the grid
        x_2d = np.linspace(-self.sm_axis * (1 + self.ecc), self.sm_axis * (1 + self.ecc), 100)
        y_2d = np.linspace(-self.sm_axis * (1 + self.ecc), self.sm_axis * (1 + self.ecc), 100)
        x_2d, y_2d = np.meshgrid(x_2d, y_2d)

        # Adding the Schwarzschild radius
        r_2d = np.sqrt(x_2d**2 + y_2d**2)
        mask = r_2d < self.r_s
        x_2d[mask] = np.nan
        y_2d[mask] = np.nan

        # Saving the grid
        if save_map:
            output_dir = 'outputfolder/grids'
            os.makedirs(output_dir, exist_ok=True)

            # Corrected filename generation
            grid_filename = (
                f"grid_e{self.ecc}"
                f"_mass_bh{self.mass_bh:.1e}"
                f"_sm_axis{self.sm_axis}"
                f"_orb_period{self.orb_period}"
                f"_{'relat' if self.relativity else 'class'}"
                f"_{self.method}.npy"
            )

            filepath = os.path.join(output_dir, grid_filename)

            np.save(filepath, np.array([x_2d, y_2d]))
            print(f'Grid saved to {filepath}')

        return x_2d, y_2d



    def slope(self, t, s):
        """
        Calculate the slopes (classical or relativistic) for the two-body problem.

        Parameters
        ----------
        t : float
            Time.
        s : np.ndarray
            State vector [x, y, vx, vy].

        Returns
        -------
        dsdt : np.ndarray
            Derivatives [vx, vy, ax, ay].
        """

        x, y, vx, vy = s
        r = np.sqrt(x**2 + y**2)

        # Angular momentum
        ang_mom = x * vy - y * vx

        # Relativistic correction
        if self.relativity is True:
            correction = 1 + (3 * ang_mom**2) / (r**2 * c_AU**2)
        else:
            correction = 1

        # Acceleration
        a = G_AU * self.mass_bh / r**2 * correction
        ax = -a * x / r
        ay = -a * y / r

        return np.array([vx, vy, ax, ay])


    def trapezoidal_euler(self, t_span, dt):
        """
        Solve the ODE using the trapezoidal Euler method.

        Parameters
        ----------
        t_span : list or tuple
            Time span as [t_start, t_end].
        dt : float
            Time step.

        Returns
        -------
        t : np.ndarray
            Time array.
        s : np.ndarray
            State vector array.
        """

        # Setting the time array
        t = np.arange(t_span[0], t_span[1], dt)

        # Empty array for the state vector
        s = np.zeros((len(t), 4))

        # Initial state vector
        s[0] = self.s0

        # Looping over the time array
        for j in range(0, len(t) - 1):
            s[j + 1] = s[j] + dt * self.slope(t[j], s[j]) # Predictor step
            s[j + 1] = s[j] + dt * (self.slope(t[j], s[j]) +
                                    self.slope(t[j + 1], s[j + 1])) / 2 # Corrector step
        return t, s


    def runge_kutta_3(self, t_span, dt):
        """
        Solve the ODE using the Runge-Kutta 3 method.

        Parameters
        ----------
        t_span : list or tuple
            Time span as [t_start, t_end].
        dt : float
            Time step.

        Returns
        -------
        t : np.ndarray
            Time array.
        s : np.ndarray
            State vector array.
        """

        # Setting the time array
        t = np.arange(t_span[0], t_span[1], dt)

        # Empty array for the state vector
        s = np.zeros((len(t), 4))

        # Initial state vector
        s[0] = self.s0

        # Looping over the time array
        for j in range(0, len(t) - 1):
            slope_1 = self.slope(t[j], s[j])
            slope_2 = self.slope(t[j] + 3/4 * dt, s[j] + 3/4 * dt * slope_1)
            slope_3 = self.slope(t[j] + 1/4 * dt, s[j] - 5/12 * dt * slope_1 + 2/3 * dt * slope_2)
            s[j + 1] = s[j] + dt * (slope_1 + 5 * slope_2 + 3 * slope_3) / 9
        return t, s


    def scipy_integration(self, t_span, dt):
        """
        Solve the ODE using SciPy's DOP853 integration method.

        Parameters
        ----------
        t_span : list or tuple
            Time span as [t_start, t_end].
        dt : float
            Time step.

        Returns
        -------
        t : np.ndarray
            Time array.
        s : np.ndarray
            State vector array.
        """

        # Setting the time array
        t = np.arange(t_span[0], t_span[1], dt)

        # Empty array for the state vector
        s = np.zeros((len(t), 4))

        # Initial state vector
        s[0] = self.s0

        # Solving the ODE
        sol = solve_ivp(self.slope, [t[0], t[-1]], self.s0, method = 'DOP853', t_eval = t)
        return sol.t, sol.y.T



# Defining the runner class

class SimulationRunner:
    """
    Runs the simulation of the two-body problem.
    """


    def __init__(self, two_body_problem):
        """
        Initialize the SimulationRunner.

        Parameters
        ----------
        two_body_problem : TwoBodyProblem
            Instance of the TwoBodyProblem class.

        Returns
        -------
        None
        """

        self.two_body_problem = two_body_problem


    def run_simulation(self):
        """
        Run the simulation.

        Returns
        -------
        t : np.ndarray
            Time array.
        s : np.ndarray
            State vector array.
        """

        # Setting the time span
        t_span = [0, self.two_body_problem.orb_period * self.two_body_problem.period]

        # Setting the time step
        if self.two_body_problem.dt is None:
            dt = self.two_body_problem.period / 200 # 200 steps per orbit
        else:
            dt = self.two_body_problem.dt

        # Check if the number of time steps is too large
        max_steps = 1e6  # Set a reasonable limit for the number of steps
        num_steps = (t_span[1] - t_span[0]) / dt
        if num_steps > max_steps:
            raise ValueError(
                f"The number of time steps ({num_steps:.0f}) is too large. "
                f"Consider increasing the time step (dt) or reducing the number orbital periods."
            )

        # Running the simulation
        if self.two_body_problem.method == 'Trapezoidal':
            t, s = self.two_body_problem.trapezoidal_euler(t_span, dt)
        elif self.two_body_problem.method == 'RK3':
            t, s = self.two_body_problem.runge_kutta_3(t_span, dt)
        elif self.two_body_problem.method == 'Scipy':
            t, s = self.two_body_problem.scipy_integration(t_span, dt)
        else:
            raise ValueError('Method not recognized. Choose from Trapezoidal, RK3 or Scipy.')

        # Creating output directory if it doesn't exist
        output_dir = 'outputfolder/orbits_data'
        os.makedirs(output_dir, exist_ok = True)

        # Generating filename based on simulation parameters
        filename = (
            f"orbit_e{self.two_body_problem.ecc}"
            f"_mass_bh{self.two_body_problem.mass_bh:.1e}"
            f"_sm_axis{self.two_body_problem.sm_axis}"
            f"_orb_period{self.two_body_problem.orb_period}"
            f"_{'relat' if self.two_body_problem.relativity else 'class'}"
            f"_{self.two_body_problem.method}.csv"
        )

        # Creating data frame
        orbital_history = pd.DataFrame({
            'time': t,
            'x': s[:, 0],
            'y': s[:, 1],
            'vx': s[:, 2],
            'vy': s[:, 3],
            'r': np.sqrt(s[:, 0]**2 + s[:, 1]**2),
            'v': np.sqrt(s[:, 2]**2 + s[:, 3]**2)
        })

        # Saving to CSV
        filepath = os.path.join(output_dir, filename)
        orbital_history.to_csv(filepath, index = False)

        return t, s



# Defining a class with analysis tools

class AnalysisTools:
    """
    Contains analysis methods for comparing simulations.
    """


    @staticmethod
    def compare_rel_vs_class(t_class, s_class, t_relat, s_relat):
        """
        Compare relativistic vs classical orbits and plot the deviation.

        Parameters
        ----------
        t_class : np.ndarray
            Time array for classical simulation.
        s_class : np.ndarray
            State vector for classical simulation.
        t_relat : np.ndarray
            Time array for relativistic simulation.
        s_relat : np.ndarray
            State vector for relativistic simulation.

        Returns
        -------
        None
        """

        # Calculate and plot deviation
        deviation = np.sqrt((s_class[:,0] - s_relat[:,0])**2 + (s_class[:,1] - s_relat[:,1])**2)

        plt.figure(figsize = (10, 6))
        plt.plot(t_class, deviation, linewidth = 2)
        plt.xlabel('Time [years]')
        plt.ylabel('Deviation [AU]')
        plt.title('Orbital Deviation Between Classical and Relativistic Cases')
        plt.grid(True, linestyle = '--')
        plt.show()


    @staticmethod
    def compare_methods(results):
        """
        Compare different integration methods and plot their differences.

        Parameters
        ----------
        results : list of tuple
            Each tuple is (t, s, label) for a method.

        Returns
        -------
        None
        """

        # Calculating and plotting the difference relative to the most accurate method
        ref_t, ref_s, ref_label = results[-1]

        plt.figure(figsize = (10, 6))
        for _, s, label in results[:-1]:
            s_interp = interp1d(ref_t, s, axis = 0, fill_value = 'extrapolate')

            # Calculating the difference
            diff = np.sqrt((s_interp(ref_t)[:,0] - ref_s[:,0])**2 +
                           (s_interp(ref_t)[:,1] - ref_s[:,1])**2)

            plt.plot(ref_t, diff, label = f'{label} vs {ref_label}', linewidth = 2)

        plt.xlabel('Time [years]')
        plt.ylabel('Position Difference [AU]')
        plt.title('Integration Methods Comparison')
        plt.legend()
        plt.grid(True, linestyle = '--')
        plt.show()


    @staticmethod
    def convergence_analysis(ref_sol, solutions_dict, dts):
        """
        Analyze convergence of different methods.

        Parameters
        ----------
        ref_sol : tuple
            Reference solution as (t, s).
        solutions_dict : dict
            Dictionary of solutions {method_name: [(t, s), ...]}.
        dts : list of float
            Time steps used.

        Returns
        -------
        errors_dict : dict
            Dictionary of RMS errors {method_name: [errors]}.
        """

        errors_dict = {}

        ref_t, ref_s = ref_sol

        for method, solutions in solutions_dict.items():
            errors = []
            for (t, s) in solutions:
                # Find overlapping time range
                t_min = max(t[0], ref_t[0])
                t_max = min(t[-1], ref_t[-1])
                ref_mask = (ref_t >= t_min) & (ref_t <= t_max)

                # Interpolate to reference time points
                interp_func = interp1d(t, s, axis=0)
                s_interp = interp_func(ref_t[ref_mask])

                # Calculate RMS error
                error = np.sqrt(np.mean(np.sum((s_interp - ref_s[ref_mask])**2, axis=1)))
                errors.append(error)

            errors_dict[method] = errors

        # Simple plotting
        plt.figure(figsize=(10, 6))
        for method, errors in errors_dict.items():
            plt.loglog(dts, errors, 'o-', linewidth = 2, label = method)

        plt.xlabel('Time step [years]')
        plt.ylabel('RMS Error [AU]')
        plt.title('Convergence Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()

        return errors_dict



# Defining the animation class

class AnimationCreator:
    """
    Creates various types of animations.
    """


    def __init__(self, two_body_problem):
        """
        Initialize the AnimationCreator.

        Parameters
        ----------
        two_body_problem : TwoBodyProblem
            Instance of the TwoBodyProblem class.

        Returns
        -------
        None
        """

        self.two_body_problem = two_body_problem


    def _get_axes_limits(self, results):
        """
        Determine axis limits based on all orbits in results.

        Parameters
        ----------
        results : list of tuple
            Each tuple contains (t, s, label).

        Returns
        -------
        x_lim : tuple
            (x_min, x_max) with margin.
        y_lim : tuple
            (y_min, y_max) with margin.
        """

        xs = []
        ys = []
        for _, s, *_ in results:
            xs.extend(s[:, 0])
            ys.extend(s[:, 1])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Adding a margin (10%)
        x_margin = 0.1 * (x_max - x_min)
        y_margin = 0.1 * (y_max - y_min)

        return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)


    def animate_rel_vs_class(self, t_class, s_class, t_relat, s_relat, save_gif = False):
        """
        Animate the comparison between classical and relativistic orbits.

        Parameters
        ----------
        t_class : np.ndarray
            Time array for classical simulation.
        s_class : np.ndarray
            State vector for classical simulation.
        t_relat : np.ndarray
            Time array for relativistic simulation.
        s_relat : np.ndarray
            State vector for relativistic simulation.
        save_gif : bool, optional
            Whether to save the animation as a GIF (default is False).

        Returns
        -------
        None
        """

        # Computing axis limits dynamically
        (x_lim, y_lim) = self._get_axes_limits([(t_class, s_class, 'Classical'),
                                                (t_relat, s_relat, 'Relativistic')])

        fig, ax = plt.subplots(figsize = (8, 8))
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect('equal')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_title('Classical vs Relativistic Orbits')

        # Schwarzschild radius
        schwarzschild = plt.Circle((0, 0), self.two_body_problem.r_s, color = 'black', fill = False)
        ax.add_patch(schwarzschild)

        # Black hole at the center
        ax.plot(0, 0, marker = 'o', color = 'black', markersize = 10)

        # Plot elements
        line_class, = ax.plot([], [], color = 'gray', linestyle = ':')
        point_class, = ax.plot([], [], marker = 'o', label = 'Classical')
        line_rel, = ax.plot([], [], color = 'gray', linestyle = ':')
        point_rel, = ax.plot([], [], marker = 'o', label = 'Relativistic')
        ax.legend(loc = 1)

        def update(frame):
            line_class.set_data([s_class[:frame,0]], [s_class[:frame,1]])
            point_class.set_data([s_class[frame,0]], [s_class[frame,1]])
            line_rel.set_data([s_relat[:frame,0]], [s_relat[:frame,1]])
            point_rel.set_data([s_relat[frame,0]], [s_relat[frame,1]])
            return line_class, point_class, line_rel, point_rel

        frames = min(len(t_class), len(t_relat))
        ani = FuncAnimation(fig, update, frames = frames, interval = 20, blit = True)

        # Saving the animation
        if save_gif:
            output_dir = 'outputfolder/animations'
            os.makedirs(output_dir, exist_ok = True)

            # Generating filename based on simulation parameters
            animation_filename = (
                f"animation_relat_class_comp_e{self.two_body_problem.ecc}"
                f"_mass_bh{self.two_body_problem.mass_bh:.1e}"
                f"_sm_axis{self.two_body_problem.sm_axis}"
                f"_orb_period{self.two_body_problem.orb_period}"
                f"_{self.two_body_problem.method}.gif"
            )

            filepath = os.path.join(output_dir, animation_filename)

            ani.save(filepath, writer = 'pillow', fps = 30)
            print(f'Gif saved to {filepath}')
            plt.close(fig)
        else:
            pass
        plt.close(fig)


    def animate_eccentricities(self, results, save_gif = False):
        """
        Animate the comparison between different eccentricities.

        Parameters
        ----------
        results : list of tuple
            Each tuple contains (t, s, label).
        save_gif : bool, optional
            Whether to save the animation as a GIF (default is False).

        Returns
        -------
        None
        """

        # Computing axis limits dynamically
        (x_lim, y_lim) = self._get_axes_limits(results)

        fig, ax = plt.subplots(figsize = (8, 8))
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect('equal')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_title('Orbits with Different Eccentricities')

        # Schwarzschild radius
        schwarzschild = plt.Circle((0, 0), self.two_body_problem.r_s, color = 'black', fill = False)
        ax.add_patch(schwarzschild)

        # Black hole at the center
        ax.plot(0, 0, marker = 'o', color = 'black', markersize = 10)

        # Create plot elements for each orbit
        lines = []
        points = []
        for _, _, label in results:
            line, = ax.plot([], [], color = 'gray', linestyle = ':')
            point, = ax.plot([], [], marker = 'o', label = label)
            lines.append(line)
            points.append(point)
        ax.legend(loc = 1)


        def update(frame):
            for i, (_, s, _) in enumerate(results):
                lines[i].set_data([s[:frame,0]], [s[:frame,1]])
                points[i].set_data([s[frame,0]], [s[frame,1]])
            return lines + points

        frames = min(len(t) for t, _, _ in results)
        ani = FuncAnimation(fig, update, frames = frames, interval = 20, blit = True)

        # Saving the animation
        if save_gif:
            output_dir = 'outputfolder/animations'
            os.makedirs(output_dir, exist_ok = True)

            # Generate filename for simulation animation
            animation_filename = (
                "animation_eccs_comp"
                f"_mass_bh{self.two_body_problem.mass_bh:.1e}"
                f"_sm_axis{self.two_body_problem.sm_axis}"
                f"_orb_period{self.two_body_problem.orb_period}"
                f"_{'relat' if self.two_body_problem.relativity else 'class'}"
                f"_{self.two_body_problem.method}.gif"
            )

            filepath = os.path.join(output_dir, animation_filename)

            ani.save(filepath, writer = 'pillow', fps = 30)
            print(f'Gif saved to {filepath}')
            plt.close(fig)
        else:
            pass
        plt.close(fig)


    def animate_methods(self, results, save_gif = False):
        """
        Animate the comparison between different integration methods.

        Parameters
        ----------
        results : list of tuple
            Each tuple contains (t, s, label).
        save_gif : bool, optional
            Whether to save the animation as a GIF (default is False).

        Returns
        -------
        None
        """

        # Computing axis limits dynamically
        (x_lim, y_lim) = self._get_axes_limits(results)

        fig, ax = plt.subplots(figsize = (8, 8))
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect('equal')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_title('Comparison of Integration Methods')

        # Schwarzschild radius
        schwarzschild = plt.Circle((0, 0), self.two_body_problem.r_s, color = 'black', fill = False)
        ax.add_patch(schwarzschild)

        # Black hole at the center
        ax.plot(0, 0, marker = 'o', color = 'black', markersize = 10)

        # Create plot elements for each method
        lines = []
        points = []
        for _, _, label in results:
            line, = ax.plot([], [], color = 'gray', linestyle = ':')
            point, = ax.plot([], [], marker = 'o', label = label)
            lines.append(line)
            points.append(point)
        ax.legend(loc = 1)

        def update(frame):
            for i, (_, s, _) in enumerate(results):
                lines[i].set_data([s[:frame,0]], [s[:frame,1]])
                points[i].set_data([s[frame,0]], [s[frame,1]])
            return lines + points

        frames = min(len(t) for t, _, _ in results)
        ani = FuncAnimation(fig, update, frames = frames, interval = 20, blit = True)

        # Saving the animation
        if save_gif:
            output_dir = 'outputfolder/animations'
            os.makedirs(output_dir, exist_ok = True)

            # Generate filename for simulation animation
            animation_filename = (
                f"animation_methods_comp_e{self.two_body_problem.ecc}"
                f"_mass_bh{self.two_body_problem.mass_bh:.1e}"
                f"_sm-axis{self.two_body_problem.sm_axis}"
                f"_orb_period{self.two_body_problem.orb_period}"
                f"_{'relat' if self.two_body_problem.relativity else 'class'}.gif"
            )

            filepath = os.path.join(output_dir, animation_filename)

            ani.save(filepath, writer = 'pillow', fps = 30)
            print(f'Gif saved to {filepath}')
            plt.close(fig)
        else:
            pass
        plt.close(fig)



# Defining the main function

def main():
    """
    Run the main program for the two-body problem simulator.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description = 'Two-body problem simulator')

    # Simulation parameters
    sim_params = parser.add_argument_group('Simulation Parameters')
    sim_params.add_argument('--ecc', type = float, default = 0, help = 'Eccentricity (default: 0)')
    sim_params.add_argument('--mass_bh', type = float, default = 5.e6,
                            help = 'Black hole mass in solar masses (default: 5.e6 [solar_mass])')
    sim_params.add_argument('--sm_axis', type = float, default = 1.,
                            help = 'Semi-major axis in AU (default: 1. [AU])')
    sim_params.add_argument('--orb_period', type = float, default = 2.,
                            help = 'Number of orbital periods (default: 2)')

    # Analysis options
    analysis = parser.add_argument_group('Analysis Options)')
    analysis.add_argument('--compare_rel_class', action = 'store_true',
                          help = 'Compare relativistic vs classical')
    analysis.add_argument('--compare_methods', action = 'store_true',
                          help = 'Compare integration methods')
    analysis.add_argument('--convergence', action = 'store_true',
                          help = 'Perform convergence analysis')

    # Output options
    output = parser.add_argument_group('Output Options')
    output.add_argument('--save_gif', action = 'store_true', help = 'Save animations as GIF files')
    output.add_argument('--save_grid', action = 'store_true',
                        help = 'Save the initial computation grid')

    args = parser.parse_args()

    analysis = AnalysisTools()
    animator = None

    # (i) Relativistic vs classical comparison
    if args.compare_rel_class:
        print('Running relativistic vs classical comparison...')

        # Classical
        system_class = TwoBodyProblem(args.ecc, args.mass_bh,
                                      args.sm_axis, args.orb_period, 'RK3', False)
        runner = SimulationRunner(system_class)
        t_class, s_class = runner.run_simulation()
        system_class.grid_generator(save_map = args.save_grid)

        # Relativistic
        system_rel = TwoBodyProblem(args.ecc, args.mass_bh,
                                    args.sm_axis, args.orb_period, 'RK3', True)
        runner = SimulationRunner(system_rel)
        t_relat, s_relat = runner.run_simulation()
        system_rel.grid_generator(save_map = args.save_grid)

        # Analysis and visualization

        analysis.compare_rel_vs_class(t_class, s_class, t_relat, s_relat)

        if not animator:
            animator = AnimationCreator(system_rel)
        animator.animate_rel_vs_class(t_class, s_class, t_relat, s_relat, save_gif = args.save_gif)

    # (m) Compare methods
    if args.compare_methods:
        print('Running integration method comparison...')
        methods = [('Trapezoidal', 'Trapezoidal'), ('RK3', 'RK3'), ('Scipy', 'Scipy')]
        results = []

        for label, method in methods:
            system = TwoBodyProblem(args.ecc, args.mass_bh,
                                    args.sm_axis, args.orb_period, method, True)
            system.grid_generator(save_map = args.save_grid)
            runner = SimulationRunner(system)
            t, s = runner.run_simulation()
            results.append((t, s, label))

        # Analysis and visualization

        analysis.compare_methods(results)

        if not animator:
            animator = AnimationCreator(system)
        animator.animate_methods(results, save_gif = args.save_gif)

    # (n) Convergence analysis
    if args.convergence:
        print('Running convergence analysis...')

        # Reference solution (smallest time step)
        system_ref = TwoBodyProblem(args.ecc, args.mass_bh,
                                    args.sm_axis, args.orb_period, 'Scipy', True)
        system_ref.grid_generator(save_map = args.save_grid)
        runner = SimulationRunner(system_ref)
        t_ref, s_ref = runner.run_simulation()

        # Defining different methods and time steps
        methods = ['Trapezoidal', 'RK3']
        dts = [system_ref.period/100, system_ref.period/200,
               system_ref.period/400, system_ref.period/800]

        # Preparing the solution for each method
        sols_dict = {}
        for method in methods:
            sols = []
            for dt in dts:
                system = TwoBodyProblem(args.ecc, args.mass_bh, args.sm_axis,
                                        int(system_ref.period/dt), method, relativity = True)
                system.dt = dt
                runner = SimulationRunner(system)
                t, s = runner.run_simulation()
                sols.append((t, s))
            sols_dict[method] = sols

        # Calling the convergence_analysis method
        analysis.convergence_analysis((t_ref, s_ref), sols_dict, dts)

if __name__ == '__main__':
    main()

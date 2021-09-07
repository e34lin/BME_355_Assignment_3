import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Circulation:
    """
    Model of systemic circulation from Ferreira et al. (2005), A Nonlinear State-Space Model
    of a Combined Cardiovascular System and a Rotary Pump, IEEE Conference on Decision and Control.
    """

    def __init__(self, HR, Emax, Emin, R1=1.0, R3=0.001):
        self.set_heart_rate(HR)

        self.Emin = Emin
        self.Emax = Emax
        self.non_slack_blood_volume = 250  # ml

        self.R1 = R1  # between .5 and 2
        self.R2 = .005
        self.R3 = R3
        self.R4 = .0398

        self.C2 = 4.4
        self.C3 = 1.33

        self.L = .0005

    def set_heart_rate(self, HR):
        """
        Sets several related variables together to ensure that they are consistent.
        :param HR: heart rate (beats per minute)
        """
        self.HR = HR
        self.tc = 60 / HR
        self.Tmax = .2 + .15 * self.tc  # contraction time

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
        :return: time derivatives of state variables
        """

        """
        WRITE CODE HERE
        Implement this by deciding whether the model is in a filling, ejecting, or isovolumic phase and using 
        the corresponding dynamic matrix. 

        As discussed in class, be careful about starting and ending the ejection phase. One approach is to check 
        whether the flow is >0, and another is to check whether x1>x3, but neither will work. The first won't start 
        propertly because flow isn't actually updated outside the ejection phase. The second won't end properly 
        because blood inertance will keep the blood moving briefly up the pressure gradient at the end of systole. 
        If the ejection phase ends in this time, the flow will remain non-zero until the next ejection phase. 
        """

        # Filling Phase
        if x[1] > x[0]:
            A = self.filling_phase_dynamic_matrix(t)

        # Ejection Phase
        elif (x[3] > 0 or x[0] > x[2]):
            A = self.ejection_phase_dynamic_matrix(t)

        # Isovolumic Phase
        else:
            A = self.isovolumic_phase_dynamic_matrix(t)

        return np.matmul(A, x)

    def isovolumic_phase_dynamic_matrix(self, t):
        """
        :param t: time (s; needed because elastance is a function of time)
        :return: A matrix for isovolumic phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt / el, 0, 0, 0],
                [0, -1 / (self.R1 * self.C2), 1 / (self.R1 * self.C2), 0],
                [0, 1 / (self.R1 * self.C3), -1 / (self.R1 * self.C3), 0],
                [0, 0, 0, 0]]

    def ejection_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt / el, 0, 0, -el],
                [0, -1 / (self.R1 * self.C2), 1 / (self.R1 * self.C2), 0],
                [0, 1 / (self.R1 * self.C3), -1 / (self.R1 * self.C3), 1 / self.C3],
                [1 / self.L, 0, -1 / self.L, -(self.R3 + self.R4) / self.L]]

    def filling_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """

        """
        WRITE CODE HERE
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt / el - el / self.R2, el / self.R2, 0, 0],
                [1 / (self.R2 * self.C2), -(self.R1 + self.R2) / (self.R1 * self.R2 * self.C2), 1 / (self.R1 * self.C2),
                 0],
                [0, 1 / (self.R1 * self.C3), -1 / (self.R1 * self.C3), 0],
                [0, 0, 0, 0]]

    def elastance(self, t):
        """
        :param t: time (needed because elastance is a function of time)
        :return: time-varying elastance
        """
        tn = self._get_normalized_time(t)
        En = 1.55 * np.power(tn / .7, 1.9) / (1 + np.power(tn / .7, 1.9)) / (1 + np.power(tn / 1.17, 21.9))
        return (self.Emax - self.Emin) * En + self.Emin

    def elastance_finite_difference(self, t):
        """
        Calculates finite-difference approximation of elastance derivative. In class I showed another method
        that calculated the derivative analytically, but I've removed it to keep things simple.
        :param t: time (needed because elastance is a function of time)
        :return: finite-difference approximation of time derivative of time-varying elastance
        """
        dt = .0001
        forward_time = t + dt
        backward_time = max(0, t - dt)  # small negative times are wrapped to end of cycle
        forward = self.elastance(forward_time)
        backward = self.elastance(backward_time)
        return (forward - backward) / (2 * dt)

    def simulate(self, total_time):
        """
        :param total_time: seconds to simulate
        :return: time, state (times at which the state is estimated, state vector at each time)
        """

        """
        WRITE CODE HERE
        Put all the blood pressure in the atria as an initial condition.
        """

        initial_condition = np.array([0, self.non_slack_blood_volume / self.C2, 0, 0])
        solution = solve_ivp(self.get_derivative, [0, total_time], initial_condition, max_step = 0.01, rtol = 1e-5, atol = 1e-8)

        return [solution.t, solution.y]

    def _get_normalized_time(self, t):
        """
        :param t: time
        :return: time normalized to self.Tmax (duration of ventricular contraction)
        """
        return (t % self.tc) / self.Tmax

    # For Question 3: Model Validation
    def get_ventricular_blood_volume(self, t, ventricular_pressure):

        # Assume V0 = 10 ml

        el = self.elastance(t)
        slack_volume = 10

        #print ((ventricular_pressure / el) + slack_volume)
        return (ventricular_pressure / el) + slack_volume

########################################################################################################################

# Question 2
def plot_question_2():
    # Using values from paper (Question 1)
    circ = Circulation(75, 2.0, 0.06)

    # Simulate for five seconds
    [time, states] = circ.simulate(5.0)

    left_ventricular_pressure = states[0, :]
    atrial_pressure = states[1, :]
    arterial_pressure = states[2, :]
    aortic_flow_rate = states[3, :]

    aortic_pressure = arterial_pressure + aortic_flow_rate * circ.R4

    plt.figure(dpi = 200)
    plt.plot(time, atrial_pressure, label = 'Atrial Pressure')
    plt.plot(time, left_ventricular_pressure, label = 'Ventricular Pressure')
    plt.plot(time, arterial_pressure, label = 'Arterial Pressure')
    plt.plot(time, aortic_pressure, label = 'Aortic Pressure Between D2 and R4')
    plt.title('Different States of Circulation VS Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend(loc = 'upper left')
    plt.show()

plot_question_2()

########################################################################################################################

# Question 3: Model Validation
def plot_pv_loops():

    # Expected ESPVR using Emax = 2.5 and V0 = 10
    x = np.linspace(40, 70, 100)
    y = 2.5 * (x - 10)

    # Simulate for 15 seconds, get ventricular pressure and volume

    # R1 = 1 mmHg s / mL
    circ_normal = Circulation(75, 2.5, 0.06)
    [t_normal, states_normal] = circ_normal.simulate(15.0)
    ventricular_pressure_normal = states_normal[0, :]
    ventricular_volume_normal = circ_normal.get_ventricular_blood_volume(t_normal, ventricular_pressure_normal)

    # R1 = 1.5 mmHg s / mL
    circ_2 = Circulation(75, 2.5, 0.06, R1 = 1.5)
    [t_2, states_2] = circ_2.simulate(15.0)
    ventricular_pressure_2 = states_2[0, :]
    ventricular_volume_2 = circ_2.get_ventricular_blood_volume(t_2, ventricular_pressure_2)

    # R1 = 2 mmHg s / mL
    circ_3 = Circulation(75, 2.5, 0.06, R1 = 2.0)
    [t_3, states_3] = circ_3.simulate(15.0)
    ventricular_pressure_3 = states_3[0,:]
    ventricular_volume_3 = circ_3.get_ventricular_blood_volume(t_3, ventricular_pressure_3)

    # R1 = 0.5 mmHg s / mL
    circ_4 = Circulation(75, 2.5, 0.06, R1 = 0.5)
    [t_4, states_4] = circ_4.simulate(15.0)
    ventricular_pressure_4 = states_4[0,:]
    ventricular_volume_4 = circ_4.get_ventricular_blood_volume(t_4, ventricular_pressure_4)

    plt.figure(dpi = 200)
    plt.plot(ventricular_volume_4[-175:],ventricular_pressure_4[-175:], label = 'R1 = 0.5 mmHg s / mL')
    plt.plot(ventricular_volume_normal[-175:],ventricular_pressure_normal[-175:], label = 'R1 = 1 mmHg s / mL')
    plt.plot(ventricular_volume_2[-175:],ventricular_pressure_2[-175:], label = 'R1 = 1.5 mmHg s / mL')
    plt.plot(ventricular_volume_3[-175:],ventricular_pressure_3[-175:], label = 'R1 = 2 mmHg s / mL')
    plt.plot(x, y, ':', label = 'y = 2.5(x-10)')
    plt.xlabel('Ventricular Volume (mL)')
    plt.ylabel('Ventricular Pressure (mmHg)')
    plt.title('Pressure-Volume (PV) Loops')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()

plot_pv_loops()


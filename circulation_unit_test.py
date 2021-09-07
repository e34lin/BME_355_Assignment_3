import numpy as np
from circulation import Circulation

# Question 3: Unit Test Verification
def ventricular_pressure_test():
    # Check that ventricular pressure is between 0 mmHg and 105 mmHg within 1 second period
    circ = Circulation(75, 2.0, 0.06)
    t, x = circ.simulate(1)

    assert np.all ((x[1, :]) > 0) and np.all ((x[1, :]) < 105)


def atrial_pressure_test():
    # Check that atrial pressure is between 5 mmHg and 60 mmHg within 1 second period
    circ = Circulation(75, 2.0, 0.06)
    t, x = circ.simulate(1)

    assert np.all ((x[1, :]) > 5) and np.all ((x[1, :]) < 60)


def arterial_pressure_test():
    # Check that ventricular pressure is between 0 mmHg and 95 mmHg within 1 second period
    circ = Circulation(75, 2.0, 0.06)
    t, x = circ.simulate(1)

    assert np.all ((x[1, :]) > 0) and np.all ((x[1, :]) < 95)


def aortic_pressure_test():
    # Check that aortic pressure is between 0 mmHg and 115 mmHg within 1 second period
    circ = Circulation(75, 2.0, 0.06)
    t, x = circ.simulate(1)

    aortic_pressure = x[2, :] + x[3, :] * circ.R4

    assert np.all ((aortic_pressure) >= 0) and np.all ((aortic_pressure) < 115)


if __name__ == "__main__":
    ventricular_pressure_test()
    atrial_pressure_test()
    arterial_pressure_test()
    aortic_pressure_test()
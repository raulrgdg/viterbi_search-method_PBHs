# Author: Gonzalo Morras Gutierrez
# E-mail: gonzalo.morras@estudiante.uam.es

import numpy as np
from lalsimulation import SimInspiralTaylorLength

M_sun = 1.98841e30
t_sun = 4.92549094830932e-6
d_sun = 1476.6250382504018
Mpc_m = 3.085677581491367e22


class myTaylorT3:
    """Implement TaylorT3 as outlined in arXiv:0907.0700."""

    def __init__(self, m1=None, m2=None, distance=None, inclination=0, sampling_rate=512, coal_time=0, f_ref=20, phi_ref=0):
        if None in (m1, m2, distance):
            raise Exception("m1, m2 and distance must be given to initialize class.")

        self.m1 = m1
        self.m2 = m2
        self.inclination = inclination
        self.distance = distance
        self.coal_time = coal_time
        self.sampling_rate = sampling_rate
        self.f_ref = f_ref
        self.phi_ref = phi_ref

        self.nu = m1 * m2 / ((m1 + m2) ** 2)
        self.Mtot_s = (m1 + m2) * t_sun
        self.Mtot_m = (m1 + m2) * d_sun
        self.delta_t = 1 / sampling_rate
        self.distance_M = distance * Mpc_m / self.Mtot_m

        self.theta_lso = self.theta(
            -SimInspiralTaylorLength(float(self.delta_t), float(m1 * M_sun), float(m2 * M_sun), float(f_ref), 0)
        )
        self.x_lso = (np.pi * f_ref * self.Mtot_s) ** (2 / 3)

    def theta(self, time):
        return (self.nu * (self.coal_time - time) / (5 * self.Mtot_s)) ** (-1 / 8)

    def time_array(self, t1, t2):
        if t1 > t2:
            raise Exception("The start time can not be larger than the end time")
        if t2 > self.coal_time:
            raise Exception("The end time can not be larger than the coalescence time.")
        return t1 + np.arange(int((t2 - t1) * self.sampling_rate) + 1) / self.sampling_rate

    def freq(self, time):
        theta = self.theta(time)
        M = self.Mtot_s
        nu = self.nu
        return ((theta**3) / (8 * np.pi * M)) * (
            1
            + ((743 / 2688) + (11 / 32) * nu) * theta**2
            - (3 / 10) * np.pi * theta**3
            + ((1855099 / 14450688) + (56975 / 258048) * nu + (371 / 2048) * nu**2) * theta**4
            - ((7729 / 21504) - (13 / 256) * nu) * np.pi * theta**5
            + (
                -(720817631400877 / 288412611379200)
                + (53 / 200) * np.pi**2
                + (107 / 280) * np.euler_gamma
                + ((25302017977 / 4161798144) - (451 / 2048) * np.pi**2) * nu
                - (30913 / 1835008) * nu**2
                + (235925 / 1769472) * nu**3
                + (107 / 280) * np.log(2 * theta)
            )
            * theta**6
            + (-(188516689 / 433520640) - (97765 / 258048) * nu + (141769 / 1290240) * nu**2) * np.pi * theta**7
        )

    def phi(self, time):
        theta = self.theta(time)
        nu = self.nu
        theta_lso = self.theta_lso
        phi_ref = self.phi_ref
        return phi_ref - (1 / (nu * theta**5)) * (
            1
            + ((3715 / 8064) + (55 / 96) * nu) * theta**2
            - ((3 * np.pi) / 4) * theta**3
            + ((9275495 / 14450688) + (284875 / 258048) * nu + (1855 / 2048) * nu**2) * theta**4
            + ((38645 / 21504) - (65 / 256) * nu) * np.log((theta / theta_lso)) * np.pi * theta**5
            + (
                (831032450749357 / 57682522275840)
                - (53 / 40) * np.pi**2
                + (-(126510089885 / 4161798144) + (2255 / 2048) * np.pi**2) * nu
                - (107 / 56) * np.euler_gamma
                + (154565 / 1835008) * nu**2
                - (1179625 / 1769472) * nu**3
                - (107 / 56) * np.log(2 * theta)
            )
            * theta**6
            + ((188516689 / 173408256) + (488825 / 516096) * nu - (141769 / 516096) * nu**2) * np.pi * theta**7
        )

    def tdstrain(self, t1, t2, PyCBC_TimeSeries=False):
        time = self.time_array(t1, t2)
        x = (np.pi * self.freq(time) * self.Mtot_s) ** (2 / 3)
        A = 2 * self.nu * x / self.distance_M
        psi = 2 * (self.phi(time) - 3 * (x**1.5) * (1 - 0.5 * self.nu * x) * np.log(x / self.x_lso))

        ci = np.cos(self.inclination)
        ci2 = ci**2
        ci4 = ci**4
        hp = A * np.cos(psi) * (
            -(1 + ci2)
            + x * ((19 / 6) + (3 * ci2 / 2) - (ci4 / 3) + self.nu * (-(19 / 6) + (11 * ci2 / 6) + ci4))
            + (x**1.5) * (-2 * np.pi * (1 + ci2))
        )
        hc = A * np.sin(psi) * (
            -2 * ci
            + x * ci * ((17 / 3) - (4 * ci2 / 3) + self.nu * (-(13 / 3) + 4 * ci2))
            + (x**1.5) * (-4 * np.pi * ci)
        )

        if PyCBC_TimeSeries:
            from pycbc.types import TimeSeries

            return TimeSeries(hp, delta_t=time[1] - time[0], epoch=time[0]), TimeSeries(
                hc, delta_t=time[1] - time[0], epoch=time[0]
            )
        return {"time": time, "hp": hp, "hc": hc}

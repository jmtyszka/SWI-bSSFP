#!/usr/bin/env python3
"""
Function library for balanced SSFP contrast equation modeling

Usage
----
bssfp_sim.py

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2016-11-01 JMT From scratch

License
----
MIT License

Copyright (c) 2016 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '0.1.0'

import argparse
import numpy as np
from numpy.linalg import inv
from numpy import dot
import matplotlib.pyplot as plt

def main():

    # Setup pulse sequence parameters
    TR_ms, TE_ms = 5.0, 2.5
    alpha_deg, phi_deg = 1.0, 0.0

    # Setup material parameters
    T1_ms, T2_ms = 1400.0, 80.0

    # Isochromat off-resonance frequencies (Hz)
    df_max = 1000.0 / (2 * TR_ms)
    f_iso = np.linspace(0.0, df_max, 512)

    # Make space for results
    M = np.zeros([f_iso.size, 3])
    print(M.shape)

    for ff, f_Hz in enumerate(f_iso):

        # Call contrast equation
        M[ff, :] = bssfp(f_Hz, TR_ms, TE_ms, alpha_deg, phi_deg, T1_ms, T2_ms)

    # Derive magnitude and phase
    Mx, My, Mz = M[:,0], M[:,1], M[:,2]
    Mxy = Mx + 1.j * My
    Mmag = np.abs(Mxy)
    Mphi = np.unwrap(np.angle(Mxy))

    # Plot results
    ax = plt.subplot(3,1,1)
    ll = ['Mx','My','Mz']
    for n in range(0,3):
        plt.plot(f_iso, M[:,n], label=ll[n])
    plt.title('bSSFP magnetization')
    plt.legend()

    ax = plt.subplot(3,1,2)
    plt.plot(f_iso, Mmag)
    plt.title('bSSFP signal magnitude')

    ax = plt.subplot(3,1,3)
    plt.plot(f_iso, Mphi)
    plt.title('bSSFP signal phase')
    plt.xlabel('Off-resonance frequency (Hz)')

    plt.show()


def bssfp(f_Hz, TR_ms, TE_ms, alpha_deg, phi_deg, T1_ms, T2_ms):
    """

    :param f: float
        Isochromat frequency in Hz
    :param TR: float
        Repetition time in ms
    :param TE:
        Echo time in ms (typically = TR/2)
    :param alpha_deg:
        RF flip angle in degrees
    :param phi_deg:
        RF phase advance per TR in degrees
    :param T1:
        T1 relaxation time in ms
    :param T2:
        T2 relaxation time in ms
    :return: numpy array of floats
        3xN array of steady-state magnetization vectors
    """

    # Convert isochromat frequency from Hz to rad/s
    w_rad_s = f_Hz * 2.0 * np.pi

    # Convert timings to seconds
    TR_s, TE_s = TR_ms / 1000.0, TE_ms / 1000.0
    T1_s, T2_s = T1_ms / 1000.0, T2_ms / 1000.0

    # Convert angles from degrees to radians
    alpha_rad = np.deg2rad(alpha_deg)
    phi_rad = np.deg2rad(phi_deg)

    # Construct B1 rotation matrix
    B1 = Rx(alpha_rad)

    # Construct free precession matrix over whole TR
    theta_TR_rad = w_rad_s * TR_s + phi_rad
    Rz_TR = Rz(theta_TR_rad)

    # Construct free precession matrix over echo time
    theta_TE_rad = w_rad_s * TE_s
    Rz_TE = Rz(theta_TE_rad)

    # Construct relaxation matrices
    E_TR = ER(TR_s, T1_s, T2_s)
    E_TE = ER(TE_s, T1_s, T2_s)

    # Identity matrix
    I = np.eye(3)

    # Equilibrium magnetization
    M_0 = np.array([0,0,1]).transpose()

    # M+ = magnetization immediately after RF pulse
    m1 = dot(E_TR, Rz_TR)
    m2 = inv(I - dot(B1, m1))
    m3 = dot(I - E_TR, M_0)
    m4 = dot(B1, m3)
    M_plus = dot(m2, m4)

    # bSSFP equilibrium magnetization at echo time
    m5 = dot(Rz_TE, M_plus)
    m6 = dot(E_TE, m5)
    m7 = dot(I - E_TE, M_0)
    M = m6 + m7

    return M


def Rx(psi):
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])


def Rz(psi):
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([[cp, sp, 0], [-sp, cp, 0], [0, 0, 1]])


def ER(T, T1, T2):
    E1, E2 = np.exp(-T/T1), np.exp(-T/T2)
    return np.array([[E2, 0, 0],[0, E2, 0], [0, 0, E1]])


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
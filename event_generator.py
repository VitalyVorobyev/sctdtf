#! /usr/bin/env python3

""" Simple toy event generator for kine fitter test """

import numpy as np
import unittest

MASS_DICT = {
    'K0_S': 497.611,
    'pi+' : 139.57018,
    # 'pi0' : 134.9766,
    # 'D0 ' : 1865.84,
}

def make_hist(data, range=None, nbins=100, density=False):
    if range is None:
        range = (np.min(data), np.max(data))
    dhist, dbins = np.histogram(data, bins=nbins, density=density, range=range)
    dbins = 0.5 * (dbins[1:] + dbins[:-1])
    norm = np.sum(dhist) / data.shape[0]
    errors = np.array([-0.5 + np.sqrt(dhist / norm + 0.25),
                        0.5 + np.sqrt(dhist / norm + 0.25)]) * norm
    return (dbins, dhist, errors)

def gamma(beta):
    """ Lorentz factor """
    return 1. / np.sqrt(1. - beta**2)

def lorentz_boost(lv, bv):
    """ Relativistic transformation """
    beta = np.linalg.norm(bv)
    gam, n = gamma(beta), bv / beta
    t, r = lv[:,0].reshape(-1, 1), lv[:,1:]
    return np.column_stack([gam * t - beta * np.dot(r, n.T),
        r + (gam - 1) * np.dot(r, n.T) * n - gam * t * beta *  n])

def energy(mass, p3):
    """ Energy from mass and 3-momentum """
    return np.sqrt(mass**2 + np.sum(p3**2, axis=-1))

def ks2pipi(N, pk=None):
    """ Generator of the Ks0 -> pi+ pi- decays """
    mks, mpi = [MASS_DICT[key] for key in ['K0_S', 'pi+']]
    epi = 0.5*mks
    ppi = np.sqrt(epi**2 - mpi**2)
    costh = 2.*np.random.rand(N) - 1
    phi = 2.*np.random.rand(N)*np.pi
    sinth = np.sqrt(1. - costh**2)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    p3pip = ppi*np.array([sinth*cosphi, sinth*sinphi, costh]).T

    if pk is not None:
        p4pip, p4pim = [np.empty((p3pip.shape[0], 4)) for _ in range(2)]
        p4pip[:, 0], p4pim[:, 0] = epi, epi
        p4pip[:, 1:], p4pim[:, 1:] = p3pip, -p3pip
        bv = -(pk.reshape(-1, 1) / energy(mks, pk)).T
        p4pip, p4pim = [lorentz_boost(x, bv) for x in [p4pip, p4pim]]
        return (p4pip[:, 1:], p4pim[:, 1:])

    return (p3pip, -p3pip)

def mass_sq(p4):
    return p4[:,0]**2 - np.sum(p4[:,1:]**2, axis=-1)

def mass(p4):
    return np.sqrt(mass_sq(p4))

def p3top4(p3, mass):
    return np.column_stack([energy(mass, p3), p3])

def measurement_sampler(p3pip, p3pim, cov):
    """ Samples measurement error """
    assert cov.shape == (3, 3)
    N = p3pip.shape[0]
    dp = np.random.multivariate_normal([0,0,0], cov, 2*N)
    p3pip += dp[:N]
    p3pim += dp[N:]
    return (p3pip, p3pim)

def generate(N, cov, ptot=None):
    """ Generates N events for a given covariance matrix """
    p3pip, p3pim = ks2pipi(N, ptot)
    return measurement_sampler(p3pip, p3pim, cov)

class TestGenerator(unittest.TestCase):
    N = 10**4
    ptot = 10**3

    @staticmethod
    def check_mass(p3pip, p3pim):
        """ Compares pi+pi- invariant mass and Ks0 mass """
        mks, mpi = [MASS_DICT[key] for key in ['K0_S', 'pi+']]
        p4pip, p4pim = [p3top4(p3, mpi) for p3 in [p3pip, p3pim]]
        return np.allclose(mks**2 * np.ones(p3pip.shape[0]), mass_sq(p4pip + p4pim))

    @staticmethod
    def check_momentum(p3pip, p3pim, ptot):
        """ Checks pi+pi- total momentum """
        return np.allclose(ptot, p3pip + p3pim)

    def test_k0s_frame_mass(self):
        p3pip, p3pim = ks2pipi(TestGenerator.N)
        self.assertTrue(TestGenerator.check_mass(p3pip, p3pim))

    def test_k0s_frame_momentum(self):
        p3pip, p3pim = ks2pipi(TestGenerator.N)
        self.assertTrue(TestGenerator.check_momentum(p3pip, p3pim, np.zeros(3)))

    def test_lab_frame_mass(self):
        ptot = TestGenerator.ptot*np.random.rand(3)
        p3pip, p3pim = ks2pipi(TestGenerator.N, ptot)
        self.assertTrue(TestGenerator.check_mass(p3pip, p3pim))

    def test_lab_frame_momentum(self):
        ptot = TestGenerator.ptot*np.random.rand(3)
        p3pip, p3pim = ks2pipi(TestGenerator.N, ptot)
        self.assertTrue(TestGenerator.check_momentum(p3pip, p3pim, ptot))

def resolution_plot():
    import matplotlib.pyplot as plt
    mpi = MASS_DICT['pi+']
    cov = np.diag([3,3,5])
    N = 10**4
    p4pip, p4pim = [p3top4(p, mpi) for p in generate(N, cov)]
    x, bins, e = make_hist(mass(p4pip + p4pim))

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    plt.tight_layout()
    plt.savefig('mpipi.pdf')
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        unittest.main()
    # unittest.main()
    resolution_plot()

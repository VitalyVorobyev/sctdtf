#! /usr/bin/env python3

""" Complete chi2 fit for reference """

import numpy as np
from scipy.linalg import block_diag

from event_generator import UNIT, MASS_DICT, p3top4, mass_sq, mass, make_hist
import eintools as et

def gmass(p4pip, p4pim):
    """ Mass constraint """
    return MASS_DICT['K0_S']**2 - mass_sq(p4pip + p4pim)

def gmomentum(p4pip, p4pim, p4k):
    """ Momentum conservation constraint """
    return p4k - p4pip - p4pim

def chi2(p3pip0, p3pim0, p4pip, p4pim, covInv):
    """ Calculates chi2 """
    return et.chi2_item(p4pip[:, 1:] - p3pip0, covInv) +\
           et.chi2_item(p4pim[:, 1:] - p3pim0, covInv)

def gradient(p3pip0, p3pim0, p4pip, p4pim, p4k, covInv, lam):
    """ 15D Gradient """
    lamm, lame, lamp =\
        lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1), lam[:, 2:]
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    ddp1 = np.dot(p3pip - p3pip0, covInv) - lame*p3pip/epip - lamp
    ddp2 = np.dot(p3pim - p3pim0, covInv) - lame*p3pim/epim - lamp
    ddek = -2*lamm*ek  + lame
    ddpk = +2*lamm*p3k + lamp
    dlamm = gmass(p4pip, p4pim)
    dlamep = gmomentum(p4pip, p4pim, p4k)

    return 2*np.column_stack([ddp1, ddp2, ddek, ddpk, dlamm, dlamep])

def hessian(p4pip, p4pim, p4k, covInv, lam):
    """ 15x15 Hessian """
    N = p4pip.shape[0]
    lamm, lame = lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1)
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    p3overEpip = p3pip / epip
    p3overEpim = p3pim / epim

    dphess = lambda e, pOvE : np.einsum('ki, ij -> kij', lame / e, np.ones((3,3))) *\
        (np.einsum('ki, kj -> kij', pOvE, pOvE) - np.eye(3))

    hess = np.zeros((N, 15, 15))
    hess[:, 0:3,  0:3] = covInv + dphess(epip, p3overEpip)
    hess[:, 3:6,  3:6] = covInv + dphess(epim, p3overEpim)
    hess[:,6:10, 6:10] = 2*np.einsum('ki, ij -> kij', lamm, np.diag([-1, 1, 1 ,1]))
    hess[:,  10,    6] = hess[:,   6,   10] = -2*ek.ravel()  # dlamm dek
    hess[:,  10, 7:10] = hess[:,7:10,   10] =  2*p3k         # dlamm dpk
    hess[:,  11,  0:3] = hess[:, 0:3,   11] = -p3pip / epip  # dp1 dlam_eps
    hess[:,  11,  3:6] = hess[:, 3:6,   11] = -p3pim / epim  # dp2 dlam_eps
    hess[:, 11:, 6:10] = hess[:,6:10,  11:] =  np.eye(4)     # dek dlam_eps
    hess[:, 12:,  0:3] = hess[:, 0:3,  12:] = -np.eye(3)     # dlam_p dp1
    hess[:, 12:,  3:6] = hess[:, 3:6,  12:] = -np.eye(3)     # dlam_p dp2
    return 2*hess

def fit_to_ks(p3pip, p3pim, cov, nit=5):
    """ cov [3 x 3] """
    N = p3pip.shape[0]
    mpi = MASS_DICT['pi+']
    p3pip0, p3pim0 = p3pip.copy(), p3pim.copy()
    p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
    p4ks = p4pip + p4pim
    covInv = np.linalg.inv(cov)

    print('Inverse covariance\n{}'.format(covInv))
    print('N = {}'.format(N))

    lam = 1*np.ones((N, 5))

    logs = {key : [] for key in ['chi2', 'xi', 'grad', 'hess', 'det', 'cov']}
    def save_log(xi, p4pip, p4pim, grad, hess):
        logs['chi2'].append(chi2(p3pip0, p3pim0, p4pip, p4pim, covInv))
        logs['xi'].append(xi[:, :10].copy())
        logs['grad'].append(grad.copy())
        logs['hess'].append(hess.copy())
        logs['det'].append(np.linalg.det(hess))
        logs['cov'].append(2 * np.linalg.inv(hess))

    def calc(xi):
        p3pip, p3pim, p4ks, lam = xi[:, :3], xi[:, 3:6], xi[:, 6:10], xi[:, 10:]
        p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
        grad = gradient(p3pip0, p3pim0, p4pip, p4pim, p4ks, covInv, lam)
        hess = hessian(p4pip, p4pim, p4ks, covInv, lam)
        return (p4pip, p4pim, grad, hess)

    xi = np.column_stack([p3pip, p3pim, p4ks, lam])
    for iter in range(nit):
        print('Iteration {}'.format(iter))
        p4pip, p4pim, grad, hess = calc(xi)
        save_log(xi, p4pip, p4pim, grad, hess)
        xi -= np.einsum('kij, ki -> kj', np.linalg.inv(hess), grad)

    p4pip, p4pim, grad, hess = calc(xi)
    save_log(xi, p4pip, p4pim, grad, hess)

    return logs

def main():
    from event_generator import generate
    cov = np.diag([3,3,5])**2 * UNIT**2
    N = 10**4
    ptot = np.array([1000, 0, 0])
    (p3pip, p3pim), p3pipGen, p3pimGen = generate(N, cov, ptot=ptot)

    logs = fit_to_ks(p3pip, p3pim, cov, nit=10)
    np.savez('logs/fitres',
        chi2=logs['chi2'],
          xi=logs['xi'],
          Ck=logs['cov'],
        grad=logs['grad'],
        hess=logs['hess'],
         det=logs['det'],
      pipgen=p3pipGen,
      pimgen=p3pimGen,
         cov=cov
    )

if __name__ == '__main__':
    main()

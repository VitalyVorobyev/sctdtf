#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2

from event_generator import UNIT, MASS_DICT, p3top4, mass_sq, mass, make_hist, energy
import eintools as et

def plot_hist(xi, xi0, pimgen, pipgen):
    ks3 = pipgen + pimgen
    xi_gen = np.hstack((pipgen, pimgen, energy(MASS_DICT['K0_S'], ks3).reshape(-1, 1), ks3))

    def augment(xi):
        ks_m = np.sqrt(xi[:,6]**2 - np.sum(xi[:,7:10]**2, axis=-1)).reshape(-1, 1)
        ks_p = np.sqrt(np.sum(xi[:,7:10]**2, axis=-1)).reshape(-1, 1)
        pip_p = np.sqrt(np.sum(xi[:,:3]**2, axis=-1)).reshape(-1, 1)
        pim_p = np.sqrt(np.sum(xi[:,3:6]**2, axis=-1)).reshape(-1, 1)
        dpx = (xi[:, 0] + xi[:, 3] - xi[:, 7]).reshape(-1, 1)
        dpy = (xi[:, 1] + xi[:, 4] - xi[:, 8]).reshape(-1, 1)
        dpz = (xi[:, 2] + xi[:, 5] - xi[:, 9]).reshape(-1, 1)

        return np.hstack((xi, ks_m, ks_p, pip_p, pim_p, dpx, dpy, dpz))

    xi = augment(xi)
    xi0 = augment(xi0)
    xi_gen = augment(xi_gen)

    label = ['pip_px', 'pip_py', 'pip_pz', 'pim_px', 'pim_py', 'pim_pz', 
    'ks_E', 'ks_px', 'ks_py', 'ks_pz', 'ks_m', 'ks_p', 'pip_p', 'pim_p', 'dpx', 'dpy', 'dpz']

    for i in range(xi.shape[1]):
        fitted = xi[:, i] - xi_gen[:, i]
        unfitted = xi0[:, i] - xi_gen[:, i]

        for _ in range(5):
            fit_mean, fit_std = np.mean(fitted), np.std(fitted)
            fitted = fitted[np.abs(fitted - fit_mean) < 5.*fit_std]
            unfit_mean, unfit_std = np.mean(unfitted), np.std(unfitted)
            unfitted = unfitted[np.abs(unfitted - unfit_mean) < 5.*unfit_std]

        plt.figure(figsize=(8,6))
        if len(fitted) != 0:
            plt.errorbar(*make_hist(fitted, density=True), linestyle='none', marker='.', markersize=4, label='fit')
        if len(unfitted) != 0:
            plt.errorbar(*make_hist(unfitted, density=True), linestyle='none', marker='.', markersize=4, label='unfit')
        plt.plot([], [], ' ', label="fit std {:0.3f}".format(fit_std))
        plt.plot([], [], ' ', label="unfit std {:0.3f}".format(unfit_std))
        plt.plot([], [], ' ', label="fit mean {:0.3f}".format(fit_mean))
        plt.plot([], [], ' ', label="unfit mean {:0.3f}".format(unfit_mean))
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel(label[i], fontsize=16)
        plt.tight_layout()
        plt.savefig('fig/fit_{}.png'.format(label[i]))
    plt.show()

    

def plot_pool(xi, cov, pimgen, pipgen):
    ks3 = pipgen + pimgen
    xi0 = np.hstack((pipgen, pimgen, energy(MASS_DICT['K0_S'], ks3).reshape(-1, 1), ks3))

    label = ['pip_px', 'pip_py', 'pip_pz', 'pim_px', 'pim_py', 'pim_pz', 
    'ks_E', 'ks_px', 'ks_py', 'ks_pz']

    for i in range(xi.shape[1]):
        m = (xi[:, i] - xi0[:, i]) / cov[:, i, i] ** 0.5
        for _ in range(5):
            mean, std = np.mean(m), np.std(m)
            m = m[np.abs(m - mean) < 5.*std]
        print(m.shape, np.mean(m), np.std(m))
        if np.std(m) < 0.000001:
            return
        x, bins, e = make_hist(m)

        plt.figure(figsize=(6,5))
        plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
        plt.grid()
        plt.xlabel(label[i], fontsize=16)
        plt.tight_layout()
        plt.savefig('fig/pool_{}.png'.format(label[i]))
    plt.show()
 

def plot_conservation(xi):
    """ """
    pi4p = p3top4(xi[:, 0:3], MASS_DICT['pi+'])       
    pi4m = p3top4(xi[:, 3:6], MASS_DICT['pi+'])
    
    m = pi4m + pi4p - xi[:, 6:]

    plt.figure(figsize=(6,5))
    for i in range(4):
        x, bins, e = make_hist(m[:, i])
        plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4, label=i)
        plt.legend()

    plt.grid()
    plt.xlabel(r'$Conservation$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()


def plot_pipi_mass(xi):
    """ """
    pi4p = p3top4(xi[:, 0:3], MASS_DICT['pi+'])        
    pi4m = p3top4(xi[:, 3:6], MASS_DICT['pi+'])
    m = mass_sq(pi4p + pi4m)**0.5

    if np.std(m) < 0.000001:
        return
    x, bins, e = make_hist(m)#, range=[497, 498])

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()


def plot_ks0_mass(xi):
    """ """
    m = mass(xi[:, 6:10])
    m = m[~np.isnan(m)]
    for _ in range(5):
        mean, std = np.mean(m), np.std(m)
        m = m[np.abs(m - mean) < 5.*std]
    print(m.shape, np.mean(m), np.std(m))
    if np.std(m) < 0.000001:
        return
    x, bins, e = make_hist(m)#, range=[497, 498])

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.xlabel(r'$m(K_S^0)$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()

def plot_chi2(chisq):
    chisq = chisq[~np.isnan(chisq) & (chisq<100)]
    rng = [0, 50]
    nbins = 50
    # for _ in range(5):
    #     mean, std = np.mean(chisq), np.std(chisq)
    #     chisq = chisq[np.abs(chisq - mean) < 3.*std]
    print(chisq.shape, chisq.mean(), chisq.std())
    x, bins, e = make_hist(chisq, range=rng, nbins=nbins, density=False)

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    norm = chisq.shape[0]*(rng[1]-rng[0])/nbins if rng is not None else 1
    plt.plot(x, norm*chi2.pdf(x, 1))
    plt.grid()
    plt.xlabel(r'$\chi^2$', fontsize=16)
    plt.tight_layout()

def print_log(data, niter):
    def print_hessian(h):
        for row in h:
            for item in row:
                print('{:9.3f}'.format(item), end=' ')
            print('')

    def print_xi(x):
        print('  p3pip', end=' ')
        for item in x[:3]:
            print('{:9.3f}'.format(item), end=' ')
        print('\n  p3pim', end=' ')
        for item in x[3:6]:
            print('{:9.3f}'.format(item), end=' ')
        print('\n    p4k', end=' ')
        for item in x[6:10]:
            print('{:9.3f}'.format(item), end=' ')
        print('\nlambdas', end=' ')
        for item in x[10:]:
            print('{:9.3f}'.format(item), end=' ')
        print('')

    for iter in range(niter):
        print('\nHessian iter {}'.format(iter))
        print_hessian(data['hess'][iter][0])

        print('\nInverse Hessian iter {}'.format(iter))
        print_hessian(np.linalg.inv(data['hess'][iter][0]))

        print('\nGradient iter {}'.format(iter))
        print_xi(data['grad'][iter][0])

        print('\nXi iter {}'.format(iter))
        print_xi(data['xi'][iter][0])

def main():
    data = np.load('logs/pfitres.npz')
    # plot_ks0_mass(data['xi'][-1])
    # plot_conservation(data['xi'][-1])
    # plot_pipi_mass(data['xi'][-1])
    # plot_chi2(data['chi2'][-1])
    # plot_pool(data['xi'][-1], data['Ck'][-1], data['pimgen'], data['pipgen'])
    plot_hist(data['xi'][-1], data['xi'][0], data['pimgen'], data['pipgen'])
    # plt.show()

if __name__ == '__main__':
    main()
    # # print_log(data, 4)
    # for idx in range(4):
    #     xi = data['xi'][idx]
    #     xi = xi[~np.isnan(xi).any(axis=1)]
    #     print(xi.shape)
    #     plot_ks0_mass(xi)
    # plt.show()

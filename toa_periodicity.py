import os
import re
import pickle
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

import scipy.stats
from scipy.special import binom, erfcinv, gammaincc


def mkdir(dirname):
    if not os.path.exists(dirname):
        print(f'Creating directory {dirname}')
        os.makedirs(dirname)
        
    
def mkdir_containing(filename):
    d = os.path.dirname(filename)
    if (d != '') and not os.path.exists(d):
        mkdir(d)


def quad(f, xmin, xmax, epsabs=0.0, epsrel=1.0e-4):
    assert xmin < xmax
    return scipy.integrate.quad(f, xmin, xmax, epsabs=epsabs, epsrel=epsrel)[0]


def savefig(filename):
    if filename is None:
        plt.show()
    else:
        mkdir_containing(filename)
        print(f'Writing {filename}')
        plt.savefig(filename)
    
    plt.clf()


def write_pickle(filename, obj):
    print(f'Writing {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    print(f'Reading {filename}')
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pvalue_to_nsigmas(p):
    assert 0.0 < p < 1.0
    return erfcinv(p) * 2**0.5


####################################################################################################


def simulate_toas(chi, npulses, nevents=None, dbar=1.0):
    """Returns shape (n,p) array."""

    assert 0.0 <= chi <= 1.0
    assert (nevents is None) or (nevents >= 1)
    assert (npulses >= 3)
    assert (dbar > 0.0)

    n = nevents if (nevents is not None) else 1
    spacings = rand.uniform(chi*dbar, (2-chi)*dbar, size=(n,npulses-1))
    
    toas = np.zeros((n,npulses))
    toas[:,1:] = np.cumsum(spacings, axis=1)

    if nevents is None:
        toas = np.reshape(toas, (npulses,))
    
    return toas


####################################################################################################


class Analyzer:
    def __init__(self, npulses, maxgap):
        """
        The number of pulses 'npulses' is the 'p' parameter from the paper.
        The 'maxgap' parameter is the 'G' parameter from the paper.
        """
        
        assert npulses >= 3
        assert maxgap >= 0
        
        self.maxgap = maxgap
        self.npulses = npulses
        self.nmat = self.make_nmat(npulses, maxgap)
        self.ntrials = len(self.nmat)

        sum_n = np.sum(self.nmat, axis=1)
        sum_nn = np.sum(self.nmat * self.nmat, axis=1)
        self.coeff = npulses / (npulses * sum_nn - sum_n**2)

        assert self.nmat.shape == (self.ntrials, npulses)
        assert self.coeff.shape == (self.ntrials,)
        
        
    @classmethod
    def make_nmat(cls, N, G, maxcount=10000, dtype=np.float):
        """
        Helper function called by constructor.
        Returns shape-(M,N) array, where M = binom(G+N-1,G).
        """

        assert N >= 1
        assert G >= 0
    
        M = int(binom(N+G-1,G) + 0.5)
        assert(M <= maxcount)
    
        if N == 1:
            return np.zeros((1,1), dtype=dtype)

        src = cls.make_nmat(N-1, G, dtype=np.int)
        dst = np.zeros((M,N), dtype=dtype)
        idst = 0

        for g in range(G+1):
            jdst = int(binom(N+g-1,g) + 0.5)
            assert np.all(src[:(jdst-idst),:]) < N-1+g
            dst[idst:jdst,:-1] = src[:(jdst-idst),:]
            dst[idst:jdst,-1] = N-1+g
            idst = jdst

        return dst


    def _prep_toas(self, toas):
        if (toas.ndim == 2) and (toas.shape[1] == self.npulses):
            tmat = toas
        elif toas.shape == (self.npulses,):
            tmat = np.reshape(toas, (1,self.npulses))
        else:
            raise RuntimeError(f'Bad ToA array shape: {toas.shape}')

        if not np.all(tmat[:,:-1] < tmat[:,1:]):
            raise RuntimeError("ToA's must be sorted")
                
        # Subtract mean ToA from each event
        u = np.mean(tmat, axis=1)
        tmat -= np.reshape(u, (-1,1))

        assert (tmat.ndim == 2) and (tmat.shape[1] == self.npulses)
        return tmat

    
    def _analyze(self, toas):
        """
        Returns n_index, score.
        Called by analyze(), plot().
        """

        t = self._prep_toas(toas)
        nevents, npulses = t.shape
        ntrials = self.ntrials

        sum_nt = np.dot(self.nmat, t.T)
        assert sum_nt.shape == (ntrials, nevents)
        
        dsigma2 = sum_nt**2 * np.reshape(self.coeff, (ntrials,1))
        n_index = np.argmax(dsigma2, axis=0)
        
        dsigma2 = np.array([ dsigma2[n_index[i],i] for i in range(nevents) ])
        sigma2_u = np.sum(t**2, axis=1)
        sigma2_r = sigma2_u - dsigma2
        
        score = np.log(sigma2_u / sigma2_r)
        assert n_index.shape == score.shape == (nevents,)

        if toas.ndim == 1:
            n_index = n_index[0]
            score = score[0]
        
        return n_index, score


    def analyze(self, toas):
        n_index, score = self._analyze(toas)
        return score


    def plot_vertical(self, toas, filename=None):
        n_index, score = self._analyze(toas)
        nvec = self.nmat[n_index]

        tvec = (toas - toas[0]) * 1000.
        slope, intercept, _, _, _ = scipy.stats.linregress(nvec, tvec)
        tpvec = slope*nvec + intercept

        fig = plt.figure(figsize=(8,2))
        ax = fig.add_subplot(111)
        
        for (t,tper) in zip(tvec, tpvec):
            ax.axvline(t*1000., color='k')
            ax.axvline(tper*1000., color='r', ls='--')

            theta = np.abs(t-tper) / slope
            textloc = max(t,tper) + 0.1 * slope
            ax.text(textloc*1000., 0.5, f'{theta:0.03f}')

        xmin = np.min(np.minimum(tvec,tpvec)) - 0.5*slope
        xmax = np.max(np.maximum(tvec,tpvec)) + slope

        ax.axes.yaxis.set_visible(False)
        ax.set_xlim(xmin*1000., xmax*1000.)
        ax.set_ylim(0,1)
        savefig(filename)


    def plot_regression(self, toas, filename=None):
        n_index, score = self._analyze(toas)
        nvec = self.nmat[n_index]

        tvec = (toas - toas[0]) * 1000.
        slope, intercept, _, _, _ = scipy.stats.linregress(nvec, tvec)

        fig = plt.figure(figsize=(4.4,3.3))
        ax = fig.add_axes([0.15, 0.14, 0.80, 0.85]) # left, bottom, width, height
        plt.scatter(nvec, tvec)
        
        xline = np.array([ np.min(nvec)-0.5, np.max(nvec)+0.5 ])
        yline = slope*xline + intercept
        plt.plot(xline, yline, ls='--')

        plt.xlabel(r'$n_j$')
        plt.ylabel(r'$t_j$')
        savefig(filename)
        

    def analyze_slow(self, toas):
        """For debugging."""
        
        tmat = self._prep_toas(toas)
        scores = np.zeros(len(tmat))

        for i,t in enumerate(tmat):
            sigma2_u = np.dot(t,t)

            for nvec in self.nmat:
                slope, intercept, _, _, _ = scipy.stats.linregress(nvec, t)
                residuals = t - (slope*nvec + intercept)
                sigma2_r = np.dot(residuals, residuals)
                scores[i] = max(scores[i], np.log(sigma2_u/sigma2_r))

        return scores[0] if (toas.ndim == 1) else scores


####################################################################################################


class TailDist:
    def __init__(self, a, q, x0):
        assert a > 0.0
        assert q > 0.0
        assert x0 > 0.0

        self.a = a
        self.q = q
        self.x0 = x0
        self.coeff = 1.0
        self.coeff = 1.0 / quad(self.pdf, x0, np.Inf)
        

    def pdf(self, x):
        """Vectorized."""
        a, q, x0 = self.a, self.q, self.x0
        return self.coeff * np.exp(-a * (x**q - x0**q))


    def icdf(self, x):
        """Vectorized."""
        a, q, x0 = self.a, self.q, self.x0
        return gammaincc(1./q, a*x**q) / gammaincc(1./q,a*x0**q)

    
    def icdf_slow(self, x):
        return quad(self.pdf, x, np.Inf)
            

class TailFitter:
    def __init__(self, sims, ntail=10**4):
        nsims = len(sims)
        assert 100 <= ntail
        assert nsims >= 100*ntail
        
        x = np.sort(sims)
        x = x[-ntail:]
        
        initial_a = 1.0 / (np.mean(x) - x[0])
        ics = np.array([ np.log(initial_a), 0.0 ])
        
        def minus_logL(v):
            loga, logq = v
            d = TailDist(np.exp(loga), np.exp(logq), x[0])
            pdf = d.pdf(x)
            return -np.sum(np.log(pdf))

        ret = scipy.optimize.minimize(minus_logL, ics, method='Nelder-Mead')
        assert ret.success == True

        self.dist = TailDist(np.exp(ret.x[0]), np.exp(ret.x[1]), x[0])
        self.sorted_tail_sims = x
        self.ntail = ntail
        self.ntot = nsims

        
    def pvalue(self, x):
        assert x > self.sorted_tail_sims[0]
        return self.dist.icdf(x) * self.ntail / float(self.ntot)
    
        
    def show_histogram(self, filename=None):
        s = self.sorted_tail_sims
        xvec = np.linspace(s[0], s[-1], 100)
        yvec = self.dist.pdf(xvec)
        
        plt.hist(s, bins=40, density=True)
        plt.plot(xvec, yvec)
        savefig(filename)


    def _ks_data(self):
        return self.dist.icdf(self.sorted_tail_sims)

    def show_ks_plot(self, filename=None):
        plt.plot(np.arange(self.ntail), self._ks_data())
        savefig(filename)

    def ks_pvalue(self):
        r = scipy.stats.kstest(self._ks_data(), lambda x:x)
        return r.pvalue


####################################################################################################


def sim_filename(chi, npulses, maxgap, nevents):
    return f'sims_n{nevents}/sim_p{npulses}_g{maxgap}_chi{chi}.npy'


def generate_sims(chi, npulses, maxgap, nevents=10**6, filename=None, blocksize=10**4, skip_if_exists=False):
    if filename is None:
        filename = sim_filename(chi, npulses, maxgap, nevents)

    mkdir_containing(filename)
    print(f'generate_sims: {npulses=}, {maxgap=}, {nevents=}, {chi=}, {filename=}')

    if skip_if_exists and os.path.exists(filename):
        print('    already exists, skipping')
        return
    
    a = Analyzer(npulses, maxgap)
    scores = np.zeros(nevents)

    n = 0
    while n < nevents:
        m = min(blocksize, nevents-n)
        toas = simulate_toas(chi, npulses, m)
        scores[n:n+m] = a.analyze(toas)
        n += m

    np.save(filename, scores)


def plot_histogram(sims, data, filename=None):
    plt.hist(sims, bins=25)
    plt.axvline(data, color='r', ls='--')
    plt.xlabel(r'Periodicity statistic $\hat S$')
    plt.ylabel(r'Counts')
    savefig(filename)

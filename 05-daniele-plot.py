#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import toa_periodicity as tp


def make_daniele_histogram(filename_stem, maxgap, smin, smax):
    data_dir = 'data_v2'
    plot_dir = 'plots_v2'
    nevents = 10**8
    chi = 0.2

    data_filename = f'{data_dir}/{filename_stem}.txt'
    print(f'Reading {data_filename}')

    data_toas = np.loadtxt(data_filename)
    npulses = len(data_toas)
    
    a = tp.Analyzer(npulses, maxgap)
    data_score = a.analyze(data_toas)
    print(f'{data_score=}')

    sim_filename = tp.sim_filename(chi, npulses, maxgap, nevents)
    sims = np.load(sim_filename)
    assert sims.shape == (nevents,)

    plt.xlabel('Score $\hat S$')
    plt.ylabel('Counts')
    plt.hist(sims, bins=np.linspace(0.0,smax,100), log=True, color='grey')
    plt.axvline(data_score, color='black', ls=':')
    
    tp.savefig(f'{plot_dir}/histogram_{filename_stem}.pdf')

        
make_daniele_histogram('event1_p9', maxgap=5, smin=2.0, smax=14.0)
make_daniele_histogram('event2', maxgap=1, smin=2.0, smax=14.0)
make_daniele_histogram('event3', maxgap=2, smin=2.0, smax=14.0)

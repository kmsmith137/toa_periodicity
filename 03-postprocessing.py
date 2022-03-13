#!/usr/bin/env python

import os
import numpy as np
import toa_periodicity as tp

# Note: keep in sync with 04-plots.py
do_tail_fits = True
data_dir = 'data_v2'
plot_dir = 'plots_v2'
analysis_dir = 'analysis_v2'
nevents = 10**8
chi_fid = 0.2
chi_all = [ 0.0, 0.1, 0.2, 0.3, 0.4 ]
assert chi_fid in chi_all

# (filename_stem, observed_gap)
# todo = [ ('event1',3), ('event2',0), ('event3',1) ]    # v1

# v2 follows
# Note: keep in sync with 04-plots.py
todo = [ ('event1_p7',6),
         ('event1_p8',5),
         ('event1_p9',4),
         ('event1_p10',3),
         ('event1_p11',2),
         ('event1_p12',1),
         ('event1_p13',1),
         ('event1_p14',2),
         ('event2',0),
         ('event3',1) ]


tp.mkdir(analysis_dir)


####################################################################################################


# (filename_stem, maxgap, chi) -> (data_score, npulses, ntrials, bf_pvalue, ana_pvalue, final_pvalue)
results = { }


for filename_stem, observed_gap in todo:
    for maxgap in [ observed_gap, observed_gap+1 ]:
        data_filename = f'{data_dir}/{filename_stem}.txt'
        print(f'Reading {data_filename}')
    
        data_toas = np.loadtxt(data_filename)
        npulses = len(data_toas)

        a = tp.Analyzer(npulses, maxgap)
        data_score = a.analyze(data_toas)
        print(f'{data_score=}')

        for chi in chi_all:
            sim_filename = tp.sim_filename(chi, npulses, maxgap, nevents)
            sims = np.load(sim_filename)
            assert sims.shape == (nevents,)

            nabove = np.sum(sims > data_score)
            bf_pvalue = nabove / float(nevents)   # can be zero
            ana_pvalue = None
            final_pvalue = bf_pvalue

            if do_tail_fits and (nabove <= 100) and (nevents >= 10**6):
                print(f'Computing analytic p-value: {filename_stem}, maxgap={maxgap}, chi={chi}')                
                tf = tp.TailFitter(sims, ntail = nevents // 10**3)
                ana_pvalue = tf.pvalue(data_score)
                final_pvalue = ana_pvalue
                print(f'BF pvalue = {bf_pvalue}, ana pvalue = {ana_pvalue}')
                print(f'KS test pvalue = {tf.ks_pvalue()}')
                
                if chi == chi_fid:
                    histogram_filename = f'{plot_dir}/{filename_stem}_maxgap{maxgap}_tail_histogram.pdf'
                    tf.show_histogram(filename = histogram_filename)
            
            k = (filename_stem, maxgap, chi)
            v = (data_score, npulses, a.ntrials, bf_pvalue, ana_pvalue, final_pvalue)
            results[k] = v
            
            if chi == chi_fid:
                histogram_filename = f'{plot_dir}/{filename_stem}_maxgap{maxgap}_histogram.pdf'
                tp.plot_histogram(sims, data_score, histogram_filename)


####################################################################################################


pvalue_filename = f'{analysis_dir}/pvalues_n{nevents}.pkl'
tp.write_pickle(pvalue_filename, results)


#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import toa_periodicity as tp

# Note: keep in sync with 03-postprocessing.py
plot_dir = 'plots_v2'
analysis_dir = 'analysis_v2'
nevents = 10**8
chi_fid = 0.2
chi_all = [ 0.0, 0.1, 0.2, 0.3, 0.4 ]

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

assert chi_fid in chi_all
tp.mkdir(analysis_dir)

# (filename_stem, maxgap, chi) -> (npulses, ntrials, bf_pvalue, ana_pvalue, final_pvalue)i
pvalue_filename = f'{analysis_dir}/pvalues_n{nevents}.pkl'
results = tp.read_pickle(pvalue_filename)



####################################################################################################


summary_filename = os.path.join(analysis_dir, 'summary.txt')
print(f'Writing {summary_filename}')

with open(summary_filename,'w') as fout:
    print('% This quasi-TeX will need a little hand-editing', file=fout)
    print(r'Event & $N_{\rm pulses}$ & $G$ & $N_{\rm trials}$ & BF $p$-value & Ana $p$-value & Gaussian sigmas \\', file=fout)
    
    for filename_stem, observed_gap in todo:
        for maxgap in [observed_gap, observed_gap+1]:
            k = filename_stem, maxgap, chi_fid
            data_score, npulses, ntrials, bf_pvalue, ana_pvalue, pvalue = results[k]
            nsigma_str = '--'

            if pvalue > 0.0:
                nsigmas = tp.pvalue_to_nsigmas(pvalue)
                nsigma_str = f'{nsigmas:.1f}'
            
            print(f'{filename_stem} & {npulses} & {maxgap} & {ntrials} & {bf_pvalue} & {ana_pvalue} & {nsigma_str} \\\\', file=fout)
        
print(f'Wrote {summary_filename}')


####################################################################################################


for filename_stem, observed_gap in todo:
    for maxgap in [ observed_gap, observed_gap+1 ]:
        pvalues = [ ]
        for chi in chi_all:
            k = filename_stem, maxgap, chi
            data_score, npulses, ntrials, bf_pvalue, ana_pvalue, pvalue = results[k]
            pvalues.append(pvalue)

        pchi_filename = os.path.join(analysis_dir, f'{filename_stem}_maxgap{maxgap}_pchi.txt')
        pchi_data = np.transpose([chi_all, pvalues])
        
        print(f'Writing {pchi_filename}')        
        np.savetxt(pchi_filename, pchi_data)


####################################################################################################


nsp_list = [ ]
pvalue_list = [ ]

for filename_stem, observed_gap in todo:
    if not filename_stem.startswith('event1_p'):
        continue

    nsp = int(filename_stem[8:])
    maxgap = observed_gap + 1
    
    k = filename_stem, maxgap, chi_fid
    data_score, npulses, ntrials, bf_pvalue, ana_pvalue, pvalue = results[k]

    nsp_list.append(nsp)
    pvalue_list.append(pvalue)

plt.semilogy(nsp_list, pvalue_list)
plt.xlabel(f'Number of subpulses')
plt.ylabel(f'$p$-value')

filename = f'{plot_dir}/pvalue_vs_nsp.pdf'
tp.savefig(filename)


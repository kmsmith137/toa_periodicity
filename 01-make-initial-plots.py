#!/usr/bin/env python

import numpy as np
import toa_periodicity as tp

data_dir = 'data_v2'
plot_dir = 'plots_v2'

# (filename_stem, ngap)
# todo = [ ('event1',3), ('event2',0), ('event3',1) ]    # v1

# v2 follows
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


####################################################################################################


for filename_stem, ngap in todo:
    toa_filename = f'{data_dir}/{filename_stem}.txt'
    vertical_filename = f'{plot_dir}/{filename_stem}_vertical.pdf'
    regression_filename = f'{plot_dir}/{filename_stem}_regression.pdf'

    toas = np.loadtxt(toa_filename)
    assert toas.ndim == 1

    npulses = len(toas)
    a = tp.Analyzer(npulses, ngap)
    a.plot_vertical(toas, filename=vertical_filename)
    a.plot_regression(toas, filename=regression_filename)
    

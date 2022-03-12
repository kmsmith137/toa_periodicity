#!/usr/bin/env python

import toa_periodicity as tp

nevents = 10**8
chi_list = [ 0.0, 0.1, 0.2, 0.3, 0.4 ]
skip_if_exists = True

# (npulses, maxgap)
todo = [ (14, 2),
         (14, 3),
         (13, 1),
         (13, 2),
         (12, 1),
         (12, 2),
         (12, 3),
         (12, 4),
         (11, 2),
         (11, 3),
         (10, 3),
         (10, 4),
         (9, 4),
         (9, 5),
         (8, 5),
         (8, 6),
         (7, 6),
         (7, 7),
         (6, 0),
         (6, 1),
         (5, 1),
         (5, 2) ]

for npulses, maxgap in todo:
    for chi in chi_list:
        tp.generate_sims(chi=chi, npulses=npulses, maxgap=maxgap, nevents=nevents, skip_if_exists=skip_if_exists)

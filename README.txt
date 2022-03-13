Periodicity analysis for Michilli et al, "Sub-second periodicity in a fast
radio burst", https://arxiv.org/abs/2107.08463

Note: these scripts take 12+ hours to run and generate ~80 GB of data!

The scripts perform auxiliary analyses which are not in the paper, and they
also use 10^8 Monte Carlo simulations per analysis. (It would be easy to modify
the scripts to run faster by doing fewer MCs and auxiliary analysis; let me
know if that would be useful.)


----------------------------------------------------------------------------------------------------


./01-make-initial-plots.py

   Makes some quick exploratory plots from the CHIME ToAs
   ("vertical" and "regression" plots).

   The plots which are relevant for the paper are:

      plots_v2/event1_p9_vertical.pdf
      plots_v2/event1_p9_regression.pdf  (*)
      
      plots_v2/event2_vertical.pdf
      plots_v2/event2_regression.pdf
      
      plots_v2/event3_vertical.pdf
      plots_v2/event3_regression.pdf

   (*) Appears in paper as "Extended Data Figure 3"


----------------------------------------------------------------------------------------------------


./02-make-sims.py

   A long-running script (12+ hours) which generates a ton (~80 GB) of simulated ToAs.
   (The CHIME ToA's are not used.)

   The output files have filenames such as:
       sims_n100000000/sim_p9_g5_chi0.2.npy

   (In code, you can call toa_periodicity.sim_filename())


----------------------------------------------------------------------------------------------------


./03-postprocessing.py

    Evaluates data S-hat statistics, compares with sims, computes p-values.

    Results written here:
       analysis_v2/pvalues_n100000000.pkl

    Writes some S-hat histograms, with filenames such as
       plots_v2/event1_p9_maxgap5_histogram.pdf


----------------------------------------------------------------------------------------------------


./04-plots.py
    
    A summary table of results is written here:
       plots_v2/summary.txt  (*)

    (*) Selected results from this table appear in "Extended Data Table 2" in the paper.


----------------------------------------------------------------------------------------------------


./05-daniele-plot.py


    Creates the following plots:

        plots_v2/histogram_event1_p9.pdf
        plots_v2/histogram_event2.pdf
        plots_v2/histogram_event3.pdf

    These plots are shown in the right column of Figure 2 in the paper.

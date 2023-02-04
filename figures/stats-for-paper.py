import cdatasets
import figurelib

#################### ICC of movement ####################

hcptrt = cdatasets.HCP1200KindaLikeTRT()

print("HCP1200-like-TRT ICC:",
      figurelib.icc_full(hcptrt.get_subject_info()['subject'], hcptrt.get_movement()))

trt = cdatasets.TRT()

print("TRT ICC:",
      figurelib.icc_full(trt.get_subject_info()['subject'], trt.get_movement()))

import scipy
import numpy as np

hcp = cdatasets.HCP1200()
spears = [scipy.stats.spearmanr(hcp.get_nodal_cmstats()['var'][i], hcp.get_nodal_metrics()['degree'][i]).correlation for i in range(0, hcp.N_subjects())]
print("Median correlation in HCP1200 between nodal vbc and degree:", np.median(spears))
parts = [figurelib.pcorr(hcp.get_nodal_cmstats()['var'][i], hcp.get_nodal_metrics()['degree'][i], hcp.get_parcelareas())[0] for i in range(0, hcp.N_subjects())]
print("Median partial correlation in HCP1200 between nodal vbc and degree:", np.median(spears))

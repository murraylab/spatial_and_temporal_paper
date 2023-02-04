import datasets
from cand import Canvas, Vector, Point
import models
import seaborn as sns
import pandas
import numpy as np
import scipy
import util
import os.path
import itertools
from figurelib import lin, names_for_stuff

import cdatasets


# model = ...

from models_for_figure2 import all_cmodels


# This can be useful when changing models: [m.cache_all() for mods in cmodels for m in mods[1]]

metrics = ["assort", "cluster", "lefficiency", "gefficiency", "modularity", "transitivity"]
param_metrics = ["lmbda", "floor", "meanar1"]
weighted_metrics = ['meancor', 'varcor', 'kurtcor']
weighted_param_getters = dict(lmbda=lambda m : m.get_lmbda(),
                              floor=lambda m : m.get_floor(),
                              meanar1=lambda m : np.mean(m.get_ar1s(), axis=1),
                              meancor=lambda m : m.get_cmstats()['meancor'],
                              varcor=lambda m : m.get_cmstats()['varcor'],
                              kurtcor=lambda m : m.get_cmstats()['kurtcor'])

all_metrics = param_metrics + weighted_metrics + metrics

def format_corr(vals):
    if None in vals:
        return ""
    val = np.mean(vals)
    saturation = max(int(val*100), 0)
    rounded = round(val, 2)
    colorstr = f"{rounded}\\cellcolor{{red!{saturation}}}"
    return colorstr

def format_lin(vals):
    if None in vals:
        return ""
    val = np.mean(vals)
    saturation = max(int(val*100), 0)
    rounded = round(val, 2)
    colorstr = f"{rounded}\\cellcolor{{red!{saturation}}}"
    return colorstr

def format_r2(vals):
    if None in vals:
        return ""
    val = np.mean(vals)
    saturation = int(np.sqrt(max(0,val))*100)
    rounded = round(val, 2)
    if rounded < -10000:
        rounded = r"<-10\textsuperscript{5}"
    elif rounded < -1000:
        rounded = r"<-10\textsuperscript{4}"
    colorstr = f"{rounded}\\cellcolor{{red!{saturation}}}"
    return colorstr


print(f"\\begin{{tabular}}{{ll{'|rrr'*len(metrics)}}}")
print(r'\toprule')
print(" & & " + " & ".join([f"\\multicolumn{{3}}{{l}}{{{names_for_stuff[met]}}}" for met in metrics]) + "\\\\")
print(" Dataset & Model & " + " & ".join([k for m in metrics for k in ["Lin", "$r_s$", "$R^2$"]]) + "\\\\")
for dsname,(dataobjs,models_for_dataset) in all_cmodels.items():
    for modelname,modelobjs in models_for_dataset:
        if modelname == models_for_dataset[0][0]:
            print("\\midrule", dsname, end="")
        else:
            print("", end="")
        print(" &", names_for_stuff[modelname], end='')
        for met in metrics:
            lins = []
            corrs = []
            r2s = []
            for mod,dat in zip(modelobjs,dataobjs):
                vals1 = dat.get_metrics()[met]
                vals2 = mod.get_metrics()[met]
                spear = scipy.stats.spearmanr(vals1, vals2)
                R2 = 1 - np.sum((np.asarray(vals1)-np.asarray(vals2))**2)/np.sum((np.asarray(vals1)-np.mean(np.asarray(vals1)))**2)
                corrs.append(spear.correlation)
                lins.append(lin(vals1, vals2))
                r2s.append(R2)
            print(" & ", format_lin(lins), " & ", format_corr(corrs), "&", format_r2(r2s), end="")
        print("\\\\")

print(r'\bottomrule')


print(r'\toprule')
print((" & & " + " & ".join([f"\\multicolumn{{3}}{{l}}{{{names_for_stuff[met]}}}" for met in param_metrics+weighted_metrics]) + "\\\\").replace("λ", r"$\lambda$").replace("∞", r"$\infty$"))
print(" Dataset & Model & " + " & ".join([k for m in metrics for k in ["Lin", "$r_s$", "$R^2$"]]) + "\\\\")
for dsname,(dataobjs,models_for_dataset) in all_cmodels.items():
    for modelname,modelobjs in models_for_dataset:
        if modelname == "degreerand":
            continue
        if modelname == models_for_dataset[0][0]:
            print("\\midrule", dsname, end="")
        else:
            print("", end="")
        print(" &", names_for_stuff[modelname], end='')
        for met in param_metrics+weighted_metrics:
            lins = []
            corrs = []
            r2s = []
            for mod,dat in zip(modelobjs,dataobjs):
                try:
                    vals1 = weighted_param_getters[met](dat)
                    vals2 = weighted_param_getters[met](mod)
                    spear = scipy.stats.spearmanr(vals1, vals2)
                    R2 = 1 - np.sum((np.asarray(vals1)-np.asarray(vals2))**2)/np.sum((np.asarray(vals1)-np.mean(np.asarray(vals1)))**2)
                except:
                    corrs.append(None)
                    lins.append(None)
                    r2s.append(None)
                else:
                    corrs.append(spear.correlation)
                    lins.append(lin(vals1, vals2))
                    r2s.append(R2)
            print(" & ", format_lin(lins), " & ", format_corr(corrs), "&", format_r2(r2s), end="")
        print("\\\\")

print(r'\bottomrule')


print(r'\end{tabular}')

import figurelib
import cdatasets
import util
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels
from figurelib import names_for_stuff

from cand import Canvas, Vector, Point

hcp = cdatasets.HCP1200()
hcptrt = cdatasets.HCP1200KindaLikeTRT()
trt = cdatasets.TRT()
cc = cdatasets.CamCanFiltered()
acfss = {name : np.asarray([[statsmodels.tsa.stattools.acf(tss[i]) for i in range(0, ds.N_regions())] for tss in ds.get_timeseries()]) for name,ds in [("hcp", hcp), ("trt", trt), ("cc", cc)]}

c = Canvas(7.2, 7.8, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)
#c.add_grid(["iccs-by-region", "icc-diff", "mean-iccs", "by-region", "by-subject", "all"], 2, Point(.5, .5, "in"), Point(7, 3.5, "in"), size=Vector(1.2, 1.2, "in"))

# c.add_axis("iccs-by-region", Point(.5, .85, "in"), Point(2, 2.35, "in"))
# c.add_axis("mean-iccs", Point(2.5, .85, "in"), Point(4, 2.35, "in"))
for DS,offset in [("hcp", Vector(0,4.7, "in")), ("trt", Vector(0, 1.7, "in"))]:
    c.add_grid([f"iccs-by-region_{DS}", f"icc-diff_{DS}", f"mean-iccs_{DS}", None], 1, Point(0.5, 1.9, "in")+offset, Point(6.9, 2.7, "in")+offset, size=Vector(.8, .8, "in"))
    c.add_text(names_for_stuff[DS], Point(.1, 3.0, "in")+offset, weight="bold", size=7, ha="left")
# c.add_grid(["by-region", "by-subject", "all", "corr"], 2, Point(4.6, .5, "in"), Point(6.9, 2.7, "in"), size=Vector(.78, .78, "in"))

for DS,offset in [("hcp", Vector(0,4.7, "in")), ("trt", Vector(0, 1.7, "in")), ("cc", Vector(0, 0, "in"))]:
    c.add_grid([f"corr_{DS}", f"by-region_{DS}", f"by-subject_{DS}", f"all_{DS}"], 1, Point(.5, .6, "in")+offset, Point(6.9, 1.4, "in")+offset, size=Vector(.8, .8, "in"))

c.add_text(names_for_stuff["camcan"], Point(.1, 1.7, "in"), weight="bold", size=7, ha="left")

for dsname,ds in [("hcp", hcptrt), ("trt", trt)]:
    DS = dsname
    fn = f"_iccs_{dsname}_region.pkl"
    if util.plock(fn):
        tss = ds.get_timeseries()
        arks = {}
        # Lage 11 is d, not an actual lag.
        for lag in range(1, 11):
            arks[lag] = []
            for subject in range(0, ds.N_subjects()):
                arks[lag].append([np.corrcoef(tss[subject][i,0:-lag], tss[subject][i,lag:])[0,1] for i in range(0, ds.N_regions())])
        iccs = {lag : [figurelib.icc_full(list(ds.get_subject_info()['subject']), np.asarray(arks[lag])[:,region])[0] for region in range(0, ds.N_regions())] for lag in range(1, 11)}
        iccs.update({12 : [figurelib.icc_full(list(ds.get_subject_info()['subject']), np.asarray(ds.get_long_memory())[:,region])[0] for region in range(0, ds.N_regions())]})
        util.psave(fn, iccs)
    else:
        iccs = util.pload(fn)
    
    fn = f"_iccs_{dsname}_mean.pkl"
    if util.plock(fn):
        mean_iccs = {lag : figurelib.icc_full(list(hcpklt.get_subject_info()['subject']), np.mean(arks[lag], axis=1)) for lag in range(1, 11)}
        util.psave(fn, mean_iccs)
    else:
        mean_iccs = util.pload(fn)
    
    ax = c.ax(f"iccs-by-region_{DS}")
    vp = ax.violinplot(np.asarray([v for k,v in iccs.items()]).T, positions=list(range(1, 11))+[12], showextrema=False, showmedians=True)
    vp['cmedians'].set_edgecolor('k')
    for body in vp['bodies']:
        body.set_facecolor((.5, .5, .5))
        body.set_edgecolor((.5, .5, .5))
        body.set_alpha(1)
    vp['bodies'][-1].set_facecolor((.8, .8, .8))
    vp['bodies'][-1].set_edgecolor((.8, .8, .8))
    
    ax.set_xlabel("Lag")
    ax.set_ylabel("ICC")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "d"])
    ax.set_title("Reliability of lag-k autocorrelation")
    sns.despine(ax=ax)
    
    ax = c.ax(f"icc-diff_{DS}")
    vp = ax.violinplot(np.asarray([v-np.asarray(iccs[1]) for k,v in iccs.items()]).T, positions=list(range(1, 11))+[12], showextrema=False, showmedians=True)
    ax.axhline(0, c='k')
    vp['cmedians'].set_edgecolor('k')
    for body in vp['bodies']:
        body.set_facecolor((.5, .5, .5))
        body.set_edgecolor((.5, .5, .5))
        body.set_alpha(1)
    vp['bodies'][-1].set_facecolor((.8, .8, .8))
    vp['bodies'][-1].set_edgecolor((.8, .8, .8))
    
    ax.set_xlabel("Lag")
    ax.set_ylabel("ΔICC")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "d"])
    ax.set_title("Reliability difference from lag-1 to lag-k")
    sns.despine(ax=ax)
    
    ax = c.ax(f"mean-iccs_{DS}")
    ax.errorbar(x=list(range(1,11)), y=[mean_iccs[i][0] for i in range(1, 11)], yerr=np.asarray([np.abs(np.asarray(mean_iccs[i][1])-mean_iccs[i][0]) for i in range(1, 11)]).T, c='k', marker='o', markersize=3)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ICC")
    ax.set_title("Reliability of mean lag-k autocorrelation")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "d"])
    sns.despine(ax=ax)
    lmreli = figurelib.icc_full(list(ds.get_subject_info()['subject']), np.mean(ds.get_long_memory(), axis=1))
    ax.errorbar(x=[12], y=[lmreli[0]], yerr=[[lmreli[0]-lmreli[1][0]], [lmreli[1][1]-lmreli[0]]], marker='o', c=(.5, .5, .5), markersize=3)


# # Regressing out v
# acfss2 = {name : np.asarray([[statsmodels.tsa.stattools.acf(tss[i]) for i in range(0, ds.N_regions())] for tss in ds.get_timeseries()]) for name,ds in [("hcpklt", hcpklt)]}
# DS = "hcpklt"
# dsname = DS
# ds = hcpklt
# lm = ds.get_long_memory()
# ar1s = ds.get_ar1s()
# acfssall2 = np.vstack(acfss2[dsname])
# partialiccs = {}
# for lag in range(2, 11):
#     m = sm.OLS(acfssall2[:,lag], sm.add_constant(acfssall2[:,1])).fit()
#     partialiccs[lag] = [icc_full(ds.get_subject_info()['subject'], m.resid.reshape(ds.N_subjects(), -1)[:,i]) for i in range(0, 360)]
#     #partialiccsb = [icc_full(ds.get_subject_info()['subject'], m.resid.reshape(-1, ds.N_subjects())[i,:]) for i in range(0, 360)]
#

# stackedmem = np.hstack(ds.get_long_memory())
# m = sm.OLS(stackedmem, sm.add_constant(acfssall2[:,1])).fit()
# partialiccs[11] = [icc_full(ds.get_subject_info()['subject'], m.resid.reshape(ds.N_subjects(), -1)[:,i]) for i in range(0, 360)]

# vp = plt.violinplot(np.asarray([[vi[0] for vi in v] for k,v in partialiccs.items()]).T, showextrema=False, showmedians=True)
# plt.show()


for dsname,ds in [("hcp", hcp), ("trt", trt), ("cc", cc)]:
    DS = dsname
    lm = ds.get_long_memory()
    ar1s = ds.get_ar1s()
    acfssall = np.vstack(acfss[dsname])
    # Split into 10 subsets for CV
    inds = np.random.permutation(acfssall.shape[0]).reshape(10 if DS != "trt" else 12,-1)
    ho_r2s = {}
    # For each lag
    for lag in range(1, 11):
        m = sm.OLS(acfssall[inds[0],lag], sm.add_constant(acfssall[inds[0],1])).fit()
        ho_r2s[lag] = []
        for rset in range(1, inds.shape[0]):
            ho_r2 = 1-np.sum((m.predict(sm.add_constant(acfssall[inds[rset],1])) - acfssall[inds[rset],lag])**2)/np.sum((acfssall[inds[rset],lag] - np.mean(acfssall[inds[rset],lag]))**2)
            ho_r2s[lag].append(ho_r2)
    # Now compare to d
    stackedmem = np.hstack(ds.get_long_memory())
    m = sm.OLS(stackedmem[inds[0]], sm.add_constant(acfssall[inds[0],1])).fit()
    ho_r2s_lm = []
    for rset in range(1, inds.shape[0]):
        ho_r2 = 1-np.sum((m.predict(sm.add_constant(acfssall[inds[rset],1])) - stackedmem[inds[rset]])**2)/np.sum((stackedmem[inds[rset]] - np.mean(stackedmem[inds[rset]]))**2)
        ho_r2s_lm.append(ho_r2)
    
    ax = c.ax(f"corr_{DS}")
    vals = np.asarray(list(ho_r2s.values()))
    ax.errorbar(list(range(1, 11)), np.median(vals, axis=1), yerr=[np.max(vals, axis=1)-np.median(vals, axis=1), np.median(vals, axis=1)-np.min(vals, axis=1)], c='k')
    ax.errorbar([12], [np.median(ho_r2s_lm)], yerr=[[np.max(ho_r2s_lm)-np.median(ho_r2s_lm)], [np.median(ho_r2s_lm)-np.min(ho_r2s_lm)]], c=(.5, .5, .5), marker='o', markersize=3)
    ax.set_ylim(0, 1)
    ax.set_title("Predictive power of lag-1 for lag-k")
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_xlabel("Lag")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "d"])
    sns.despine(ax=ax)
    
    ax = c.ax(f"by-region_{DS}")
    figurelib.corplot(np.mean(ar1s, axis=0), np.mean(lm, axis=0), "Mean regional TA-$\Delta_1$", "Mean regional d", ax=ax, showr2=False)
    ax.set_title("Mean regional autocorrelation")
    
    ax = c.ax(f"by-subject_{DS}")
    ax.cla()
    figurelib.corplot(np.mean(ar1s, axis=1), np.mean(lm, axis=1), "Mean subject TA-$\Delta_1$", "Mean subject d", ax=ax, showr2=False, alpha=(.05 if DS=="hcp" else .2))
    ax.set_title("Mean subject autocorrelation")
    
    ax = c.ax(f"all_{DS}")
    ax.cla()
    figurelib.corplot(np.asarray(ar1s).flatten(), np.asarray(lm).flatten(), "Suject-region TA-$\Delta_1$", "Subject-region d", ax=ax, showr2=False, markersize=.1, color=(0, 0, 0, .02), rasterized=True)
    ax.set_title("Subject-region autocorrelation")
    #ax.set_rasterized(True)



# bplot = plt.boxplot(np.asarray(list(ho_r2s.values())).T, showfliers=False)
# for patch in bplot['medians']:
#     patch.set_color('k')

# lagcors = [[scipy.stats.spearmanr(acfss[subj,:,1], acfss[subj,:,lag]).correlation for lag in range(1, 11)] for subj in range(0, hcp.N_subjects())]
# ax = plt.gca()
# ax.violinplot(np.asarray(lagcors), showextrema=False, showmedians=True)
# ax.axhline(0, c='k')
# sns.despine(ax=ax)

# acfssall = np.vstack(acfss)
# lagcors = [[scipy.stats.spearmanr(acfss[:,region,1], acfss[:,region,lag]).correlation for lag in range(1, 11)] for region in range(0, hcp.N_regions())]
# ax = plt.gca()
# ax.violinplot(np.asarray(lagcors), showextrema=False, showmedians=True)
# ax.axhline(0, c='k')
# sns.despine(ax=ax)

c.add_figure_labels([
    ("a", "iccs-by-region_hcp", Vector(-.61, 0, "cm")),
    ("b", "icc-diff_hcp", Vector(-.61, 0, "cm")),
    ("c", "mean-iccs_hcp", Vector(-.61, 0, "cm")),
    ("d", "corr_hcp", Vector(-.61, 0, "cm")),
    ("e", "by-region_hcp", Vector(-.61, 0, "cm")),
    ("f", "by-subject_hcp", Vector(-.61, 0, "cm")),
    ("g", "all_hcp", Vector(-.61, 0, "cm")),
    ("h", "iccs-by-region_trt", Vector(-.61, 0, "cm")),
    ("i", "icc-diff_trt", Vector(-.61, 0, "cm")),
    ("j", "mean-iccs_trt", Vector(-.61, 0, "cm")),
    ("k", "corr_trt", Vector(-.61, 0, "cm")),
    ("l", "by-region_trt", Vector(-.61, 0, "cm")),
    ("m", "by-subject_trt", Vector(-.61, 0, "cm")),
    ("n", "all_trt", Vector(-.61, 0, "cm")),
    ("o", "corr_cc", Vector(-.61, 0, "cm")),
    ("p", "by-region_cc", Vector(-.61, 0, "cm")),
    ("q", "by-subject_cc", Vector(-.61, 0, "cm")),
    ("r", "all_cc", Vector(-.61, 0, "cm"))], size=8)


c.save("figure1-supp-ta.pdf")

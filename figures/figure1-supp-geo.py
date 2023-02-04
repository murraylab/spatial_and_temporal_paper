import datasets
from cand import Canvas, Vector, Point
import models
import seaborn as sns
import os.path
import pandas
import numpy as np
import scipy
import wbplot
import util
from figurelib import corplot, icc_full, simplescatter, fingerprint, names_for_stuff, short_names_for_stuff, ARROWSTYLE, trt_bootstrap_icc

# Set this to 0 to make the real figure, or 1 or 2 to make the
# supplements

FILENAME = "figure1.pdf"

import cdatasets
data = cdatasets.HCP1200()
data_rep = cdatasets.HCP1200(3)
datatrt = cdatasets.HCP1200KindaLikeTRT()
#datatrt = cdatasets.TRT()

titlestyle = {"weight": "bold", "size": 7}
metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity']
graph_metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity', 'meancor', 'varcor', 'kurtcor']
all_metrics = ["lmbda", "floor", "meanar1", 'assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity', "meancor", "varcor", "kurtcor"]
fplabels = ["ar1", "cm", "chance", "gbc", "vbc", "kbc", "degree", "centrality"]
nodal_metrics = ["mean", "var", "kurt", "degree", "centrality"]
nodal_metrics_ar1 = ["ar1", "mean", "var", "kurt", "degree", "centrality"]


#################### Set up the canvas ####################

#c = Canvas(9.724, 6.4, "in", fontsize=6, fontsize_ticks=5) # min size = 5, max size = 7, figure letter label size = 8 pt
c = Canvas(7.2, 5.2, "in") # min size = 5, max size = 7, figure letter label size = 8 pt
c.set_font("Nimbus Sans", ticksize=5, size=6)
# c.debug_grid(Vector(1, 1, "in"))
# c.debug_grid(Vector(.5, .5, "in"), linewidth=.5)

ROW1 = 4.1
ROW1b = 3.8
BOXSIZE = .13
# Graph theory diagram
c.add_unit("diagram", Vector(.55, .55, "in"), Point(.3, ROW1, "in"))
c.add_box(Point(0, 0, "diagram"), Point(6.25, 1.5, "diagram"), boxstyle="Round", zorder=-10, fill=True, facecolor='#f8f8f8', linewidth=.5)

c.add_axis("minigrid", Point(4.4, ROW1b, "in"), Point(4.4+8*BOXSIZE, ROW1b+8*BOXSIZE, "in"))

c.add_axis("ar1diagram", Point(6.0, ROW1b-.2, "in")-Vector(0, .4, "cm"), Point(6.9, ROW1b+.2, "in")-Vector(0, .4, "cm"))
c.add_axis("sadiagram", Point(6.0, ROW1b+.6, "in")-Vector(0, .4, "cm"), Point(6.9, ROW1b+.6+.5, "in"))

ROW2 = 2.0

itemheight = .125
c.add_axis("reliabilities", Point(0.7, ROW2, "in"), Point(1.25, ROW2+itemheight*len(all_metrics), "in"))
c.add_grid(["lmbda", "floor", "meanar1"], 3, Point(1, .05, "axis_reliabilities")+Vector(.30,0,"in"), Point(1, .95, "axis_reliabilities")+Vector(.65,0,"in"),size=Vector(.35, .35, "in"))

c.add_axis("grid", Point(2.6, ROW2, "in")+Vector(0, .4, "cm"), Point(2.6+BOXSIZE*5, ROW2+.15+BOXSIZE*9, "in"))

brain_img_loc = Point(3.5, ROW2, "in")+Vector(0, .4, "cm")
brain_img_height = Vector(0, 0.90, "in")

itemheight_reli = .16
c.add_axis("region_reliability", Point(5.7, ROW2, "in"), Point(6.9, ROW2+itemheight_reli*len(nodal_metrics_ar1), "in"))

ROW3 = .4

brain_reliability_img_loc = Point(.2, ROW3, "in") + Vector(0, .4, "cm")

itemheight = .13
c.add_axis("fingerprint", Point(2.2, ROW3, "in"), Point(2.9, ROW3+itemheight*len(fplabels), "in"))

c.add_axis("ar1_corr_boxplots", Point(3.6, ROW3, "in"), Point(4.7, ROW3+itemheight_reli*len(nodal_metrics), "in"))
c.add_grid(["nodal_var_ar1_mean", "nodal_degree_ar1_mean"], 2, Point(5.2, ROW3+.1, "in"), Point(5.6, ROW3+1.1, "in"), size=Vector(.4, .4, "in"))

c.add_axis("reliability_vs_ar1", Point(6.2, ROW3, "in"), Point(6.9, ROW3+.7, "in"))
# c.add_axis("reliability_hist", Point(0, 1, "axis_reliability_vs_ar1"), Point(1, 1, "axis_reliability_vs_ar1")+Vector(0, .2, "in"))



#################### Minigrid ####################

metrics_vals = data.get_metrics()
metrics_vals.update(data.get_cmstats())

grid = np.zeros((len(graph_metrics), len(graph_metrics)))*np.nan
gridp = np.zeros((len(graph_metrics), len(graph_metrics)))*np.nan
labels = grid.tolist()
for i in range(0, len(graph_metrics)):
    for j in range(i+1, len(graph_metrics)):
        spear = scipy.stats.spearmanr(metrics_vals[graph_metrics[j]], metrics_vals[graph_metrics[i]])
        if spear.pvalue < .05:
            grid[i,j] = np.abs(spear.correlation)
        else:
            grid[i,j] = np.abs(spear.correlation)
        gridp[i,j] = spear.pvalue
        #labels[i][j] = f"{spear.correlation:.2f}"
        labels[i][j] = "**" if spear.pvalue < .01/36 else "*" if spear.pvalue < .05/36 else ""

n_sig = np.nansum(gridp<.05)
n_total = np.nansum(gridp>=-1)

# fn = f"_cache_f1_minigrid_n_sig_random_{datatrt.name}.pkl"
# if not os.path.exists(fn):
#     nullgridp = np.zeros((10000, len(graph_metrics), len(graph_metrics)))*np.nan
#     for k in range(0, 10000):
#         perm = np.random.permutation(len(metrics_vals[graph_metrics[0]]))
#         for i in range(0, len(graph_metrics)):
#             for j in range(i+1, len(graph_metrics)):
#                 spear = scipy.stats.spearmanr(metrics_vals[graph_metrics[j]], np.asarray(metrics_vals[graph_metrics[i]])[perm])
#                 nullgridp[k,i,j] = spear.pvalue
#     n_sig_random = np.nansum(nullgridp<.05, axis=(1,2))
#     util.psave(fn, n_sig_random)
# else:
#     n_sig_random = util.pload(fn)


ax = c.ax("minigrid")
VSCALE = (0, 1)
ax.imshow(grid.T[1:,:-1], vmin=VSCALE[0], vmax=VSCALE[1], cmap="Blues", aspect='auto')
ax.axis('off')
c.add_colorbar("minigrid_colorbar", Point(0, 0, "axis_minigrid")-Vector(0, .4, "cm"), Point(1, 0, "axis_minigrid")-Vector(0, .2, "cm"), cmap="Blues", bounds=VSCALE)
c.add_text("Absolute Spearman correlation", Point(.5, 0, "axis_minigrid")-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")
c.add_text("Correlation among graph metrics", Point(.5, 0, "axis_minigrid")+Vector(0, 1.20, "in"), ha="center", **titlestyle)

#sig_sig = "**" if np.mean(n_sig_random>n_sig)<.01 else "*" if np.mean(n_sig_random>n_sig)<.05 else ""
#sig_sig = ""
#c.add_text(f"{n_sig}/{n_total}\nsignificant{sig_sig}", Point(.8, .9, "axis_minigrid"), style="italic")

for i in range(0, len(graph_metrics)-1):
    c.add_text(short_names_for_stuff[graph_metrics[i]], Point(i, -.7+i, "minigrid"), rotation=0, horizontalalignment="left", verticalalignment="bottom", size=5)

for i in range(0, len(graph_metrics)-1):
    c.add_text(short_names_for_stuff[graph_metrics[i+1]], Point(-.7, i, "minigrid"), horizontalalignment="right", verticalalignment="center", size=5)

for i in range(0, len(graph_metrics)-1):
    for j in range(i, len(graph_metrics)-1):
        c.add_text(labels[i][j+1], Point(i, j+.3, "minigrid"), horizontalalignment="center", verticalalignment="center", size=10)


#################### Make the grid of correlations ####################


ts_metrics_vals = {"meanar1": np.mean(data.get_ar1s(), axis=1),
                   "loglmbda": np.log(data.get_lmbda()),
                   "lmbda": data.get_lmbda(),
                   "floor": data.get_floor(),
                   "meancor": data.get_cmstats()['meancor'],
                   "varcor": data.get_cmstats()['varcor'],
                   "kurtcor": data.get_cmstats()['kurtcor'],
}
ts_metrics = ["lmbda", "floor", "spatialmetrics", "meanar1", "allmetrics"] # List them because dicts are unordered

graph_metric_values = data.get_metrics()
graph_metric_values.update(data.get_cmstats())

grid = np.zeros((len(ts_metrics), len(graph_metrics)))*np.nan
labels = grid.tolist()
for i in [0, 1, 3]:
    for j in range(0, len(graph_metrics)):
        spear = scipy.stats.spearmanr(graph_metric_values[graph_metrics[j]], ts_metrics_vals[ts_metrics[i]])
        if spear.pvalue < .05:
            grid[i,j] = np.abs(spear.correlation)
        else:
            grid[i,j] = np.abs(spear.correlation)
        #labels[i][j] = f"{spear.correlation:.2f}"
        labels[i][j] = "**" if spear.pvalue < .01/27 else "*" if spear.pvalue < .05/27 else ""

import statsmodels.formula.api as smf
graph_metric_values.update(ts_metrics_vals)
df = pandas.DataFrame(graph_metric_values)
df_train = df.sample(frac=.5, random_state=0)
df_test = df.drop(df_train.index)
for j in range(0, len(graph_metrics)):
    predicted = smf.ols(f"{graph_metrics[j]} ~ lmbda + floor", data=df_train).fit().predict(df_test)
    spear = scipy.stats.spearmanr(predicted, df_test[graph_metrics[j]])
    print(graph_metrics[j], spear)
    grid[2,j] = spear.correlation
    labels[2][j] = "**" if spear.pvalue < .01/18 else "*" if spear.pvalue < .05/18 else ""
    predicted = smf.ols(f"{graph_metrics[j]} ~ meanar1 + lmbda + floor", data=df_train).fit().predict(df_test)
    spear = scipy.stats.spearmanr(predicted, df_test[graph_metrics[j]])
    print(graph_metrics[j], spear)
    grid[4,j] = spear.correlation
    labels[4][j] = "**" if spear.pvalue < .01/18 else "*" if spear.pvalue < .05/18 else ""

ax = c.ax("grid")
VSCALE = (0, np.ceil(np.max(grid)*10)/10)
ax.imshow(grid.T, vmin=VSCALE[0], vmax=VSCALE[1], cmap="Blues", aspect='auto')
ax.axis('off')
c.add_colorbar("grid_colorbar", Point(0, 0, "axis_grid")-Vector(0, .4, "cm"), Point(1, 0, "axis_grid")-Vector(0, .2, "cm"), cmap="Blues", bounds=VSCALE)
c.add_text("Absolute Spearman correlation", Point(.5, 0, "axis_grid")-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")

for i in range(0, len(ts_metrics)):
    c.add_text(short_names_for_stuff[ts_metrics[i]], Point(i, -.7, "grid"), horizontalalignment="left", verticalalignment="bottom", size=5, rotation=30)

for i in range(0, len(graph_metrics)):
    c.add_text(short_names_for_stuff[graph_metrics[i]], Point(-.7, i, "grid"), horizontalalignment="right", verticalalignment="center", size=5)

for i in range(0, len(ts_metrics)):
    for j in range(0, len(graph_metrics)):
        c.add_text(labels[i][j], Point(i, j+.3, "grid"), horizontalalignment="center", verticalalignment="center", size=10)


c.add_text("Correlation with graph metrics", Point(.5, 0, "axis_grid")+Vector(0, 1.5, "in"), ha="center", **titlestyle)


#################### Test retest ####################

ax = c.ax("reliabilities")


metrics_vals = {"meanar1": np.mean(datatrt.get_ar1s(), axis=1),
                   "loglmbda": np.log(datatrt.get_lmbda()),
                   "lmbda": datatrt.get_lmbda(),
                   "floor": datatrt.get_floor(),
}
metrics_vals.update(datatrt.get_cmstats(True))
metrics_vals.update(datatrt.get_metrics(True))

fn = f"_cache_brain_metric_reliability_{datatrt.name}.pkl"
if not os.path.exists(fn):
    reliabilities = {k : icc_full(datatrt.get_subject_info()['subject'], metrics_vals[k]) for k in all_metrics}
    util.psave(fn, reliabilities)
else:
    reliabilities = util.pload(fn)

fn = f"_cache_brain_metric_reliability_{datatrt.name}_simulated.pkl"
if not os.path.exists(fn):
    reliabilities_simulated = {}
    for k in all_metrics:
        reliabilities_simulated[k] = trt_bootstrap_icc(datatrt, np.asarray(metrics_vals[k]), 1000)
        util.psave(fn, reliabilities_simulated, overwrite=True)
        print("Saved", k)
else:
    reliabilities_simulated = util.pload(fn)

ax.barh(range(0, len(all_metrics)), [reliabilities[k][0] for k in all_metrics], xerr=np.asarray([np.abs(np.asarray(reliabilities[k][1])-reliabilities[k][0]) for k in all_metrics]).T, color=['r' if k in ['meanar1', 'lmbda', 'floor'] else (.5, .5, .5) if 'cor' in k else 'k' for k in all_metrics], clip_on=False, height=.8, error_kw={"clip_on": False})
# ax.errorbar([reliabilities[k][0] for k in all_metrics], range(0, len(all_metrics)), xerr=np.asarray([np.abs(np.asarray(reliabilities[k][1])-reliabilities[k][0]) for k in all_metrics]).T, clip_on=False, color='k', linestyle='none')

ax.set_xlim(0, .8)
sns.despine(ax=ax)
ax.set_yticks([])
ax.set_ylim(-.5, len(all_metrics) - .5)
ax.invert_yaxis()

reliability_ar1_sim = np.asarray([r[0] for r in reliabilities_simulated["meanar1"]])
reliability_lmbda_sim = np.asarray([r[0] for r in reliabilities_simulated["lmbda"]])
reliability_floor_sim = np.asarray([r[0] for r in reliabilities_simulated["floor"]])
for i in range(0, len(all_metrics)):
    reliability_i_sim = [r[0] for r in reliabilities_simulated[all_metrics[i]]]
    lt_ar1_sim = np.mean(reliability_ar1_sim<=reliability_i_sim)
    lt_lmbda_sim = np.mean(reliability_lmbda_sim<=reliability_i_sim)
    lt_floor_sim = np.mean(reliability_floor_sim<=reliability_i_sim)
    rs = ""
    rs += "#" if lt_lmbda_sim < .01 else "+" if lt_lmbda_sim < .05 else "_"
    rs += "#" if lt_floor_sim < .01 else "+" if lt_floor_sim < .05 else "_"
    rs += "#" if lt_ar1_sim < .01 else "+" if lt_ar1_sim < .05 else "_"
    c.add_text(rs, Point(reliabilities[all_metrics[i]][0], i+.4, "reliabilities")+Vector(.35, .1, "cm"), horizontalalignment="center", verticalalignment="center", size=5, font="Noto Mono", weight="regular")


    # if reliabilities[all_metrics[i]][2] <.01:
    #     c.add_text("**", Point(reliabilities[all_metrics[i]][0], i+.4, "reliabilities")+Vector(.25, 0, "cm"), horizontalalignment="center", verticalalignment="center", size=10)
    # elif reliabilities[all_metrics[i]][2] <.05:
    #     c.add_text("**", Point(reliabilities[all_metrics[i]][0], i+.4, "reliabilities")+Vector(.25, 0, "cm"), horizontalalignment="center", verticalalignment="center", size=10)


for i in range(0, len(all_metrics)):
    c.add_text(names_for_stuff[all_metrics[i]], Point(0, i, "reliabilities")+Vector(-.1, 0, "cm"), horizontalalignment="right", verticalalignment="center", size=5)

c.add_text("Test-retest reliability", Point(0, 0, "axis_reliabilities")+Vector(0,1.5, "in")+Vector(0, .4, "cm"), ha="left", **titlestyle)
ax.set_xlabel("Reliability (ICC)")

arrowx = max([reliabilities[v][0] for v in ["lmbda", "floor", "meanar1"]])
ax = c.ax("lmbda")
ax.cla()
simplescatter(data.get_lmbda(), data_rep.get_lmbda(), s=.4, c='k', alpha=.25, rasterized=True, ax=ax, linewidth=0)
ax.set_xticks([])
ax.set_yticks([])
c.add_text("Session 2", Point(-.15, .5, "axis_lmbda"), rotation=90, size=5)
ind = all_metrics.index("lmbda")
arrowfrm = Point(arrowx, ind, "reliabilities")
arrowto = Point(-.25, .5, "axis_lmbda")
c.add_arrow(arrowfrm, arrowto, lw=1, arrowstyle=ARROWSTYLE)

ax = c.ax("floor")
ax.cla()
simplescatter(data.get_floor(), data_rep.get_floor(), s=.4, c='k', alpha=.25, rasterized=True, ax=ax, linewidth=0)
ax.set_xticks([])
ax.set_yticks([])
c.add_text("Session 2", Point(-.15, .5, "axis_floor"), rotation=90, size=5)
ind = all_metrics.index("floor")
arrowfrm = Point(arrowx, ind, "reliabilities")
arrowto = Point(-.05, 1, "axis_floor")
c.add_arrow(arrowfrm, arrowto, lw=1, arrowstyle=ARROWSTYLE)


ax = c.ax("meanar1")
ax.cla()
simplescatter(np.mean(data.get_ar1s(), axis=1), np.mean(data_rep.get_ar1s(), axis=1), s=.4, c='k', alpha=.25, rasterized=True, ax=ax, linewidth=0)
ax.set_xticks([])
ax.set_yticks([])
c.add_text("Session 1", Point(.5, -.2, "axis_meanar1"), size=5)
c.add_text("Session 2", Point(-.15, .5, "axis_meanar1"), rotation=90, size=5)
ind = all_metrics.index("meanar1")
arrowfrm = Point(arrowx, ind, "reliabilities")
arrowto = Point(-.05, 1, "axis_meanar1")
c.add_arrow(arrowfrm, arrowto, lw=1, arrowstyle=ARROWSTYLE)


#################### AR1 vs other stuff ####################

# ax = c.ax("ar1lmbda")
# corplot(np.mean(data.get_ar1s(), axis=1), np.log(data.get_lmbda()), "", "log($\\lambda$)", ax=ax, showr2="r2")

# ax = c.ax("ar1gbc")
# corplot(np.mean(data.get_ar1s(), axis=1), data.get_floor(), "AR1", "GC", ax=ax, showr2="r2")


#################### Fingerprinting ####################

fn = f"_f1_cache_fingerprint_{datatrt.name}_{data.name}_{data_rep.name}.pkl"
if os.path.exists(fn):
    (fingerprintcache, fingerprintcache_pair, same_subject, diff_subject, diff_regions) = util.pload(fn)
else:
    inds = np.triu_indices(datatrt.N_regions(), 1)
    mats_reduced = np.asarray([m[inds] for m in datatrt.get_matrices()])
    subjects = np.asarray(datatrt.get_subject_info()['subject'])
    run = datatrt.get_subject_info()['run']
    scan = datatrt.get_subject_info()['scan']
    chance = (len(subjects)/len(set(subjects))-1)/(len(subjects)-1)
    # fingerprintcache = {
    #     "ar1": [fingerprint(subjects[run==(i+1)], datatrt.get_ar1s()[run==(i+1)]) for i in range(0, len(set(run)))],
    #     "cm": [fingerprint(subjects[run==(i+1)], mats_reduced[run==(i+1)]) for i in range(0, len(set(run)))],
    #     "gbc": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['mean'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "vbc": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['var'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "kbc": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['kurt'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "meants": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['ts_mean'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "varts": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['ts_var'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "kurtts": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['ts_kurt'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "lefficiency": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['lefficiency'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "cluster": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['cluster'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "centrality": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['centrality'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "degree": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['degree'][run==(i+1)]) for i in range(0, len(set(run)))],
    #     "chance": [chance for i in range(0, len(set(run)))],
    #     }
    fingerprintcache_pair = {
        "ar1": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_ar1s()[(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "cm": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], mats_reduced[(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "gbc": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_cmstats()['mean'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "vbc": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_cmstats()['var'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "kbc": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_cmstats()['kurt'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "meants": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_cmstats()['ts_mean'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "varts": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_cmstats()['ts_var'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "kurtts": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_cmstats()['ts_kurt'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "lefficiency": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_metrics()['lefficiency'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "cluster": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_metrics()['cluster'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "centrality": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_metrics()['centrality'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "degree": [fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_nodal_metrics()['degree'][(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)],
        "chance": [chance for i in range(0, len(set(scan))) for j in range(0, i)],
        }
    fingerprintcache = {
        "ar1": fingerprint(subjects, datatrt.get_ar1s()),
        "cm": fingerprint(subjects, mats_reduced),
        "gbc": fingerprint(subjects, datatrt.get_nodal_cmstats()['mean']),
        "vbc": fingerprint(subjects, datatrt.get_nodal_cmstats()['var']),
        "kbc": fingerprint(subjects, datatrt.get_nodal_cmstats()['kurt']),
        "meants": fingerprint(subjects, datatrt.get_nodal_cmstats()['ts_mean']),
        "varts": fingerprint(subjects, datatrt.get_nodal_cmstats()['ts_var']),
        "kurtts": fingerprint(subjects, datatrt.get_nodal_cmstats()['ts_kurt']),
        "lefficiency": fingerprint(subjects, datatrt.get_nodal_metrics()['lefficiency']),
        "cluster": fingerprint(subjects, datatrt.get_nodal_metrics()['cluster']),
        "centrality": fingerprint(subjects, datatrt.get_nodal_metrics()['centrality']),
        "degree": fingerprint(subjects, datatrt.get_nodal_metrics()['degree']),
        "chance": chance,
        }
    tocor = {
        "ar1": data.get_ar1s(),
        "gbc": data.get_nodal_cmstats()['mean'],
        "vbc": data.get_nodal_cmstats()['var'],
        "kbc": data.get_nodal_cmstats()['kurt'],
        "meants": data.get_nodal_cmstats()['ts_mean'],
        "varts": data.get_nodal_cmstats()['ts_var'],
        "kurtts": data.get_nodal_cmstats()['ts_kurt'],
        "lefficiency": data.get_nodal_metrics()['lefficiency'],
        "cluster": data.get_nodal_metrics()['cluster'],
        "centrality": data.get_nodal_metrics()['centrality'],
        "degree": data.get_nodal_metrics()['degree'],
        "cm": np.asarray([cm[np.triu_indices(data.N_regions())] for cm in data.get_matrices()]),
    }
    tocor_rep = {
        "ar1": data_rep.get_ar1s(),
        "gbc": data_rep.get_nodal_cmstats()['mean'],
        "vbc": data_rep.get_nodal_cmstats()['var'],
        "kbc": data_rep.get_nodal_cmstats()['kurt'],
        "meants": data_rep.get_nodal_cmstats()['ts_mean'],
        "varts": data_rep.get_nodal_cmstats()['ts_var'],
        "kurtts": data_rep.get_nodal_cmstats()['ts_kurt'],
        "lefficiency": data_rep.get_nodal_metrics()['lefficiency'],
        "cluster": data_rep.get_nodal_metrics()['cluster'],
        "centrality": data_rep.get_nodal_metrics()['centrality'],
        "degree": data_rep.get_nodal_metrics()['degree'],
        "cm": np.asarray([cm[np.triu_indices(data_rep.N_regions())] for cm in data_rep.get_matrices()]),
    }
    same_subject = {k : [scipy.stats.spearmanr(tocor[k][i], tocor_rep[k][i]).correlation for i in range(0, data.N_subjects())] for k in tocor.keys()}
    rng = np.random.RandomState(0)
    diff_subject = {k : [scipy.stats.spearmanr(tocor[k][rn], tocor_rep[k][(rn + rng.randint(1, data.N_subjects()-1)) % data.N_subjects()]).correlation for rn in rng.randint(data.N_subjects(), size=1000)] for k in tocor.keys()}
    diff_regions = {k : [scipy.stats.spearmanr(tocor[k][rng.randint(data.N_subjects())][rng.permutation(len(tocor[k][0]))], tocor_rep[k][rng.randint(data.N_subjects())][rng.permutation(len(tocor[k][0]))]).correlation for _ in range(0, 1000)] for k in tocor.keys()}
    util.psave(fn, (fingerprintcache,fingerprintcache_pair,same_subject,diff_subject, diff_regions))


ax = c.ax("fingerprint")
ax.cla()
barheights = [np.mean(fingerprintcache_pair[k]) if k != "" else 0 for k in fplabels]
ax.barh(range(0, len(fplabels)), barheights, color=['r' if k in ['ar1'] else (.5, .5, .5) if 'bc' in k else 'k' for k in fplabels], clip_on=False)
points = np.asarray([(k, fpv) for k in range(0, len(fplabels)) for fpv in fingerprintcache_pair[fplabels[k]]])
ax.scatter(points[:,1], points[:,0]+np.random.uniform(-.3, .3, len(points[:,0])), s=1, c='#555555', zorder=10)
#ax.errorbar(barheights, range(0, len(fplabels)), xerr=[scipy.stats.sem(fingerprintcache_pair[k]) if k != "" else 0 for k in fplabels], clip_on=False, c='k', linestyle='none')
ax.set_xlim(0, 1)
ax.invert_yaxis()
sns.despine(ax=ax)
ax.set_yticks([])
ax.set_ylim(-.5, len(fplabels)-.5)
ax.invert_yaxis()
ax.set_xlabel("Subject identification rate")
c.add_text("Fingerprinting", Point(.2, 0, "axis_fingerprint")+Vector(0,1.15, "in"), ha="center", **titlestyle)


for i in reversed(range(0, len(fplabels))):
    if fplabels[i] == "": continue
    c.add_text(names_for_stuff[fplabels[i]], Point(0, i, "fingerprint")+Vector(-.1, 0, "cm"), horizontalalignment="right", verticalalignment="center", size=5)


# ax = c.ax("ar1dist")
# ax.cla()
# bins = np.linspace(-.2, 1, 25)
# ax.hist(same_subject['ar1'], alpha=.5, density=True, bins=bins)
# ax.hist(diff_subject['ar1'], alpha=.5, density=True, bins=bins)
# ax.hist(diff_regions['ar1'], alpha=.5, density=True, bins=bins)
# sns.despine(ax=ax, left=True)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlim(-.2, 1)
# ax.axvline(0, c='k', lw=1)
# ax.axvline(1, c='k', lw=1, clip_on=False)

# ax.set_xticks([0, .5, 1])
# ax.set_xlabel("Spearman correlation")

# c.add_legend(Point(0, 2, "axis_ar1dist"), [("Test-retest", {"linewidth": 4, "c": sns.color_palette()[0]}),
#                                                ("Different subjects", {"linewidth": 4, "c": sns.color_palette()[1]}),
#                                                ("Different regions", {"linewidth": 4, "c": sns.color_palette()[2]})],
#              sym_width=Vector(.8, 0, "Msize"), line_spacing=Vector(0, 1.3, "Msize"))

#################### Spatial autocorrelation diagram ####################

example_subj = 60
fn = f"_f1_cache_examplecm_{data.name}_{example_subj}.pkl"
if os.path.exists(fn):
    example_cm = util.pload(fn)
else:
    example_cm = data.get_matrices()[example_subj]
    util.psave(fn, example_cm)

example_params = data.get_lmbda()[example_subj], data.get_floor()[example_subj]
example_dist = data.get_dists()


cm_flat = example_cm.flatten()
dist_flat = example_dist.flatten()

df = pandas.DataFrame(np.asarray([dist_flat, cm_flat]).T, columns=["dist", "corr"])
discretization = 10
df['dist_bin'] = np.round(df['dist']/discretization)*discretization
df_binned = df.groupby('dist_bin').mean().reset_index().sort_values('dist_bin')
df_binned_sem = df.groupby('dist_bin').sem().reset_index().sort_values('dist_bin')
binned_dist_flat = df_binned['dist_bin']
binned_cm_flat = df_binned['corr']
binned_cm_sem_flat = df_binned_sem['corr']
binned_dist_flat[0] = 0
spatialfunc = lambda v : np.exp(-binned_dist_flat/v[0])*(1-v[1])+v[1]


ax = c.ax("sadiagram")
ax.cla()

linecolor = 'r'#(0.8, .2, .2)
labelcolor = (0,170/255,0)

#rand_i = np.random.permutation(len(dist_flat))[0:1000]
#ax.scatter(dist_flat[rand_i], cm_flat[rand_i], c='#aaaaaa', s=.2, alpha=.4)
#ax.fill_between(binned_dist_flat, binned_cm_flat+binned_cm_sem_flat*1.96, binned_cm_flat-binned_cm_sem_flat*1.96, color='r')
ax.bar(binned_dist_flat, binned_cm_flat, discretization, color='k')

#ax.plot(binned_dist_flat, binned_cm_flat, c='r', linewidth=2)
#ax.errorbar(binned_dist_flat, binned_cm_flat, yerr=binned_cm_sem_flat, c='k')
ax.plot(binned_dist_flat, spatialfunc(example_params), linewidth=2, c=linecolor)

ax.set_xlabel("Distance")
ax.set_ylabel("FC")
ax.set_ylim(0, 1)
ax.set_xlim(0, 90)

c.add_arrow(Point(1.02, 0, ("axis_sadiagram", "sadiagram")), Point(1.02, .15, ("axis_sadiagram", "sadiagram")), arrowstyle="|-|,widthA=2,widthB=2", shrinkA=0, shrinkB=0, color=labelcolor)
c.add_text(names_for_stuff["floor"], Point(1.0, .28, ("axis_sadiagram", "sadiagram")), horizontalalignment="center", verticalalignment="center", color=labelcolor)
c.add_arrow(Point(0, .37, "sadiagram"), Point(20, .37, "sadiagram"), arrowstyle="|-|,widthA=2,widthB=2", color=labelcolor, shrinkA=0, shrinkB=0)
c.add_text(names_for_stuff["lmbda"], Point(30, .45, "sadiagram"), color=labelcolor)
#c.add_text("$\\lambda$ = spatial autocorrelation", Point(20, .5, "sadiagram"), horizontalalignment="left", verticalalignment="center")
ax.set_yticks([0, .5, 1])
ax.set_xticks([])
sns.despine(ax=ax)
c.add_legend(Point(.2, .85, "axis_sadiagram"),
             [("Binned correlations", {"color": "k", "linewidth": 4}), ("Best fit line", {"color": linecolor, "linewidth": 2})],
             sym_width=Vector(.8, 0, "Msize"), line_spacing=Vector(0, 1.5, "Msize"))

#c.add_text("$\\rho(i,j) = \\text{SA-∞} + (1-\\text{SA-∞}) e^{-d_{ij}/\\text{SA-λ}}$", Point(.6, .35, "axis_sadiagram"), size=7)
#c.add_image("equation.png", Point(.05, .85, "axis_sadiagram"), ha="left", width=Vector(2.8, 0, "cm"))
c.add_text("Definition of SA-λ & SA-∞", Point(.5, 0, ("axis_sadiagram", "axis_minigrid"))+Vector(0, 1.2, "in"), ha="center", **titlestyle)



ax = c.ax("ar1diagram")
ax.cla()
rn = np.random.RandomState(2).random(10)
ax.plot(range(0, 10), rn, marker="o", c='k', markersize=1)
ax.plot(range(1, 11), rn, marker="o", c='k', linestyle="--", dashes=(2,1), markersize=1)
c.add_text("$corr(x[t],x[t+1])$", Point(.8, 0.1, "axis_ar1diagram"))
ax.axis("off")
c.add_text("Definition of TA-$\Delta_1$", Point(.5, 0, "axis_ar1diagram")+Vector(0, .5, "in"), ha="center", **titlestyle)

#################### Plot mean AR1 on the brain ####################

mean_ar1s = np.mean(data.get_ar1s(), axis=0)
VRANGE = (0, 1)
fn = f"_cache_brain_ar1_{data.name}.png"
if not os.path.exists(fn): # No need to regenerate if it exists
    wbplot.pscalar(fn, mean_ar1s, vrange=VRANGE, cmap='viridis', transparent=True)

c.add_image(fn, brain_img_loc, height=brain_img_height, horizontalalignment="left", verticalalignment="bottom", unitname="brainimage")
c.add_colorbar("brainimage_colorbar3", Point(.2, 0, "brainimage")+Vector(0, -.4, "cm"), Point(.8, 0, "brainimage")+Vector(0, -.2, "cm"), cmap="viridis", bounds=VRANGE)
c.add_text("TA-$\Delta_1$", Point(.5, 0, "brainimage")+Vector(0, -.9, "cm"), horizontalalignment="center", va="top")
c.add_text("Mean regional TA-$\Delta_1$", Point(.5, 0, "brainimage")+Vector(0, 1.15, "in")-Vector(0, .4, "cm"), ha="center", **titlestyle)

VRANGE_ICC = (.2, .8)
fn = f"_cache_brain_reliability_{datatrt.name}.png"
if not os.path.exists(fn): # No need to regenerate if it exists
    region_reliabilities = {}
    region_reliabilities['ar1'] = [icc_full(datatrt.get_subject_info()['subject'], datatrt.get_ar1s()[:,i])[0] for i in range(0, datatrt.N_regions())]
    for met in ['mean', 'var', 'kurt']:
        region_reliabilities[met] = [icc_full(datatrt.get_subject_info()['subject'], datatrt.get_nodal_cmstats()[met][:,i])[0] for i in range(0, datatrt.N_regions())]
    for met in ['degree', 'centrality']:
        region_reliabilities[met] = [icc_full(datatrt.get_subject_info()['subject'], datatrt.get_nodal_metrics()[met][:,i])[0] for i in range(0, datatrt.N_regions())]
    wbplot.pscalar(fn, np.asarray(region_reliabilities['ar1']), vrange=VRANGE_ICC, cmap='viridis', transparent=True)
    util.psave(fn+".pkl", region_reliabilities)
else:
    region_reliabilities = util.pload(fn+".pkl")

c.add_image(fn, brain_reliability_img_loc, height=brain_img_height, horizontalalignment="left", verticalalignment="bottom", unitname="brainreliabilityimage")
c.add_colorbar("brainreliabilityimage_colorbar3", Point(.2, 0, "brainreliabilityimage")+Vector(0, -.4, "cm"), Point(.8, 0, "brainreliabilityimage")+Vector(0, -.2, "cm"), cmap="viridis", bounds=VRANGE_ICC)
c.ax("brainreliabilityimage_colorbar3").set_xlabel("Reliability of TA-$\Delta_1$ (ICC)")
c.add_text("Reliability of regional TA-$\Delta_1$", Point(.5, 0, "brainreliabilityimage")+Vector(0,1.15, "in")-Vector(0, .4, "cm"), ha="center", **titlestyle)

ax = c.ax("reliability_vs_ar1")
ax.cla()
corplot(np.mean(datatrt.get_ar1s(), axis=0), region_reliabilities['ar1'], "TA-$\Delta_1$ reliability (ICC)", "Mean regional TA-$\Delta_1$", ax=ax, showr2=False, alpha=.5, markersize=2, rasterized=True)
c.add_text("TA-$\Delta_1$ by reliability", Point(.5, 1, "axis_reliability_vs_ar1")+Vector(0,.2, "cm"), ha="center", **titlestyle)

# ax = c.ax("reliability_hist")
# ax.hist(region_reliabilities['ar1'], color='k', bins=20)
# ax.set_xlim(*c.ax("reliability_vs_ar1").get_xlim())
# ax.set_yticks([])
# ax.set_xticks([])
# sns.despine(ax=ax, left=True)


#################### Reliabilities of individual brain regions ####################

ax = c.ax("region_reliability")
ax.cla()
ordered_reliabilities = [region_reliabilities[k] for k in nodal_metrics_ar1]
bplot = ax.boxplot(ordered_reliabilities, medianprops={"color": "k"}, vert=False, showfliers=False)
sns.despine(ax=ax, left=True)
ax.axvline(0, c='k')
ax.set_yticks([])
ax.set_ylim(.5, len(nodal_metrics_ar1)+.5)
ax.set_xlabel("Reliability (ICC)")
ax.invert_yaxis()
c.add_text("Reliability of nodal metrics", Point(.5, 0, "axis_region_reliability")+Vector(0, 1.15, "in"), ha="center", **titlestyle)
bplot['boxes'][0].set_color('r')
bplot['medians'][0].set_color('r')
bplot['whiskers'][0].set_color('r')
bplot['whiskers'][1].set_color('r')
bplot['caps'][0].set_color('r')
bplot['caps'][1].set_color('r')

for i in range(0, len(nodal_metrics_ar1)):
    c.add_text(short_names_for_stuff[nodal_metrics_ar1[i]], Point(-.05, i+1, "region_reliability"), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)


#################### Degree vs AR1 ####################

tocor = {
    "mean": data.get_nodal_cmstats()['mean'],
    "var": data.get_nodal_cmstats()['var'],
    "kurt": data.get_nodal_cmstats()['kurt'],
    "ts_mean": data.get_nodal_cmstats()['ts_mean'],
    "ts_var": data.get_nodal_cmstats()['ts_var'],
    "ts_kurt": data.get_nodal_cmstats()['ts_kurt'],
    "lefficiency": data.get_nodal_metrics()['lefficiency'],
    "cluster": data.get_nodal_metrics()['cluster'],
    "centrality": data.get_nodal_metrics()['centrality'],
    "degree": data.get_nodal_metrics()['degree'],
}

# same_subject = {k : [scipy.stats.spearmanr(tocor[k][i], data.get_ar1s()[i]).correlation for i in range(0, data.N_subjects())] for k in tocor.keys()}
# rng = np.random.RandomState(0)
# diff_subject = {k : [scipy.stats.spearmanr(tocor[k][rng.randint(data.N_subjects())], data.get_ar1s()[rng.randint(data.N_subjects())]).correlation for _ in range(0, 1000)] for k in tocor.keys()}
# diff_regions = {k : [scipy.stats.spearmanr(tocor[k][rng.randint(data.N_subjects())][rng.permutation(data.N_regions())], data.get_ar1s()[rng.randint(data.N_subjects())][rng.permutation(data.N_regions())]).correlation for _ in range(0, 1000)] for k in tocor.keys()}


# ax = c.ax("nodal_degree_ar1_hist")
# ax.cla()
# degree_corr = [scipy.stats.spearmanr(data.get_ar1s()[i], data.get_nodal_metrics()['degree'][i]).correlation for i in range(0, data.N_subjects())]
# lefficiency_corr = [scipy.stats.spearmanr(data.get_ar1s()[i], data.get_nodal_metrics()['lefficiency'][i]).correlation for i in range(0, data.N_subjects())]
# centrality_corr = [scipy.stats.spearmanr(data.get_ar1s()[i], data.get_nodal_metrics()['centrality'][i]).correlation for i in range(0, data.N_subjects())]
# degree_corr_scramble = [scipy.stats.spearmanr(data.get_ar1s()[i], np.sum(data.get_thresh(), axis=1)[i,np.random.permutation(data.N_regions())]).correlation for i in range(0, data.N_subjects())]
# bins = np.linspace(-1, 1, 31)
# ax.hist(degree_corr, bins=bins, density=True, alpha=.5)
# ax.hist(lefficiency_corr, bins=bins, density=True, alpha=.5)
# ax.hist(centrality_corr, bins=bins, density=True, alpha=.5)
# ax.hist(degree_corr_scramble, bins=bins, density=True, alpha=.5)
# c.add_legend(Point(.08, .85, "axis_nodal_degree_ar1_hist"), [("Data", {"linewidth": 4, "c": sns.color_palette()[0]}),
#                                                         ("Scramble", {"linewidth": 4, "c": sns.color_palette()[1]})],
#              sym_width=Vector(.8, 0, "Msize"), line_spacing=Vector(0, 1.8, "Msize"))
# sns.despine(ax=ax)
# ax.set_yticks([])
# ax.set_ylabel("Fraction of subjects")
# ax.axvline(0, c='k', linewidth=.5)
# c.add_text("Degree-AR1 regional correlations", Point(.5, 0, "axis_nodal_degree_ar1_hist")+Vector(0, -.25, "in"), horizontalalignment="center")

ax = c.ax("ar1_corr_boxplots")
ax.cla()
all_nodal_metrics = data.get_nodal_metrics()
all_nodal_metrics.update(data.get_nodal_cmstats())
ordered_corrs = [[scipy.stats.spearmanr(data.get_ar1s()[i], all_nodal_metrics[k][i]).correlation for i in range(0, data.N_subjects())] for k in nodal_metrics]
#colors = [(.3, .3, .3)]*3+["k"]*2
colors = ["k"]*5
for col in set(colors):
    ax.boxplot([oc for co,oc in zip(colors,ordered_corrs) if co == col], positions=[i+1 for i,co in enumerate(colors) if co==col], medianprops={"color": col}, boxprops={"color": col}, whiskerprops={"color": col}, capprops={"color": col}, vert=False, showfliers=False, widths=.5)

sns.despine(ax=ax, left=True)
ax.axvline(0, c='k')
ax.set_yticks([])
ax.set_xlabel("Spearman correlation")
ax.set_ylim(.5, len(nodal_metrics)+.5)
ax.invert_yaxis()
c.add_text("Correlation with TA-$\Delta_1$", Point(.5, 1, "axis_ar1_corr_boxplots")+Vector(0,.2, "cm"), ha="center", **titlestyle)

for i in range(0, len(nodal_metrics)):
    c.add_text(short_names_for_stuff[nodal_metrics[i]], Point(-.05, i+1, ("axis_ar1_corr_boxplots", "ar1_corr_boxplots")), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)


for nm in ["var", "degree"]:
    axname = f"nodal_{nm}_ar1_mean"
    ax = c.ax(axname)
    ax.cla()
    mean_nm = np.mean(all_nodal_metrics[nm], axis=0)
    simplescatter(np.mean(data.get_ar1s(), axis=0), mean_nm, ax=ax, s=.45, c='k', diag=False, alpha=.8, linewidth=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    c.add_text(short_names_for_stuff["ar1"], Point(.5, -.2, "axis_"+axname), ha="center", va="center", size=5)
    c.add_text(short_names_for_stuff[nm], Point(-.2, .5, "axis_"+axname), ha="center", va="center", size=5, rotation=90)
    ind = nodal_metrics.index(nm)
    arrowfrm = Point(1, ind+1, "ar1_corr_boxplots")
    arrowto = Point(-.35, .5, f"axis_nodal_{nm}_ar1_mean")
    c.add_arrow(arrowfrm, arrowto, lw=1, arrowstyle=ARROWSTYLE)

#################### Diagram ####################

c.set_default_unit("diagram")

# Set up timeseries we will use for this diagram
tss = np.random.RandomState(1).randn(8, 50)+.4*np.sin(np.linspace(0, 2*np.pi))

# Make the timeseries
ax = c.add_axis("dia_timeseries", Point(0, .5), Point(1, 1.5))
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')

ax.axis('off')

# Make the correlation matrix
ax = c.add_axis("dia_cm", Point(1.75, .5), Point(2.75, 1.5))
ax.imshow(np.corrcoef(tss)-np.eye(tss.shape[0]), cmap="bwr", vmin=-.8, vmax=.8)
ax.axis("off")

# Make the adjacency matrix
ax = c.add_axis("dia_adj", Point(3.5, .5), Point(4.5, 1.5))
thresh = util.threshold_matrix(util.make_perfectly_symmetric(np.corrcoef(tss)), .3)
ax.imshow(thresh, cmap="gray_r")
ax.set_xticks([])
ax.set_yticks([])

# Draw graph
c.add_unit("dia_graph", Vector(1, 1), Point(5.25, .5))
base = Point(.5, .5, "dia_graph")
a = 1/(1+np.sqrt(2))
ds = (.5 - a/np.sqrt(2))*.8
dl = .5*.8
# Could substitute this octagon for a circle, would allow for changing N_timeseries
node_loc = [base + Vector(-ds, dl),
            base + Vector(ds, dl),
            base + Vector(dl, ds),
            base + Vector(dl, -ds),
            base + Vector(ds, -dl),
            base + Vector(-ds, -dl),
            base + Vector(-dl, -ds),
            base + Vector(-dl, ds)]

# Draw edges
for i in range(0, 8):
    for j in range(0, i):
        if thresh[i,j]:
            c.add_line(node_loc[i], node_loc[j], c='k')

# Draw nodes
text_offset = Vector(0, -.01)
for i in range(0, 8):
    c.add_marker(node_loc[i], marker="o", markersize=8, c='k')
    #c.add_text(str(i), node_loc[i]+text_offset, color='w', size=6, horizontalalignment="center", verticalalignment="center")


# Add titles
title_height = Vector(0, .085)
c.add_text("Timeseries", Point(.5, 1, "axis_dia_timeseries") + title_height, size=7)
c.add_text("Correlation matrix", Point(.5, 1, "axis_dia_cm") + title_height, size=7)
c.add_text("Adjacency matrix", Point(.5, 1, "axis_dia_adj") + title_height, size=7)
c.add_text("Graph", Point(.5, 1, "dia_graph") + title_height, size=7)

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_arrow(Point(1, .5, "axis_dia_timeseries")+spacing, Point(0, .5, "axis_dia_cm")-spacing, arrowstyle=ARROWSTYLE)
c.add_arrow(Point(1, .5, "axis_dia_cm")+spacing, Point(0, .5, "axis_dia_adj")-spacing, arrowstyle=ARROWSTYLE)
c.add_text("=", Point(.5, 1, "axis_dia_adj") | Point(.5, 0, "dia_graph"), size=24)

# Add text above the arrows
offset = Vector(0, .1, "in")
c.add_text("Pairwise\ncorrelation", offset+(Point(1, .5, "axis_dia_timeseries") | Point(0, .5, "axis_dia_cm")), horizontalalignment="center", verticalalignment="bottom")
c.add_text("Threshold", offset + (Point(1, .5, "axis_dia_cm") | Point(0, .5, "axis_dia_adj")), horizontalalignment="center", verticalalignment="bottom")


# Add metric arrows
c.add_image("arrow.png", Point(.05, .45, ("axis_dia_timeseries", "diagram")), height=Vector(0, .3), width=Vector(.3, 0), horizontalalignment="left", verticalalignment="top")
c.add_text("Timeseries metrics:", Point(.4, .40, ("axis_dia_timeseries", "diagram")), horizontalalignment="left", verticalalignment="top", weight="bold", size=5)
c.add_text("SA-λ, SA-∞, TA-$\Delta_1$", Point(.4, .25, ("axis_dia_timeseries", "diagram")), horizontalalignment="left", verticalalignment="top", size=5)

c.add_image("arrow.png", Point(.05, .45, ("axis_dia_cm", "diagram")), height=Vector(0, .3), width=Vector(.3, 0), horizontalalignment="left", verticalalignment="top")
c.add_text("Weighted graph metrics:", Point(.4, .4, ("axis_dia_cm", "diagram")), horizontalalignment="left", verticalalignment="top", weight="bold", size=5)
c.add_text("mean-FC, var-FC, kurt-FC", Point(.4, .25, ("axis_dia_cm", "diagram")), horizontalalignment="left", verticalalignment="top", size=5)
c.add_text("Weighted nodal metrics:", Point(.4, .1, ("axis_dia_cm", "diagram")), horizontalalignment="left", verticalalignment="top", weight="bold", size=5)
c.add_text("Nodal mean-FC, var-FC, kurt-FC", Point(.4, -.05, ("axis_dia_cm", "diagram")), horizontalalignment="left", verticalalignment="top", size=5)

gm_offset = Vector(.15, 0, "cm")
c.add_image("arrow.png", Point(.15, .45, ("axis_dia_adj", "diagram"))+gm_offset, height=Vector(0, .3), width=Vector(.3, 0), horizontalalignment="left", verticalalignment="top")
c.add_text("Graph metrics:", Point(.5, .4, ("axis_dia_adj", "diagram"))+gm_offset, horizontalalignment="left", verticalalignment="top", weight="bold", size=5)
c.add_text("assortativity, clustering,\nlocal/global efficiency,\nmodularity, transitivity", Point(.5, .25, ("axis_dia_adj", "diagram"))+gm_offset, horizontalalignment="left", verticalalignment="top", size=5)

graphoffset = Vector(.4, 0, "cm")
c.add_image("arrow.png", graphoffset+Point(.05, .45, ("dia_graph", "diagram")), height=Vector(0, .3), width=Vector(.3, 0), horizontalalignment="left", verticalalignment="top")
c.add_text("Nodal\nmetrics:", graphoffset+Point(.4, .4, ("dia_graph", "diagram")), horizontalalignment="left", verticalalignment="top", weight="bold", size=5)
c.add_text("degree,\ncentrality", graphoffset+Point(.4, .09, ("dia_graph", "diagram")), horizontalalignment="left", verticalalignment="top", size=5)

c.add_box(Point(0, 0, "diagram"), Point(6.25, 1.5, "diagram"), boxstyle="Round", zorder=-10, fill=True, facecolor='#f8f8f8', linewidth=.5)

ROW1_LABS = .95+ROW1
ROW2a_LABS = 1.68+ROW2
ROW2b_LABS = 1.2+ROW2
ROW3a_LABS = 1.2+ROW3
ROW3b_LABS = .9+ROW3
c.add_figure_labels([
    ("a", "diagram", Point(.05, ROW1_LABS, "in")),
    ("b", "minigrid", Point(4.0, ROW1_LABS, "in")),
    ("c", "sadiagram", Point(5.80, ROW1_LABS, "in")),
    ("d", "reliabilities", Point(.5, ROW2a_LABS, "in")),
    ("e", "grid", Point(2.15, ROW2a_LABS, "in")),
    ("f", "brainimage", Point(3.6, ROW2b_LABS, "in")),
    ("g", "region_reliability", Point(5.35, ROW2b_LABS, "in")),
    ("h", "brainreliabilityimage", Point(.2, ROW3a_LABS, "in")),
    ("i", "fingerprint", Point(1.9, ROW3a_LABS, "in")),
    ("j", "ar1_corr_boxplots", Point(3.5, ROW3b_LABS+.1, "in")),
    ("k", "reliability_vs_ar1", Point(6, ROW3b_LABS, "in")),
], size=8)

c.save(FILENAME)
c.show()

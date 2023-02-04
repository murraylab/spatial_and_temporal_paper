import datasets
import models
import cdatasets
from cand import Canvas, Vector, Point
import models
import seaborn as sns
import pandas
import numpy as np
import scipy
import util
import os.path
import itertools
from figurelib import corplot, corplot_c, MODEL_PAL, get_cm_lmbda_params, metric_names, lin, names_for_stuff, simplescatter, short_names_for_stuff, COLOR_CYCLE, icc_full, trt_bootstrap_icc
import matplotlib


FILENAME = "figure2-supp-geo.pdf"
MODEL = "Colorless"
DATASET = 'hcp'

import cdatasets
# data = [cdatasets.HCP(i) for i in range(0, 4)]
# dataretest = data[1:4]+[data[0]]

# fit_params = util.pload("fit_parameters.pkl")


from models_for_figure2 import all_cmodels
data,cmodels = all_cmodels['hcpgeo']
data_nongeo,cmodels_nongeo = all_cmodels['hcp']
datatrt = cdatasets.HCP1200GeoKindaLikeTRT(hemi="R")

titlestyle = {"weight": "bold", "size": 7}

# colorless_params = [[v for _,v in sorted(fit_params[(DATASET+str(i), "Colorless", 'eigsL2')].items())] for i in range(0, 4)]
# colorfull_params = [[v for _,v in sorted(fit_params[(DATASET+str(i), "Colorfull", 'eigsL2')].items())] for i in range(0, 4)]
# spaceonly_params = [[v for _,v in sorted(fit_params[(DATASET+str(i), "Spaceonly", 'eigsL2')].items())] for i in range(0, 4)]
# generative = [cdatasets.ModelColorless(fit_of=data[i], params=colorless_params[i], seed=25+i) for i in range(0, 4)]
# colorfull = [cdatasets.ModelColorfull(fit_of=data[i], params=colorfull_params[i], seed=30+i) for i in range(0, 4)]
# spaceonly = [cdatasets.ModelSpaceonly(fit_of=data[i], params=spaceonly_params[i], seed=35+i) for i in range(0, 4)]
# color_surrogate = [cdatasets.ColorSurrogate(surrogate_of=data[i], seed=40+i) for i in range(0, 4)]
# cftimeonly = [cdatasets.ModelColorfullTimeonly(fit_of=data[i], seed=i+20) for i in range(0, 4)]
# timeonly = [cdatasets.ModelColorlessTimeonly(fit_of=data[i], seed=i+15) for i in range(0, 4)]
# phase = [cdatasets.PhaseRandomize(surrogate_of=data[i], seed=i) for i in range(0, 4)]
# #color_surrogate_correlated = [cdatasets.ColorSurrogateCorrelated(surrogate_of=data[i], seed=45+i) for i in range(0, 4)]
# eigen = [cdatasets.Eigensurrogate(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
# zalesky = [cdatasets.Zalesky2012(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
# degreerand = [cdatasets.DegreeRandomize(surrogate_of=data[i], seed=i+5) for i in range(0, 4)]

# cmodels = [("retest", dataretest),
#            ("ColorlessHet", generative),
#            ("Colorsurrogate", color_surrogate),
#            #("Colorfull", colorfull),
#            ("Spaceonly", spaceonly),
#            #("ColorfullTimeonly", cftimeonly),
#            ("ColorlessTimeonly", timeonly),
#            ("phase", phase),
#            #("eigen", eigen),
#            ("zalesky2012", zalesky),
#            ("degreerand", degreerand)]
cmodels_ts = cmodels[0:-3]
cmodels_cm = cmodels[0:-1]

# This can be useful when changing models: [m.cache_all() for mods in cmodels for m in mods[1]]

metrics = ["assort", "cluster", "lefficiency", "gefficiency", "modularity", "transitivity"]
param_metrics = ["lmbda", "floor", "meanar1"]
weighted_metrics = ['meancor', 'varcor', 'kurtcor']
all_metrics = weighted_metrics + metrics


#################### Set up layout ####################

c = Canvas(7.2, 4.80, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)

ROW1 = 4.6



ROW15 = 3.55


ROW2 = 3.1

c.add_axis("geo_vs_nongeo_lmbda", Point(0.5, ROW2, "in"), Point(1.5, ROW2+1, "in"))
c.add_axis("geo_vs_nongeo_floor", Point(2.2, ROW2, "in"), Point(3.2, ROW2+1, "in"))

itemheight = .145
BOXSIZE = .125
c.add_axis("reliabilities", Point(4.1, ROW2, "in"), Point(4.65, ROW2+itemheight*len(all_metrics), "in"))
c.add_axis("grid", Point(5.5, ROW2, "in"), Point(5.5+BOXSIZE*5, ROW2+.15+BOXSIZE*9, "in"))


ROW3 = 1.45
ROW4 = 0.25

c.add_axis("barplots", Point(.4, ROW4, "in"), Point(3.2, ROW4+.9, "in"))
c.add_axis("barplots2", Point(.4, ROW3, "in"), Point(6.0, ROW3+1, "in"))

legendpos = Point(1.0, .9, "axis_barplots2")+Vector(.1, 0, "cm")


row4a = ["degreedist", "degree"]
c.add_axis("degreedist", Point(3.7, ROW4+.1, "in"), Point(5.1, ROW4+.8, "in"))
c.add_axis("degree", Point(5.7, ROW4+.1, "in"), Point(6.8, ROW4+.8, "in"))
#c.add_grid(row4a, 2, Point(6, ROW3, "in"), Point(6.8, ROW3+.9+1.1, "in"), size=Vector(.8, .8, "in"))

# betzelfigs = ["betzel_dist", "betzel_neighbors"]
# c.add_grid(betzelfigs, 1, Point(4.5, ROW4, "in"), Point(6.8, ROW4+.7, "in"), size=Vector(.7, .7, "in"))



c.add_legend(legendpos,
             [(names_for_stuff[m], {'linestyle': 'none', 'marker': 's', 'markersize': 6, 'color': MODEL_PAL[m]}) for m,_ in cmodels],
             line_spacing=Vector(0, 1.3, "Msize")
             )


#################### Degree distribution ####################

ax = c.ax("degreedist")
ax.cla()
bins = np.arange(0, 175, 2)+.5
ax.plot(bins[0:-1], np.histogram(np.sum(data[0].get_thresh(), axis=1).flatten(), bins=bins)[0]/data[0].N_subjects(), c='k', linewidth=5)
for modelname,model in cmodels[:0:-1]:
    ax.plot(bins[0:-1], np.histogram(np.sum(model[0].get_thresh(), axis=1).flatten(), bins=bins)[0]/data[0].N_subjects(), c=MODEL_PAL[modelname], linewidth=2)

ax.set_yscale('log')
ax.set_xscale('log')
# This disables minor ticks.  I really like minor ticks for log scaled axes,
# but I can't for the life of me figure out why they don't don't work on the y
# axis, so we just have to disable them on the x axis too.
ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=[]))
sns.despine(ax=ax)
c.add_text("Degree", Point(.5, -.35, "axis_degreedist"))
ax.set_ylabel("# nodes")
c.add_text("Degree distribution", Point(.5, 1.0, "axis_degreedist")+Vector(0,.2, "cm"), **titlestyle)


#################### Grid ####################



ts_metrics_vals = {"meanar1": np.mean(data[0].get_ar1s(), axis=1),
                   "loglmbda": np.log(data[0].get_lmbda()),
                   "lmbda": data[0].get_lmbda(),
                   "floor": data[0].get_floor(),
                   "meancor": data[0].get_cmstats()['meancor'],
                   "varcor": data[0].get_cmstats()['varcor'],
                   "kurtcor": data[0].get_cmstats()['kurtcor'],
}
ts_metrics = ["lmbda", "floor", "spatialmetrics", "meanar1", "allmetrics"] # List them because dicts are unordered

graph_metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity', 'meancor', 'varcor', 'kurtcor']

graph_metric_values = data[0].get_metrics()
graph_metric_values.update(data[0].get_cmstats())

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


c.add_text("Correlation with graph metrics", Point(.5, 0, "axis_grid")+Vector(0, 1.65, "in"), ha="center", **titlestyle)




#################### Reliabilities ####################

ax = c.ax("reliabilities")


all_metrics_rel = ["lmbda", "floor", "meanar1", 'assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity', "meancor", "varcor", "kurtcor"]

metrics_vals = {"meanar1": np.mean(datatrt.get_ar1s(), axis=1),
                   "loglmbda": np.log(datatrt.get_lmbda()),
                   "lmbda": datatrt.get_lmbda(),
                   "floor": datatrt.get_floor(),
}
metrics_vals.update(datatrt.get_cmstats(True))
metrics_vals.update(datatrt.get_metrics(True))

fn = f"_cache_brain_metric_reliability_{datatrt.name}.pkl"
if not os.path.exists(fn):
    reliabilities = {k : icc_full(datatrt.get_subject_info()['subject'], metrics_vals[k]) for k in metrics_vals.keys()}
    util.psave(fn, reliabilities)
else:
    reliabilities = util.pload(fn)

fn = f"_cache_brain_metric_reliability_{datatrt.name}_simulated.pkl"
if not os.path.exists(fn):
    reliabilities_simulated = {}
    for k in all_metrics_rel:
        reliabilities_simulated[k] = trt_bootstrap_icc(datatrt, np.asarray(metrics_vals[k]), 1000)
        util.psave(fn, reliabilities_simulated, overwrite=True)
        print("Saved", k)
else:
    reliabilities_simulated = util.pload(fn)

ax.barh(range(0, len(all_metrics_rel)), [reliabilities[k][0] for k in all_metrics_rel], xerr=np.asarray([np.abs(np.asarray(reliabilities[k][1])-reliabilities[k][0]) for k in all_metrics_rel]).T, color=['r' if k in ['meanar1', 'lmbda', 'floor'] else (.5, .5, .5) if 'cor' in k else 'k' for k in all_metrics_rel], clip_on=False, height=.8, error_kw={"clip_on": False})
# ax.errorbar([reliabilities[k][0] for k in all_metrics], range(0, len(all_metrics)), xerr=np.asarray([np.abs(np.asarray(reliabilities[k][1])-reliabilities[k][0]) for k in all_metrics]).T, clip_on=False, color='k', linestyle='none')

ax.set_xlim(0, .8)
sns.despine(ax=ax)
ax.set_yticks([])
ax.set_ylim(-.5, len(all_metrics_rel) - .5)
ax.invert_yaxis()

reliability_ar1_sim = np.asarray([r[0] for r in reliabilities_simulated["meanar1"]])
reliability_lmbda_sim = np.asarray([r[0] for r in reliabilities_simulated["lmbda"]])
reliability_floor_sim = np.asarray([r[0] for r in reliabilities_simulated["floor"]])
for i in range(0, len(all_metrics_rel)):
    reliability_i_sim = [r[0] for r in reliabilities_simulated[all_metrics_rel[i]]]
    lt_ar1_sim = np.mean(reliability_ar1_sim<=reliability_i_sim)
    lt_lmbda_sim = np.mean(reliability_lmbda_sim<=reliability_i_sim)
    lt_floor_sim = np.mean(reliability_floor_sim<=reliability_i_sim)
    rs = ""
    rs += "#" if lt_lmbda_sim < .01 else "+" if lt_lmbda_sim < .05 else "-"
    rs += "#" if lt_floor_sim < .01 else "+" if lt_floor_sim < .05 else "-"
    rs += "#" if lt_ar1_sim < .01 else "+" if lt_ar1_sim < .05 else "-"
    c.add_text(rs, Point(reliabilities[all_metrics_rel[i]][0], i+.4, "reliabilities")+Vector(.35, .1, "cm"), horizontalalignment="center", verticalalignment="center", size=5, font="Noto Mono", weight="regular")


    # if reliabilities[all_metrics[i]][2] <.01:
    #     c.add_text("**", Point(reliabilities[all_metrics[i]][0], i+.4, "reliabilities")+Vector(.25, 0, "cm"), horizontalalignment="center", verticalalignment="center", size=10)
    # elif reliabilities[all_metrics[i]][2] <.05:
    #     c.add_text("**", Point(reliabilities[all_metrics[i]][0], i+.4, "reliabilities")+Vector(.25, 0, "cm"), horizontalalignment="center", verticalalignment="center", size=10)


for i in range(0, len(all_metrics_rel)):
    c.add_text(names_for_stuff[all_metrics_rel[i]], Point(0, i, "reliabilities")+Vector(-.1, 0, "cm"), horizontalalignment="right", verticalalignment="center", size=5)

c.add_text("Test-retest reliability", Point(0, 0, "axis_reliabilities")+Vector(0,1.65, "in")-Vector(.6,.6,"cm"), ha="left", **titlestyle)
ax.set_xlabel("Reliability (ICC)")





#################### Geo vs non-geo ####################

ax = c.ax("geo_vs_nongeo_lmbda")
corplot(data[0].get_lmbda(), data_nongeo[0].get_lmbda(), "SA-λ under geodesic distance", "SA-λ under Euclidean distance", ax=ax, showr2=False, alpha=.5, markersize=2, rasterized=True)
c.add_text("SA-λ for geodesic and\nEuclidean distance", Point(.5, 1.3, "axis_geo_vs_nongeo_lmbda"), ha="center", **titlestyle)

ax = c.ax("geo_vs_nongeo_floor")
corplot(data[0].get_floor(), data_nongeo[0].get_floor(), "SA-∞ under geodesic distance", "SA-∞ under Euclidean distance", ax=ax, showr2=False, alpha=.5, markersize=2, rasterized=True)
c.add_text("SA-∞ for geodesic and\nEuclidean distance", Point(.5, 1.3, "axis_geo_vs_nongeo_floor"), ha="center", **titlestyle)


#################### Nodal metrics ####################

# ax = c.ax("degree")
# ax.cla()

# bins = np.linspace(-.5, 1, 16)
# for modelname,model in cmodels:
#     model_degree_corrs = [scipy.stats.spearmanr(np.sum(model[j].get_thresh()[i], axis=1), np.sum(data[j].get_thresh()[i], axis=1)).correlation for i in range(0, data[0].N_subjects()) for j in range(0, len(model))]
#     ax.hist(model_degree_corrs, bins=bins, density=True, alpha=.5, color=MODEL_PAL[modelname])


# sns.despine(ax=ax)
# ax.set_yticks([])
# ax.set_xlim(-.5, 1)
# ax.set_ylabel("Fraction of subjects")
# ax.set_xlabel("Model-data correlation")
# ax.axvline(0, c='k', linewidth=.5)

# c.add_text("Nodal degree", Point(.5, 1.05, "axis_degree"))



ax = c.ax("degree")
ax.cla()
nodal_metrics = ["mean", "var", "kurt", "degree", "centrality"]
model = cmodels[1][1]
data_values = data[0].get_nodal_metrics()
data_values.update(data[0].get_nodal_cmstats())
model_values = model[0].get_nodal_metrics()
model_values.update(model[0].get_nodal_cmstats())
ax.set_xlabel("Degree")

values = []
for metric in nodal_metrics:
    #_vals = np.asarray([scipy.stats.spearmanr(data_values[metric][:,j], model_values[metric][:,j]).correlation for j in range(0, model[0].N_regions())])
    _vals = np.asarray([lin(data_values[metric][:,j], model_values[metric][:,j]) for j in range(0, model[0].N_regions())])
    values.append(_vals[~np.isnan(_vals)])

# bplot = ax.boxplot(values, medianprops={"color": "k"}, boxprops={"color": "k"}, patch_artist=True, vert=False, showfliers=False)
# for box in bplot['boxes']:
#     box.set_facecolor('r')

boxcolor = 'k'
ax.boxplot(values, medianprops={"color": boxcolor}, boxprops={"color": boxcolor}, whiskerprops={"color": boxcolor}, capprops={"color": boxcolor}, vert=False, showfliers=False)
sns.despine(ax=ax, left=True)
ax.axvline(0, c='k')
ax.set_yticks([])
ax.set_xlabel("Lin's concordance")
ax.invert_yaxis()
c.add_text("Nodal data-model correlations", Point(.5, 1, "axis_degree")+Vector(0,.2, "cm"), ha="center", **titlestyle)

for i in range(0, len(nodal_metrics)):
    c.add_text(short_names_for_stuff[nodal_metrics[i]], Point(.02, i+1, ("axis_degree", "degree")), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)

#################### Barplots ####################


weighted = [(modelname, metric, np.asarray(data[i].get_cmstats()[metric])[model[i].get_valid()], np.asarray(model[i].get_cmstats()[metric])[model[i].get_valid()])
            for modelname, model in cmodels_cm for metric in weighted_metrics for i in range(0, 4)]

unweighted = [(modelname, metric, np.asarray(data[i].get_metrics()[metric])[model[i].get_valid()], np.asarray(model[i].get_metrics()[metric])[model[i].get_valid()])
               for modelname, model in cmodels for metric in metrics for i in range(0, 4)]

params = [(modelname, "loglmbda", np.log(data[i].get_lmbda())[model[i].get_valid()], np.log(model[i].get_lmbda())[model[i].get_valid()]) for modelname,model in cmodels_cm for i in range(0, 4)] + \
          [(modelname, "floor", np.asarray(data[i].get_floor())[model[i].get_valid()], np.asarray(model[i].get_floor())[model[i].get_valid()]) for modelname,model in cmodels_cm for i in range(0, 4)] + \
          [(modelname, "meanar1", np.mean(data[i].get_ar1s(), axis=1)[model[i].get_valid()], np.mean(model[i].get_ar1s(), axis=1)[model[i].get_valid()]) for modelname,model in cmodels_ts for i in range(0, 4)]

rows = weighted + unweighted + params

lins = [(modelname, metric, lin(dpoints, mpoints)) for modelname, metric, dpoints, mpoints in rows]

ax = c.ax("barplots")
ax.cla()
df = pandas.DataFrame(lins, columns=["modelname", "metric", "lin"])
import matplotlib.pyplot as plt
df['metricname'] = df.apply(lambda x : names_for_stuff[x['metric']], axis=1)
sns.barplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[0:3]], errwidth=0)
strip = sns.stripplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[0:3]], dodge=True, size=2)
for pc in strip.collections:
    pc.set_facecolor('#555555')
# bp = sns.boxplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[0:3]], fliersize=0, width=.8)
# bp.legend_.remove()
# for i,box in enumerate(ax.artists):
#     bc = box.get_facecolor()
#     box.set_edgecolor(bc)
#     for j in range(6*i,6*(i+1)):
#          ax.lines[j].set_color(bc)
#     box.set_facecolor("white")


# sns.pointplot(data=df, x="metric", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=all_metrics, satuation=1, dodge=True, s=.5)
ax.legend([], [], frameon=False)
sns.despine(ax=ax, bottom=True)
ax.axhline(0, linewidth=1, c='k')
ax.xaxis.set_tick_params(top=False, bottom=False)
ax.xaxis.labelpad = -100
ax.set_xlabel("")
ax.set_ylabel("Lin's concordance")
ax.set_xlim(-.5, 2.5)
c.add_text("Model-data similarity", Point(.5, 1.05, "axis_barplots2"), **titlestyle)
for _t in ax.xaxis.get_majorticklabels():
    _t.set_y(.08)

ax = c.ax("barplots2")
ax.cla()
df = pandas.DataFrame(lins, columns=["modelname", "metric", "lin"])
df['metricname'] = df.apply(lambda x : names_for_stuff[x['metric']], axis=1)
import matplotlib.pyplot as plt
sns.barplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[3:9]], errwidth=0)
strip = sns.stripplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[3:9]], dodge=True, size=2)
for pc in strip.collections:
    pc.set_facecolor('#555555')

ax.legend([], [], frameon=False)
sns.despine(ax=ax, bottom=True)
ax.axhline(0, linewidth=1, c='k')
ax.xaxis.set_tick_params(top=False, bottom=False)
ax.set_xlabel("")
ax.set_ylabel("Lin's concordance")
for _t in ax.xaxis.get_majorticklabels():
    _t.set_y(.13)





ROW1LABELS = 4.6
c.add_figure_labels([("a", "geo_vs_nongeo_lmbda", Point(.35, ROW1LABELS, "in")),
                     ("b", "geo_vs_nongeo_floor", Point(1.95, ROW1LABELS, "in")),
                     ("c", "reliabilities", Point(3.7, ROW1LABELS, "in")),
                     ("d", "grid", Point(5.0, ROW1LABELS, "in")),
                     ("e", "barplots2"),
                     ("f", "degreedist"),
                     ("g", "degree", Vector(-1.2, 0, "cm")),
                     # ("h", "betzel_dist"),
                     # ("i", "betzel_neighbors"),
], size=8)


c.save(FILENAME)
c.show()

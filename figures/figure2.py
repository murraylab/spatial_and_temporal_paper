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
from figurelib import corplot_c, MODEL_PAL, get_cm_lmbda_params, metric_names, lin, names_for_stuff, simplescatter, short_names_for_stuff, COLOR_CYCLE
import matplotlib


FILENAME = "figure2.pdf"
MODEL = "Colorless"
DATASET = 'hcp'

import cdatasets
# data = [cdatasets.HCP(i) for i in range(0, 4)]
# dataretest = data[1:4]+[data[0]]

# fit_params = util.pload("fit_parameters.pkl")


from models_for_figure2 import all_cmodels
data,cmodels = all_cmodels['hcp']

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
param_metrics = ["loglmbda", "floor", "meanar1"]
weighted_metrics = ['meancor', 'varcor', 'kurtcor']
all_metrics = weighted_metrics + metrics


#################### Set up layout ####################

c = Canvas(7.2, 5.70, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)

ROW1 = 4.6

c.add_unit("diagram", Vector(.55, .55, "in"), Point(.3, ROW1, "in"))

c.add_axis("eigs", Point(4.4, ROW1+.2, "in"), Point(5, ROW1+.2+.6, "in"))
c.add_axis("linexample", Point(5.6, ROW1+.2, "in"), Point(6.2, ROW1+.2+.6, "in"))

ROW15 = 3.55

c.add_grid(["cm_"+cm[0] for cm in cmodels], 1, Point(.01, ROW15, ("figure", "in")), Point(.99, ROW15+.65, ("figure", "in")), spacing=Vector(.1,0,"in"))

ROW2 = 2.6


c.add_grid(["graph_"+cm[0] for cm in cmodels], 1, Point(.01, ROW2, ("figure", "in")), Point(.99, ROW2+.9, ("figure", "in")), spacing=Vector(0,0,"in"))

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


#################### Example Lin plot ####################

# c = Canvas(4, 4, "in")
# c.add_axis("linexample", Point(1, 1, "in"), Point(2, 2, "in"))
ax = c.ax("linexample")

lindat = np.linspace(0, 10, 21)

rng = np.random.RandomState(4)

# low_precision_low_var = lindat/6.5
# high_precision_high_var = lindat + rng.randn(len(lindat))*1.95
# negative = lindat*-.29
# negative2 = lindat*-.21+6
# shifted = lindat-5.2

highspear = (lindat + rng.randn(len(lindat))*.6)/5
highr2 = lindat*0 + np.mean(lindat) + rng.randn(len(lindat))*.4
highall = lindat + rng.randn(len(lindat))*.6

items = [(highspear, "$\\approx 1$", "$\\ll 0$", "$\\approx 0$"),
         (highr2,    "$\\approx 0$", "$\\approx 0$", "$\\approx 0$"),
         (highall,   "$\\approx 1$", "$\\approx 1$", "$\\approx 1$")]

ax.cla()
sns.despine(ax=ax)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.set_xticks([])
ax.set_yticks([])
legend_items = []
linstyle = {"markersize": 2, "marker": '', "linewidth": 2}#, "linestyle": 'none'}
lincolors = sns.color_palette()[5:]
for totry,color in zip(items, lincolors):
    ax.plot(lindat, totry[0], c=color, **linstyle)

ident_style = dict(c="k", linestyle='--', linewidth=1.5)
ax.plot(lindat, lindat, c='k', linestyle='--', linewidth=1.5)
# c.add_legend(Point(1.3, .9, "axis_linexample"),
#              [("1.0", ident_style)]+[(item[1]+"\t"+item[2]+"\t"+item[3], dict(color=color, **linstyle)) for item,color in zip(items,lincolors)],
#              line_spacing=Vector(0, 1.5, "Msize"), sym_width=Vector(1.8, 0, "Msize"), padding_sep=Vector(.5, 0, "Msize"))

legendpos = Point(1.3, .7, "axis_linexample")

# legendpos = Point(2, 2, "in")
textspace = Vector(.2, .1, "in")
labels = ["r", "R$^2$", "Lin"]
for j in range(0, 3):
    c.add_text(labels[j], legendpos + textspace.width()*(j+1), ha="center", va="baseline", weight="bold")
    for i in range(0, len(items)):
        c.add_text(items[i][j+1], legendpos + textspace.width()*(j+1) + textspace.height()*-(i+1), ha="center", va="baseline")

c.add_text("Identity (x=y)", legendpos + textspace.width()*2 + textspace.height()*-4, ha="center", va="baseline")

linewidth = Vector(.23, 0, "cm")
lineshift = Vector(0, .1, "cm")
for i in range(0, len(items)):
    itempos = legendpos + textspace.height()*-(i+1) + lineshift
    c.add_line(itempos, itempos + linewidth, color=lincolors[i], **linstyle)

itempos = legendpos + textspace.height()*-(len(items)+1) + lineshift
c.add_line(itempos, itempos + linewidth, **ident_style, dashes=[2, 1])

c.add_text("Lin's concordance examples", Point(1.1, 1.1, "axis_linexample"), **titlestyle)

#################### Eigenvalues ####################

ax = c.ax("eigs")
ax.cla()
eig_subjs = np.random.RandomState(0).choice(range(0, data[0].N_subjects()), replace=False, size=20)
eig_subjs = [4]
eig_models = ["ColorlessHet"]
for i in eig_subjs:
    ax.plot(range(1, data[0].N_regions()+1), list(reversed(data[0].get_eigenvalues()[0])), c='k', linewidth=5)
    for modelname,model in cmodels_cm:
        if modelname not in eig_models:
            continue
        ev = np.asarray(list(reversed(model[0].get_eigenvalues()[0])))
        ev = ev[ev>1e-7]
        ax.plot(range(1, len(ev)+1), ev, c=MODEL_PAL[modelname], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
# This disables minor ticks.  I really like minor ticks for log scaled axes,
# but I can't for the life of me figure out why they don't don't work on the y
# axis, so we just have to disable them on the x axis too.
ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=[]))
sns.despine(ax=ax)
#ax.set_xlabel("Eigenvalue rank")
c.add_text("Eigenvalue rank", Point(.5, -.4, "axis_eigs"))
ax.set_ylabel("Eigenvalue")

c.add_text("Eigenvalues (rank-ordered)", Point(.5, 1.1, "axis_eigs"), **titlestyle)


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
# sns.pointplot(data=df, x="metric", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=all_metrics, satuation=1, dodge=True, s=.5)
# bp = sns.boxplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[3:9]], fliersize=0, width=.8)
# bp.legend_.remove()
# for i,box in enumerate(ax.artists):
#     bc = box.get_facecolor()
#     box.set_edgecolor(bc)
#     for j in range(6*i,6*(i+1)):
#          ax.lines[j].set_color(bc)
#     box.set_facecolor("white")

ax.legend([], [], frameon=False)
sns.despine(ax=ax, bottom=True)
ax.axhline(0, linewidth=1, c='k')
ax.xaxis.set_tick_params(top=False, bottom=False)
ax.set_xlabel("")
ax.set_ylabel("Lin's concordance")
for _t in ax.xaxis.get_majorticklabels():
    _t.set_y(.13)

# TODO add in later 
# for i in range(0, 12):
#     axname ="mini_"+all_metrics[i]
#     basepos = Point(i, 1.3, "barplots")
#     ext = Vector(.11, .11, "in")
#     try:
#         c.add_axis(axname, basepos-ext, basepos+ext)
#     except:
#         pass
#     ax = c.ax(axname)
#     ax.cla()
#     filt_rows = [r for r in rows if r[0] == MODEL and r[1] == all_metrics[i]]
#     simplescatter(filt_rows[0][2], filt_rows[0][3], ax=ax, c=MODEL_PAL[MODEL], s=1)
#     ax.set_xticks([])
#     ax.set_yticks([])

#################### Draw network ####################

import networkx
import fa2
import random

graphs = []
for cm in cmodels:
    graphs.append(networkx.from_numpy_array(cm[1][0].get_thresh(5)[300]))

fn = "_f2_fa2.pkl"
if util.plock(fn):
    positions = []
    for graph in graphs:
        random.seed(4)
        positions.append(fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(graph, pos=None, iterations=2000))
    util.psave(fn, positions)
else:
    positions = util.pload(fn)

for i in range(0, len(graphs)):
    ax = c.ax("graph_"+cmodels[i][0])
    ax.cla()
    networkx.draw_networkx_nodes(graphs[i], positions[i], node_size=2, node_color="k", alpha=0.8, ax=ax, linewidths=0)
    networkx.draw_networkx_edges(graphs[i], positions[i], edge_color=MODEL_PAL[cmodels[i][0]], alpha=(.15 if cmodels[i][0]!="retest" else .05), ax=ax)
    if cmodels[i][0] == "ColorlessHet":
        ax.invert_yaxis()
    ax.axis('off')
    ax.set_rasterized(True)
    titlename = short_names_for_stuff[cmodels[i][0] if "retest" != cmodels[i][0] else "data"]
    c.add_text(titlename, Point(.5, 1.0, "axis_cm_"+cmodels[i][0])+Vector(0, .1, "cm"), va="bottom")

#################### Adjacency matrices ####################

for cm in cmodels:
    ax = c.ax("cm_"+cm[0])
    ax.imshow(cm[1][0].get_thresh()[300], interpolation="none", cmap="gray_r")
    ax.axis("on")
    ax.patch.set_edgecolor(MODEL_PAL[cm[0]])
    ax.patch.set_linewidth(4)
    ax.set_xticks([])
    ax.set_yticks([])

# #################### Betzel ####################

# ax = c.ax("betzel_dist")
# df = pandas.read_pickle("line_dist.pandas.pkl")
# df_group = df.groupby(['eta', 'gamma'])['lmbda'].agg(['mean', 'sem']).reset_index().sort_values('eta')
# gammas = list(sorted(set(df_group['gamma'])))
# for i,gamma in enumerate(gammas):
#     df_gamma = df_group.query(f'gamma == {gamma}')
#     ax.errorbar(x=df_gamma['eta'], y=df_gamma['mean'], yerr=df_gamma['sem'], color=COLOR_CYCLE[i], elinewidth=.75)

# sns.despine(ax=ax)
# ax.set_xlabel("EC distance parameter")
# ax.set_ylabel("Generative model SA")
# c.add_legend(Point(1, .6, "axis_betzel_dist"),
#              list(zip([f"{g}" for g in gammas], [{"color": c} for c in COLOR_CYCLE])),
#              sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
# c.add_text("EC cluster", Point(1.2, .8, "axis_betzel_dist"))
# c.add_text("SA in the EC model", Point(.8, 1.1, "axis_betzel_dist"), **titlestyle)

# ax = c.ax("betzel_neighbors")
# df = pandas.read_pickle("line_nbrs.pandas.pkl")
# df_group = df.groupby(['eta', 'gamma'])['ar1'].agg(['mean', 'sem']).reset_index().sort_values('gamma')
# etas = list(sorted(set(df_group['eta'])))
# for i,eta in enumerate(etas):
#     df_eta = df_group.query(f'eta == {eta}')
#     ax.errorbar(x=df_eta['gamma'], y=df_eta['mean'], yerr=df_eta['sem'], color=COLOR_CYCLE[i], elinewidth=.75)

# sns.despine(ax=ax)
# ax.set_xlabel("EC cluster parameter")
# ax.set_ylabel("Generative model TA")
# c.add_legend(Point(1.1, .6, "axis_betzel_neighbors"),
#              list(zip([f"{e}" for e in etas], [{"color": c} for c in COLOR_CYCLE])),
#              sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
# c.add_text("EC distance", Point(1.2, 0.8, "axis_betzel_neighbors"))
# c.add_text("TA in the EC model", Point(.8, 1.1, "axis_betzel_neighbors"), **titlestyle)


#################### Diagram ####################

c.set_default_unit("diagram")

arrowstyle = dict(lw=1, arrowstyle="->,head_width=2,head_length=4")

# Set up timeseries we will use for this diagram
tss = util.spatial_temporal_timeseries(np.eye(8), np.asarray([util.make_spectrum(50, .72, 2, .03)]*8))

axis_names = ["dia_timeseries", "dia_spatial", "dia_whitenoise", "dia_output"]
c.add_grid(axis_names, 1, Point(.25, .25), Point(6.0, 1.25), size=Vector(1, 1))

# Make the timeseries
ax = c.ax("dia_timeseries")
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')

ax.axis('off')

# Show spatial correlation
from scipy.ndimage.filters import gaussian_filter
ax = c.ax("dia_spatial")
ax.imshow(gaussian_filter(np.random.RandomState(0).randn(100,100), sigma=10), cmap="gray_r")
ax.axis("off")

# Show adding random gaussian noise
ax = c.ax("dia_whitenoise")
tss_white = np.random.RandomState(5).random((8,1)) * np.random.RandomState(2).randn(8, 50) * 1.5
for i in range(0, tss_white.shape[0]):
    ax.plot(tss_white[i]*.3+i, clip_on=False, linewidth=.5, c='k')

ax.axis('off')


# Final timeseries
import models
ax = c.ax("dia_output")
ax.cla()
for i in range(0, tss.shape[0]):
    ax.plot(i+(tss[i] + tss_white[i])/8, clip_on=False, linewidth=.5, c='k')

ax.axis('off')

# Add titles
title_height = Vector(0, .05)
c.add_text("Filtered random\nwalk timeseries", Point(.5, 0, "axis_dia_timeseries") - title_height, size=5, verticalalignment="top")
c.add_text("Correlate according\nto SA-λ and SA-$\infty$", Point(.5, 0, "axis_dia_spatial") - title_height, size=5, verticalalignment="top")
c.add_text("Noise w/ heterogeneous\nvariance, matches TA-$\Delta_1$", Point(.5, 0, "axis_dia_whitenoise") - title_height, size=5, verticalalignment="top")


c.add_text("Smooth timeseries", Point(.5, 1, "axis_dia_timeseries") + title_height, size=6, verticalalignment="bottom")
c.add_text("Spatial embedding", Point(.5, 1, "axis_dia_spatial") + title_height, size=6, verticalalignment="bottom")
c.add_text("Regional noise", Point(.5, 1, "axis_dia_whitenoise") + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate timeseries", Point(.5, 1, "axis_dia_output") + title_height, size=6, verticalalignment="bottom")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_text("×", Point(1, .5, "axis_dia_timeseries") | Point(0, .5, "axis_dia_spatial"), size=16)
c.add_text("+", Point(1, .5, "axis_dia_spatial") | Point(0, .5, "axis_dia_whitenoise"), size=16)
c.add_arrow(Point(1, .5, "axis_dia_whitenoise")+spacing, Point(0, .5, "axis_dia_output")-spacing, **arrowstyle)

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(6.25, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[MODEL])
c.add_text("Spatiotemporal model", Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")

# # Remove axis labels
# for a in grid:
#     if a not in ["ar1ar1", "lmbdalmbda", "floorfloor"]:
#         c.ax(a).set_ylabel("")
#     if a not in ["floorfloor", "kurtcor", "lefficiency", "transitivity"]:
#         c.ax(a).set_xlabel("")


ROW1LABELS = 5.6
c.add_figure_labels([("a", "diagram", Point(.05, ROW1LABELS, "in")),
                     ("b", "eigs", Point(4, ROW1LABELS, "in")),
                     ("c", "linexample", Point(5.45, ROW1LABELS, "in")),
                     ("d", "cm_"+cmodels[0][0], Point(.05, 4.3, "in")),
                     ("e", "barplots2"),
                     ("f", "degreedist"),
                     ("g", "degree", Vector(-1.2, 0, "cm")),
                     # ("h", "betzel_dist"),
                     # ("i", "betzel_neighbors"),
], size=8)


c.save(FILENAME)
c.show()

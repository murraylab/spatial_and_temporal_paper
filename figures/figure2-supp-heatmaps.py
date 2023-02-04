from models import Model_Colorfull_Hom, Model_Colorless_Hom
import datasets
import util
import pandas
import os
import numpy as np
from figurelib import get_cm_lmbda_params, names_for_stuff, COLOR_CYCLE
from cand import Canvas, Vector, Point
import matplotlib
import seaborn as sns

# Prev



# lmbdas = np.linspace(5, 40, 4)
# ar1s = np.linspace(0.5, .9, 5)

dist = datasets.get_hcp_distances()

num_timepoints = 1100
TR = .72

MODEL = Model_Colorless_Hom
#MODEL = Model_Colorfull_Hom

if MODEL == Model_Colorfull_Hom:
    fn = f"_f2sheat_cache.pkl"
    floor = 0
    replicates = 1
    lmbdas = np.linspace(2, 40, 16)
    ar1s = np.linspace(0, .9, 17)
elif MODEL == Model_Colorless_Hom:
    fn = f"_f2sheat_less_cache.pkl"
    floor = .129
    replicates = 8
    lmbdas = np.linspace(2, 30, 13)
    ar1s = np.linspace(0, .7, 13)


if os.path.exists(fn):
    all_metrics = util.pload(fn)
else:
    all_metrics = []
    for i in range(0, len(lmbdas)):
        for j in range(0, len(ar1s)):
            for r in range(0, replicates):
                tss = MODEL.generate_timeseries(dist, params={"lmbda": lmbdas[i], "ar1": ar1s[j], "floor": floor}, num_timepoints=num_timepoints, TR=TR, seed=i*100+j*10000+r)
                cm = util.correlation_matrix_pearson(tss)
                #graph = util.threshold_matrix(util.make_perfectly_symmetric(cm))
                metrics = util.graph_metrics_from_cm(cm)
                params = get_cm_lmbda_params(cm, dist)
                metrics['direct_lmbda'] = params[0]
                metrics['direct_floor'] = params[1]
                metrics['meanar1'] = np.mean(util.get_ar1s(tss))
                all_metrics.append((i, j, r, metrics))
                print(i, r, lmbdas[i], j, ar1s[j])
    util.psave(fn, all_metrics)
    

all_metrics_d = {}
for i,j,r,mets in all_metrics:
    if (i,j) not in all_metrics_d.keys():
        all_metrics_d[(i,j)] = {}
    for k in mets.keys():
        if k not in all_metrics_d[(i,j)]:
            all_metrics_d[(i,j)][k] = []
        all_metrics_d[(i,j)][k].append(mets[k])



#c = Canvas(7.2, 2.5, "in", fontsize=6, fontsize_ticks=5)
c = Canvas(7.2, 6.5, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)
graph_metrics = ["assort", "cluster", "gefficiency", "lefficiency", "modularity", "transitivity"]
metrics = ["meanar1", "direct_lmbda", "direct_floor", "meancor", "varcor", "kurtcor"] + graph_metrics
c.add_grid(metrics, 2, Point(.5, 1.8, "in"), Point(6.8, 3.3, "in"), size=Vector(.4, .4, "in"))


betzelfigs = ["betzel_dist", "betzel_neighbors", "betzel_dist_cross", "betzel_neighbors_cross"]
c.add_grid(betzelfigs, 2, Point(2.4, 4.1, "in"), Point(4.8, 6.2, "in"), size=Vector(.7, .7, "in"))

titlestyle = {"weight": "bold", "size": 7}



ranges = {"meanar1": [.5, .7, .9],
          "direct_lmbda": [0, 20, 40],
          "direct_floor": [-.1, 0, .1],
          "meancor": [0, .15, .3],
          "varcor": [0, .015, .03],
          "kurtcor": [0, 5, 10],
          "assort": [0, .3, .6],
          "cluster": [.2, .4, .6],
          "lefficiency": [.4, .6, .8],
          "gefficiency": [.4, .5, .6],
          "modularity": [.2, .4, .6],
          "transitivity": [.003, .006, .009]}
          

#for i,m in enumerate(metric_names.keys()):
for i,m in enumerate(metrics):
    ax = c.ax(m)
    grid = [[np.mean(all_metrics_d[(i,j)][m]) for i in range(0, len(lmbdas))] for j in range(0, len(ar1s))]
    ax.imshow(grid, aspect='auto')#, vmin=ranges[m][0], vmax=ranges[m][-1]);
    ax.set_xlabel(names_for_stuff["lmbdagen"])
    if m in ["meanar1", "assort"]:
        ax.set_ylabel(names_for_stuff["ar1gen"])
    tick_inds = [0, (len(lmbdas)-1)//2, len(lmbdas)-1]
    ax.set_xticks(tick_inds)
    ax.set_xticklabels(map(lambda x : "%.2g" % x, list(lmbdas[tick_inds])))
    tick_inds = [0, (len(ar1s)-1)//2, len(ar1s)-1]
    ax.set_yticks(tick_inds)
    ax.set_yticklabels(map(lambda x : "%.2g" % x, list(ar1s[tick_inds])))
    ax.set_title(names_for_stuff[m.replace("direct_", "")])
    #cb = c.add_colorbar("cb_"+m, Point(1.05, 0, "axis_"+m), Point(1.15, 1, "axis_"+m), "viridis", bounds=(ranges[m][0], ranges[m][-1])) # bounds=(np.min(grid), np.max(grid)))#
    cb = c.add_colorbar("cb_"+m, Point(1.05, 0, "axis_"+m), Point(1.15, 1, "axis_"+m), cmap=matplotlib.cm.viridis, bounds=(np.min(grid), np.max(grid)))



section_label_offset = Vector(-.1, .2, "in")
model_label_offset = Vector(-.3, .4, "in")
c.add_text("Timeseries metrics:", section_label_offset+Point(0, 1, "axis_meanar1"), size=6, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("FC metrics:", section_label_offset+Point(0, 1, "axis_meancor"), size=6, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("Graph metrics:", section_label_offset+Point(0, 1, "axis_assort"), size=6, weight="bold", horizontalalignment="left", verticalalignment="bottom")

c.add_text("Generative model", model_label_offset+Point(0, 1, "axis_meanar1"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")










#################### Betzel heatmaps ####################

df = pandas.read_pickle("betzel_grid.pandas.pkl")

metrics_betzel = list(map(lambda x : x+"_betzel", graph_metrics))
c.add_grid(metrics_betzel, 1, Point(.5, .5, "in"), Point(6.8, .9, "in"), size=Vector(.4, .4, "in"))

#for i,m in enumerate(metric_names.keys()):
etas = np.asarray(list(reversed(sorted(set(df["eta"])))))
gammas = np.asarray(sorted(set(df["gamma"])))
#gammas = gammas[gammas <= 1]
#gammas = gammas[gammas >= -1]
for i,m in enumerate(metrics_betzel):
    ax = c.ax(m)
    mname = m[:-7]
    grid = [[np.mean(df.query(f"eta == {eta} and gamma == {gamma}")[mname]) for eta in etas] for gamma in gammas]
    ax.imshow(grid, aspect='auto')#, vmin=ranges[mname][0], vmax=ranges[mname][-1]);
    ax.set_xlabel("EC distance parameter")
    if mname == "assort":
        ax.set_ylabel("EC cluster parameter")
    tick_inds = [0, (len(etas)-1)//3, 2*(len(etas)-1)//3, len(etas)-1]
    ax.set_xticks(tick_inds)
    ax.set_xticklabels(map(lambda x : "%.2g" % x, list(etas[tick_inds])))
    tick_inds = [0, (len(gammas)-1)//2, len(gammas)-1]
    ax.set_yticks(tick_inds)
    ax.set_yticklabels(map(lambda x : "%.2g" % x, list(gammas[tick_inds])))
    ax.set_title(names_for_stuff[mname])
    #cb = c.add_colorbar("cb_"+m, Point(1.05, 0, "axis_"+m), Point(1.15, 1, "axis_"+m), "viridis", bounds=(ranges[mname][0], ranges[mname][-1])) # bounds=(np.min(grid), np.max(grid)))#
    cb = c.add_colorbar("cb_"+m, Point(1.05, 0, "axis_"+m), Point(1.15, 1, "axis_"+m), cmap=matplotlib.cm.viridis, bounds=(np.min(grid), np.max(grid)))
    #cb.set_ticks(list(ranges[m]))


c.add_text("Graph metrics:", section_label_offset+Point(0, 1, "axis_assort_betzel"), size=6, weight="bold", horizontalalignment="left", verticalalignment="bottom")

c.add_text("Economical clustering (EC) model", model_label_offset+Point(0, 1, "axis_assort_betzel"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")


# BETZEL


#################### Betzel - more important ####################

ax = c.ax("betzel_dist")
df = pandas.read_pickle("line_dist.pandas.pkl")
df_group = df.groupby(['eta', 'gamma'])['lmbda'].agg(['mean', 'sem']).reset_index().sort_values('eta')
gammas = list(sorted(set(df_group['gamma'])))
for i,gamma in enumerate(gammas):
    df_gamma = df_group.query(f'gamma == {gamma}')
    ax.errorbar(x=df_gamma['eta'], y=df_gamma['mean'], yerr=df_gamma['sem'], color=COLOR_CYCLE[i], elinewidth=.75)

sns.despine(ax=ax)
ax.set_xlabel("EC distance parameter")
ax.set_ylabel(names_for_stuff['lmbdagen'])
c.add_legend(Point(1, .6, "axis_betzel_dist"),
             list(zip([f"{g}" for g in gammas], [{"color": c} for c in COLOR_CYCLE])),
             sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
c.add_text("EC cluster\nparam", Point(1.2, .8, "axis_betzel_dist"))
c.add_text(names_for_stuff['lmbdagen']+" vs distance parameter", Point(.8, 1.1, "axis_betzel_dist"), **titlestyle)

ax = c.ax("betzel_neighbors")
df = pandas.read_pickle("line_nbrs.pandas.pkl")
df_group = df.groupby(['eta', 'gamma'])['ar1'].agg(['mean', 'sem']).reset_index().sort_values('gamma')
etas = list(sorted(set(df_group['eta'])))
for i,eta in enumerate(etas):
    df_eta = df_group.query(f'eta == {eta}')
    ax.errorbar(x=df_eta['gamma'], y=df_eta['mean'], yerr=df_eta['sem'], color=COLOR_CYCLE[i], elinewidth=.75)

sns.despine(ax=ax)
ax.set_xlabel("EC cluster parameter")
ax.set_ylabel(names_for_stuff['ar1gen'])
c.add_legend(Point(1.1, .6, "axis_betzel_neighbors"),
             list(zip([f"{e}" for e in etas], [{"color": c} for c in COLOR_CYCLE])),
             sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
c.add_text("EC distance\nparam", Point(1.2, 0.8, "axis_betzel_neighbors"))
c.add_text(names_for_stuff['ar1gen']+" vs cluster parameter", Point(.8, 1.1, "axis_betzel_neighbors"), **titlestyle)

#################### Betzel - less important ####################

ax = c.ax("betzel_dist_cross")
df = pandas.read_pickle("line_dist.pandas.pkl")
df_group = df.groupby(['eta', 'gamma'])['ar1'].agg(['mean', 'sem']).reset_index().sort_values('eta')
gammas = list(sorted(set(df_group['gamma'])))
for i,gamma in enumerate(gammas):
    df_gamma = df_group.query(f'gamma == {gamma}')
    ax.errorbar(x=df_gamma['eta'], y=df_gamma['mean'], yerr=df_gamma['sem'], color=COLOR_CYCLE[i], elinewidth=.75)

sns.despine(ax=ax)
ax.set_xlabel("EC distance parameter")
ax.set_ylabel(names_for_stuff['ar1gen'])
c.add_legend(Point(1.1, .6, "axis_betzel_dist_cross"),
             list(zip([f"{g}" for g in gammas], [{"color": c} for c in COLOR_CYCLE])),
             sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
c.add_text("EC cluster\nparam", Point(1.3, .8, "axis_betzel_dist_cross"))
c.add_text(names_for_stuff['ar1gen']+" vs distance parameter", Point(.8, 1.1, "axis_betzel_dist_cross"), **titlestyle)

ax = c.ax("betzel_neighbors_cross")
df = pandas.read_pickle("line_nbrs.pandas.pkl")
df_group = df.groupby(['eta', 'gamma'])['lmbda'].agg(['mean', 'sem']).reset_index().sort_values('gamma')
etas = list(sorted(set(df_group['eta'])))
for i,eta in enumerate(etas):
    df_eta = df_group.query(f'eta == {eta}')
    ax.errorbar(x=df_eta['gamma'], y=df_eta['mean'], yerr=df_eta['sem'], color=COLOR_CYCLE[i], elinewidth=.75)

sns.despine(ax=ax)
ax.set_xlabel("EC cluster parameter")
ax.set_ylabel(names_for_stuff['lmbdagen'])
c.add_legend(Point(1.1, .6, "axis_betzel_neighbors_cross"),
             list(zip([f"{e}" for e in etas], [{"color": c} for c in COLOR_CYCLE])),
             sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
c.add_text("EC distance\nparam", Point(1.3, 0.8, "axis_betzel_neighbors_cross"))
c.add_text(names_for_stuff['lmbdagen']+" vs cluster parameter", Point(.8, 1.1, "axis_betzel_neighbors_cross"), **titlestyle)

c.add_figure_labels([("a", "betzel_dist", Vector(-.2, 0, "cm")),
                     ("b", "betzel_neighbors", Vector(-.2, 0, "cm")),
                     ("c", "betzel_dist_cross", Vector(-.2, 0, "cm")),
                     ("d", "betzel_neighbors_cross", Vector(-.2, 0, "cm")),
                     ("e", "meanar1", Vector(-.2, 0, "cm")),
                     ("f", "meancor", Vector(-.2, 0, "cm")),
                     ("g", "assort", Vector(-.2, 0, "cm")),
                     ("h", "assort_betzel", Vector(-.2, .4, "cm")),
                    ], size=8)



c.save("figure2-supp-heatmaps.pdf")
#c.show()

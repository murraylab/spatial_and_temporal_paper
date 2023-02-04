import cdatasets
from cand import Canvas, Vector, Point
import seaborn as sns
import pandas
import numpy as np
import util
from figurelib import MODEL_PAL, lin, names_for_stuff, short_names_for_stuff


FILENAME = "figure2-supp-alldatasets.pdf"






from models_for_figure2 import all_cmodels



data_hcpgsr, cmodels_hcpgsr = all_cmodels["hcpgsr"]
data_camcan, cmodels_camcan = all_cmodels["camcan"]
data_trt, cmodels_trt = all_cmodels["trt"]





# This can be useful when changing models: [m.cache_all() for mods in cmodels for m in mods[1]]







#################### Set up layout ####################

c = Canvas(7.2, 7.6, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)




ROW2 = 1.75
ROW1 = ROW2 + 1

# c.add_axis("barplots", Point(.4, ROW1, "in"), Point(5.5, ROW1+.6, "in"))
# c.add_axis("barplots2", Point(.4, ROW2, "in"), Point(5.5, ROW2+.6, "in"))

# c.add_axis("degreedist", Point(6.4, ROW1, "in"), Point(6.4+.6, ROW1+.6, "in"))
# c.add_axis("degree", Point(6.4, ROW2, "in"), Point(6.4+.6, ROW2+.6, "in"))

legendpos = Point(6, 7.5, "in")

# ROW4 = .4

rowheight = 2.5
rowbase = .5
for i,modelname in enumerate(reversed(['hcpgsr', 'trt', 'camcan'])):
    ROW2 = rowbase + i*rowheight
    ROW1 = ROW2 + 1.15
    c.add_axis("barplots_"+modelname, Point(.4, ROW2, "in"), Point(2.95, ROW2+.6, "in"))
    c.add_axis("barplots2_"+modelname, Point(.4, ROW1, "in"), Point(5.5, ROW1+.6, "in"))
    c.add_axis("degreedist_"+modelname, Point(3.5, ROW2, "in"), Point(4.4, ROW2+.6, "in"))
    c.add_axis("degree_"+modelname, Point(5.0, ROW2, "in"), Point(5.9, ROW2+.6, "in"))
    c.add_text(names_for_stuff[modelname], Point(.1, ROW1+.85, "in"), weight="bold", size=7, ha="left")



c.add_legend(legendpos,
             [(names_for_stuff[m], {'linestyle': 'none', 'marker': 's', 'markersize': 6, 'color': MODEL_PAL[m]}) for m,_ in cmodels_hcpgsr],
             line_spacing=Vector(0, 1.3, "Msize")
             )



for sfx,cmodels,data in [("_hcpgsr", cmodels_hcpgsr, data_hcpgsr), ("_trt", cmodels_trt, data_trt), ("_camcan", cmodels_camcan, data_camcan)]:
    cmodels_ts = cmodels[:-3]
    cmodels_cm = cmodels[:-1]
    metrics = ["assort", "cluster", "lefficiency", "gefficiency", "modularity", "transitivity"]
    param_metrics = ["lmbda", "floor", "meanar1"]
    if sfx == "_camcan":
        param_metrics[0] = "loglmbda"
    weighted_metrics = ['meancor', 'varcor', 'kurtcor']
    all_metrics = param_metrics + weighted_metrics + metrics
    #################### Degree distribution ####################
    
    ax = c.ax("degreedist"+sfx)
    ax.cla()
    bins = np.arange(0, 175, 2)+.5
    ax.plot(bins[0:-1], np.histogram(np.sum(data[0].get_thresh(), axis=1).flatten(), bins=bins)[0]/data[0].N_subjects(), c='k', linewidth=5)
    for modelname,model in (cmodels[:0:-1] if sfx != "_camcan" else cmodels[::-1]):
        ax.plot(bins[0:-1], np.histogram(np.sum(model[0].get_thresh(), axis=1).flatten(), bins=bins)[0]/data[0].N_subjects(), c=MODEL_PAL[modelname], linewidth=2)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    sns.despine(ax=ax)
    ax.set_xlabel("Degree")
    ax.set_ylabel("# nodes")
    c.add_text("Degree distribution", Point(.5, 1.05, "axis_degreedist"+sfx))
    
    #################### Nodal metrics ####################
    
    ax = c.ax("degree"+sfx)
    ax.cla()
    nodal_metrics = ["mean", "var", "kurt", "degree", "centrality"]
    model = cmodels[1][1]
    data_values = data[0].get_nodal_metrics()
    data_values.update(data[0].get_nodal_cmstats())
    model_values = model[0].get_nodal_metrics()
    model_values.update(model[0].get_nodal_cmstats())
    
    values = []
    for metric in nodal_metrics:
        _vals = np.asarray([lin(data_values[metric][:,j], model_values[metric][:,j]) for j in range(0, model[0].N_regions())])
        values.append(_vals[~np.isnan(_vals)])
    
    ax.boxplot(values, medianprops={"color": "r"}, boxprops={"color": "r"}, whiskerprops={"color": "r"}, capprops={"color": "r"}, vert=False, showfliers=False)
    sns.despine(ax=ax, left=True)
    ax.axvline(0, c='k')
    ax.set_yticks([])
    ax.set_xlabel("Lin's concordance")
    ax.invert_yaxis()
    c.add_text("Regional data-model correlations", Point(.5, 1, "axis_degree"+sfx)+Vector(0,.2, "cm"), ha="center")
    
    for i in range(0, len(nodal_metrics)):
        c.add_text(short_names_for_stuff[nodal_metrics[i]], Point(0, i+1, ("axis_degree"+sfx, "degree"+sfx)), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)
    
    #################### Barplots ####################
        
    
    weighted = [(modelname, metric, np.asarray(data[i].get_cmstats()[metric])[model[i].get_valid()], np.asarray(model[i].get_cmstats()[metric])[model[i].get_valid()])
                for modelname, model in cmodels_cm for metric in weighted_metrics for i in range(0, len(data))]
    
    unweighted = [(modelname, metric, np.asarray(data[i].get_metrics()[metric])[model[i].get_valid()], np.asarray(model[i].get_metrics()[metric])[model[i].get_valid()])
                for modelname, model in cmodels for metric in metrics for i in range(0, len(data))]
    
    params = [(modelname, "loglmbda", np.log(data[i].get_lmbda())[model[i].get_valid()], np.log(model[i].get_lmbda())[model[i].get_valid()]) for modelname,model in cmodels_cm for i in range(0, len(data))] + \
             [(modelname, "lmbda", np.asarray(data[i].get_lmbda())[model[i].get_valid()], np.asarray(model[i].get_lmbda())[model[i].get_valid()]) for modelname,model in cmodels_cm for i in range(0, len(data))] + \
             [(modelname, "floor", np.asarray(data[i].get_floor())[model[i].get_valid()], np.asarray(model[i].get_floor())[model[i].get_valid()]) for modelname,model in cmodels_cm for i in range(0, len(data))] + \
             [(modelname, "meanar1", np.mean(data[i].get_ar1s(), axis=1)[model[i].get_valid()], np.mean(model[i].get_ar1s(), axis=1)[model[i].get_valid()]) for modelname,model in cmodels_ts for i in range(0, len(data))]
    # weighted = [(modelname, metric, data[i].get_cmstats()[metric], model[i].get_cmstats()[metric])
    #             for modelname, model in cmodels_cm for metric in weighted_metrics for i in range(0, len(data))]
    
    # unweighted = [(modelname, metric, data[i].get_metrics()[metric], model[i].get_metrics()[metric])
    #             for modelname, model in cmodels for metric in metrics for i in range(0, len(data))]
    
    # params = [(modelname, "loglmbda", np.log(data[i].get_lmbda()), np.log(model[i].get_lmbda())) for modelname,model in cmodels_cm for i in range(0, len(data))] + \
    #         [(modelname, "floor", data[i].get_floor(), model[i].get_floor()) for modelname,model in cmodels_cm for i in range(0, len(data))] + \
    #         [(modelname, "meanar1", np.mean(data[i].get_ar1s(), axis=1), np.mean(model[i].get_ar1s(), axis=1)) for modelname,model in cmodels_ts for i in range(0, len(data))]
    
    rows = weighted + unweighted + params
    
    lins = [(modelname, metric, lin(dpoints, mpoints)) for modelname, metric, dpoints, mpoints in rows]
    
    ax = c.ax("barplots"+sfx)
    ax.cla()
    ax.set_xlim(-.5, 2.5)
    df = pandas.DataFrame(lins, columns=["modelname", "metric", "lin"])
    import matplotlib.pyplot as plt
    df['metricname'] = df.apply(lambda x : names_for_stuff[x['metric']], axis=1)
    sns.barplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[3:6]], errwidth=1)
    strip = sns.stripplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[3:6]], dodge=True, size=2)
    for pc in strip.collections:
        pc.set_facecolor('#555555')
    # sns.pointplot(data=df, x="metric", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=all_metrics, satuation=1, dodge=True, s=.5)
    ax.legend([], [], frameon=False)
    sns.despine(ax=ax, bottom=True)
    ax.axhline(0, linewidth=1, c='k')
    ax.xaxis.set_tick_params(top=False, bottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("Lin's concordance")
    
    ax = c.ax("barplots2"+sfx)
    ax.cla()
    ax.set_xlim(-.5, 5.5)
    df = pandas.DataFrame(lins, columns=["modelname", "metric", "lin"])
    import matplotlib.pyplot as plt
    df['metricname'] = df.apply(lambda x : names_for_stuff[x['metric']], axis=1)
    sns.barplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[6:12]], errwidth=1)
    strip = sns.stripplot(data=df, x="metricname", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=[names_for_stuff[m] for m in all_metrics[6:12]], dodge=True, size=2)
    for pc in strip.collections:
        pc.set_facecolor('#555555')
    # sns.pointplot(data=df, x="metric", hue="modelname", y="lin", ax=ax, palette=MODEL_PAL, order=all_metrics, satuation=1, dodge=True, s=.5)
    ax.legend([], [], frameon=False)
    sns.despine(ax=ax, bottom=True)
    ax.axhline(0, linewidth=1, c='k')
    ax.xaxis.set_tick_params(top=False, bottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("Lin's concordance")



c.add_figure_labels([("a", "barplots2_hcpgsr"),
                     ("b", "degreedist_hcpgsr", Vector(-.3, 0, "cm")),
                     ("c", "degree_hcpgsr", Vector(-.3, 0, "cm")),
                     ("d", "barplots2_trt"),
                     ("e", "degreedist_trt", Vector(-.3, 0, "cm")),
                     ("f", "degree_trt", Vector(-.3, 0, "cm")),
                     ("g", "barplots2_camcan"),
                     ("h", "degreedist_camcan", Vector(-.3, 0, "cm")),
                     ("i", "degree_camcan", Vector(-.3, 0, "cm")),
], size=8)


c.save(FILENAME)
c.show()

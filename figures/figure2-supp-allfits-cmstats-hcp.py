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
from figurelib import corplot, corplot_sig, MODEL_PAL, get_cm_lmbda_params, metric_names, lin, names_for_stuff, simplescatter, short_names_for_stuff


FILENAME = f"figure2-supp-allfits-cmstats-hcp.pdf"
MODEL = "Colorless"
DATASET = 'hcp'

import cdatasets


from models_for_figure2 import all_cmodels
data,cmodels = all_cmodels['hcp']
cmodels = cmodels[:-1] # Remove degreerand

# data = [cdatasets.HCP(i) for i in range(0, 4)]
# dataretest = data[1:4]+[data[0]]

# fit_params = util.pload("fit_parameters.pkl")

# # model = ...

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
#            ("Colorfull", colorfull),
#            ("Spaceonly", spaceonly),
#            #("ColorfullTimeonly", cftimeonly),
#            ("ColorlessTimeonly", timeonly),
#            ("zalesky2012", zalesky),
#            ("phase", phase),
#            #("eigen", eigen),
#            #("degreerand", degreerand),
#            ]
cmodels_ts = cmodels[0:-3]
cmodels_cm = cmodels[0:-1]

# This can be useful when changing models: [m.cache_all() for mods in cmodels for m in mods[1]]

param_metrics = ["lmbda", "floor", "meanar1"]
weighted_metrics = ['meancor', 'varcor', 'kurtcor']
metrics = param_metrics + weighted_metrics

def metric_of_interest(model, metric, valid=None):
    if valid is None:
        valid = model.get_valid()
    try:
        if metric == "lmbda":
            vals = model.get_lmbda()
        elif metric == "floor":
            vals = model.get_floor()
        elif metric == "meanar1":
            vals = np.mean(model.get_ar1s(), axis=1)
        else:
            vals = model.get_cmstats()[metric]
        return np.asarray(vals)[valid]
    except NotImplementedError:
        return [0]*model.N_subjects()

# Compute upper and lower bounds for each metric's plot.  We want them to all
# be the same for a given metric.
maxes = {met: np.max([np.max(metric_of_interest(session,met)) for m in cmodels for session in m[1]]) for met in metrics}
mins = {met: np.min([np.min(metric_of_interest(session,met)) for m in cmodels for session in m[1]]) for met in metrics}
bounds = {}
for met in metrics:
    rng = maxes[met] - mins[met]
    bounds[met] = (mins[met] - rng*.05, maxes[met] + rng*.05)

HEIGHT = 6.8


c = Canvas(7.2, HEIGHT, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)
#c.add_text(f"Graph metrics, model vs data - {DATASET}", Point(.5, .98), horizontalalignment="center", size=7)

names = [m[0]+"_"+met for m in cmodels for met in metrics]
c.add_grid(names, len(cmodels), Point(.5, .5, "in"), Point(6.8, HEIGHT-.4, "in"), size=Vector(.5, .5, "in"))


for mod in cmodels:
    c.add_box(Point(0, 0, f"axis_{mod[0]}_{metrics[0]}") - Vector(.45, .07, "in"),
              Point(1, 1, f"axis_{mod[0]}_{metrics[-1]}") + Vector(.3, .15, "in"),
              boxstyle="Round,pad=0,rounding_size=.3", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[mod[0]])
    for met in metrics:
        print(met, mod)
        axname = mod[0]+"_"+met
        ax = c.ax(axname)
        all_model_points = [el for mod_el in mod[1] for el in metric_of_interest(mod_el,met)]
        all_data_points = [el for data_el,mod_el in zip(data,mod[1]) for el in metric_of_interest(data_el,met, valid=mod_el.get_valid())]
        if not (met == "meanar1" and mod[0] == "zalesky2012"):
            corplot(all_data_points, all_model_points, "", "", diag=True, showr2="lin", ax=ax, markersize=1, color=(0, 0, 0, .04), rasterized=True)
        else:
            ax.axis("off")
            continue
        # Add lines to represent the mean on each axis
        ax.axvline(np.mean(all_data_points), c='gray', linestyle='--', linewidth=.5)
        ax.axhline(np.mean(all_model_points), c='gray', linestyle='--', linewidth=.5)
        
        ax.set_xlim(*bounds[met])
        ax.set_ylim(*bounds[met])
        # Add titles and adjust the aesthetics of being in a grid
        if mod == cmodels[0]:
            c.add_text(names_for_stuff[met], Point(.5, 1.4, "axis_"+axname), weight="bold", size=7)
        if met == metrics[0]:
            c.add_text(names_for_stuff[mod[0]], Point(-.7, 1.15, "axis_"+axname), weight="bold", size=7, ha="left")
            if mod != cmodels[0]:
                ax.set_ylabel("Model")
            else:
                ax.set_ylabel("Data (retest)")
        if mod != cmodels[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Data")
        if mod == cmodels[-2] and met == "meanar1":
            ax.set_xlabel("Data")
        sns.despine(ax=ax)


c.save(FILENAME)
c.show()

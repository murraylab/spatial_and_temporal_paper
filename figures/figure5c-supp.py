# Pharmacology figure
import cdatasets
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from cand import Canvas, Vector, Point
from figurelib import names_for_stuff, short_names_for_stuff, simplescatter
import os
import seaborn as sns
import scipy.stats
import pandas
import util

# Regional ar1 differences histogram (six histograms)
# X Subject lmbda, gc, ar1 differences
# X Graph metric differences
# Brain of mean ar1 differences at each timepoint (six brains)
# X Final row from figure1-supp-condensed
# X Correlation with HCP

GSR=True

_pal = sns.color_palette("Dark2")
PAL = [_pal[0], _pal[3], _pal[4]]
hueorder = ["LSD", "Psilocybin", "LSD+Ket"]
BOXSIZE = .11
itemheight_reli = .105
itemheight_reg = .19
itemheight = .12
titlestyle = {"weight": "bold", "size": 7}
metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity']
graph_metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity', 'meancor', 'varcor', 'kurtcor']
all_metrics = ["meanar1", "lmbda", "floor", "meancor", "varcor", "kurtcor", 'assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity']
regressed_out = ["regressed_meanar1", "movement"]
fplabels = ["ar1", "cm", "chance", "gbc", "vbc", "kbc", "degree", "centrality"]
nodal_metrics = ["mean", "var", "kurt", "degree", "centrality"]
nodal_metrics_ar1 = ["ar1", "mean", "var", "kurt", "degree", "centrality"]

if False: # we already know this is 0/10000 for both datasets
    hcp = cdatasets.HCP()
    hcpar1 = np.mean(hcp.get_ar1s(), axis=0)
    corr_lsd = scipy.stats.spearmanr(hcpar1, np.mean(LSDall.get_ar1s(), axis=0)).correlation
    corr_psi = scipy.stats.spearmanr(hcpar1, np.mean(PsiAll.get_ar1s(), axis=0)).correlation
    from brainsmash.mapgen.base import Base as BrainSmashGen
    if util.plock("surrogates.pkl"):
        generator = BrainSmashGen(np.mean(hcp.get_ar1s(), axis=0), hcp.get_dists())
        util.psave("surrogates.pkl", generator(n=10000))
    nulls = util.pload("surrogates.pkl")
    nulls_corr_lsd = [scipy.stats.spearmanr(np.mean(LSDall.get_ar1s(), axis=0), nulls[i]).correlation for i in range(0, len(nulls))]
    n_greater_lsd = np.sum(np.asarray(nulls_corr_lsd)>corr_lsd)
    nulls_corr_psi = [scipy.stats.spearmanr(np.mean(PsiAll.get_ar1s(), axis=0), nulls[i]).correlation for i in range(0, len(nulls))]
    n_greater_psi = np.sum(np.asarray(nulls_corr_psi)>corr_psi)
    print(f"LSD: p = {n_greater_lsd}/{len(nulls_corr_lsd)}")
    print(f"Psilocybin: p = {n_greater_psi}/{len(nulls_corr_psi)}")


exps = ["Control", "LSD", "LSD+Ket"]
tps = ["early", "late"]
lsd = {exp : {tp : cdatasets.LSD(exp, tp, gsr=GSR) for tp in tps} for exp in exps}

# Timepoints: 20 min, 40 min, 70 min
pexps = ["Control", "Psilocybin"]
ptps = ["early", "middle", "late"]
psi = {exp : {tp : cdatasets.Psilocybin(exp, "middle" if tp == "early" else tp, gsr=GSR) for tp in ptps} for exp in pexps}

#LSDall = cdatasets.Join([cdatasets.LSD(exp, time, gsr=GSR) for exp in ["Control", "LSD", "LSD+Ket"] for time in ["early", "late"]], name="LSDall")
LSDall = cdatasets.Join([cdatasets.LSD(exp, time, gsr=GSR) for exp in ["LSD"] for time in ["early", "late"]], name="LSDall")
#PsiAll = cdatasets.Join([cdatasets.Psilocybin(exp, time, gsr=GSR) for exp in ["Control", "Psilocybin"] for time in ["early", "middle", "late"]], name="PsiAll")
PsiAll = cdatasets.Join([cdatasets.Psilocybin(exp, time, gsr=GSR) for exp in ["Psilocybin"] for time in ["middle", "late"]], name="PsiAll")
hcp = cdatasets.HCP()

LSDctrl = cdatasets.Join([cdatasets.LSD(exp, time, gsr=GSR) for exp in ["Control"] for time in ["early", "late"]], name="LSDctrl")
Psictrl = cdatasets.Join([cdatasets.Psilocybin(exp, time, gsr=GSR) for exp in ["Control"] for time in ["middle", "late"]], name="Psictrl")
LSDket = cdatasets.Join([cdatasets.LSD(exp, time, gsr=GSR) for exp in ["LSD+Ket"] for time in ["early", "late"]], name="LSDket")

c = Canvas(7.2, 6.8, "in")
c.set_font("Nimbus Sans", ticksize=5, size=6)

c.add_grid([met+"_box" for met in all_metrics[0:3]], 1, Point(.4, 2.6, "in"), Point(3.7, 3.2, "in"), size=Vector(.85, 0.6, "in"))
c.add_grid([met+"_box" for met in all_metrics[3:6]]+[None,None,None]+[met+"_box" for met in all_metrics[6:]], 2, Point(.4, .2, "in"), Point(7.1, 1.95, "in"), size=Vector(.85, 0.6, "in"))
c.add_grid([met+"_box" for met in regressed_out], 1, Point(4.2, 2.6, "in"), Point(6.3, 3.2, "in"), size=Vector(.85, .6, "in"))

brain_img_height = Vector(0, 0.80, "in")
brain_img_loc = {}
for sfx,ROW in [("_LSDall", 5.2), ("_PsiAll", 4.1)]:
    c.add_axis("minigrid"+sfx, Point(.6, ROW, "in")+Vector(0, .4, "cm"), Point(.6+8*BOXSIZE, ROW+8*BOXSIZE, "in")+Vector(0, .4, "cm"))
    c.add_axis("grid"+sfx, Point(2.2, ROW, "in")+Vector(0, .4, "cm"), Point(2.2+BOXSIZE*5, ROW+.15+BOXSIZE*9, "in"))
    brain_img_loc[sfx] = Point(2.9, ROW, "in")+Vector(0, .4, "cm")
    c.add_axis("hcpar1"+sfx, Point(4.65, ROW, "in")+Vector(0, .4, "cm"), Point(5.45, ROW+.8, "in"))
    c.add_axis("ar1_corr_boxplots"+sfx, Point(6.1, ROW, "in")+Vector(0, .4, "cm"), Point(7.1, ROW+itemheight_reg*len(nodal_metrics), "in"))



for data, sfx in [(LSDall, "_LSDall"),
                  (PsiAll, "_PsiAll")]:
    
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
            
            
            
    ax = c.ax("minigrid"+sfx)
    VSCALE = (0, 1)
    ax.imshow(grid.T[1:,:-1], vmin=VSCALE[0], vmax=VSCALE[1], cmap="Blues", aspect='auto')
    ax.axis('off')
    if sfx == "_PsiAll":
        c.add_colorbar("minigrid_colorbar"+sfx, Point(0, 0, "axis_minigrid"+sfx)-Vector(0, .4, "cm"), Point(1, 0, "axis_minigrid"+sfx)-Vector(0, .2, "cm"), cmap="Blues", bounds=VSCALE)
        c.add_text("Absolute Spearman correlation", Point(.5, 0, "axis_minigrid"+sfx)-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")
    
    for i in range(0, len(graph_metrics)-1):
        c.add_text(short_names_for_stuff[graph_metrics[i]], Point(i, -.7+i, "minigrid"+sfx), rotation=0, horizontalalignment="left", verticalalignment="bottom", size=5)
        
    for i in range(0, len(graph_metrics)-1):
        c.add_text(short_names_for_stuff[graph_metrics[i+1]], Point(-.7, i, "minigrid"+sfx), horizontalalignment="right", verticalalignment="center", size=5)
        
    for i in range(0, len(graph_metrics)-1):
        for j in range(i, len(graph_metrics)-1):
            c.add_text(labels[i][j+1], Point(i, j+.3, "minigrid"+sfx), horizontalalignment="center", verticalalignment="center", size=10)
            
            
    ax = c.ax("hcpar1"+sfx)
    simplescatter(np.mean(hcp.get_ar1s(), axis=0), np.mean((LSDctrl if "LSD" in sfx else Psictrl).get_ar1s(), axis=0), ax=ax, diag=True, s=1, c=(.6, .6, .6), alpha=.8)
    #if "LSD" in sfx:
    #    simplescatter(np.mean(hcp.get_ar1s(), axis=0), np.mean(LSDket.get_ar1s(), axis=0), ax=ax, diag=True, s=.2, c='g')
    simplescatter(np.mean(hcp.get_ar1s(), axis=0), np.mean(data.get_ar1s(), axis=0), ax=ax, diag=True, s=1, c='k', alpha=.5)
    ax.set_xlim(.01, .99)
    if sfx == "_PsiAll":
        ax.set_xlabel("HCP regional TA-$\\Delta_1$")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(names_for_stuff[sfx[1:]]+" regional TA-$\\Delta_1$")
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
    if sfx == "_camcan":
        ts_metrics[0] = "loglmbda"
    
    graph_metric_values = data.get_metrics()
    graph_metric_values.update(data.get_cmstats())
    
    grid = np.zeros((len(ts_metrics), len(graph_metrics)))*np.nan
    labels = grid.tolist()
    for i in [0, 1, 3]:
        for j in range(0, len(graph_metrics)):
            spear = scipy.stats.spearmanr(graph_metric_values[graph_metrics[j]], ts_metrics_vals[ts_metrics[i]])
            if spear.pvalue < .05/27:
                grid[i,j] = np.abs(spear.correlation)
            else:
                grid[i,j] = np.abs(spear.correlation)
            #labels[i][j] = f"{spear.correlation:.2f}"
            labels[i][j] = "**" if spear.pvalue < .01/27 else "*" if spear.pvalue < .05/27 else ""
            
    graph_metric_values.update(ts_metrics_vals)
    df = pandas.DataFrame(graph_metric_values)
    for j in range(0, len(graph_metrics)):
        predicted = []
        for lo in df.index:
            predicted.append(smf.ols(f"{graph_metrics[j]} ~ {ts_metrics[0]} + floor", data=df.drop(lo)).fit().predict(df.loc[[lo]]))
        spear = scipy.stats.spearmanr(predicted, df[graph_metrics[j]])
        print(graph_metrics[j], spear)
        grid[2,j] = spear.correlation
        labels[2][j] = "**" if spear.pvalue < .01/18 else "*" if spear.pvalue < .05/18 else ""
        predicted = []
        for lo in df.index:
            predicted.append(smf.ols(f"{graph_metrics[j]} ~ meanar1 + {ts_metrics[0]} + floor", data=df).fit().predict(df.loc[[lo]]))
        spear = scipy.stats.spearmanr(predicted, df[graph_metrics[j]])
        print(graph_metrics[j], spear)
        grid[4,j] = spear.correlation
        labels[4][j] = "**" if spear.pvalue < .01/18 else "*" if spear.pvalue < .05/18 else ""
        
    ax = c.ax("grid"+sfx)
    VSCALE = (0, .7)
    ax.imshow(grid.T, vmin=VSCALE[0], vmax=VSCALE[1], cmap="Blues", aspect='auto')
    ax.axis('off')
    if sfx == "_PsiAll":
        c.add_colorbar("grid_colorbar"+sfx, Point(0, 0, "axis_grid"+sfx)-Vector(0, .4, "cm"), Point(1, 0, "axis_grid"+sfx)-Vector(0, .2, "cm"), cmap="Blues", bounds=VSCALE)
        c.add_text("Absolute Spearman correlation", Point(.5, 0, "axis_grid"+sfx)-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")
    
    if sfx == "_LSDall":
        for i in range(0, len(ts_metrics)):
            c.add_text(short_names_for_stuff[ts_metrics[i]], Point(i, -.7, "grid"+sfx), horizontalalignment="left", verticalalignment="bottom", size=5, rotation=30)
        
    for i in range(0, len(graph_metrics)):
        c.add_text(short_names_for_stuff[graph_metrics[i]], Point(-.7, i, "grid"+sfx), horizontalalignment="right", verticalalignment="center", size=5)
        
    for i in range(0, len(ts_metrics)):
        for j in range(0, len(graph_metrics)):
            c.add_text(labels[i][j], Point(i, j+.3, "grid"+sfx), horizontalalignment="center", verticalalignment="center", size=10)
            
            
    
    
    #################### Degree vs AR1 ####################
    
    
    ax = c.ax("ar1_corr_boxplots"+sfx)
    ax.cla()
    all_nodal_metrics = data.get_nodal_metrics()
    all_nodal_metrics.update(data.get_nodal_cmstats())
    ordered_corrs = [[scipy.stats.spearmanr(data.get_ar1s()[i], all_nodal_metrics[k][i]).correlation for i in range(0, data.N_subjects())] for k in nodal_metrics]
    # colors = [(.3, .3, .3)]*3+["k"]*2
    colors = ["k"]*5
    for col in set(colors):
        ax.boxplot([oc for co,oc in zip(colors,ordered_corrs) if co == col], positions=[i+1 for i,co in enumerate(colors) if co==col], medianprops={"color": col}, boxprops={"color": col}, whiskerprops={"color": col}, capprops={"color": col}, vert=False, showfliers=False, widths=.5)
    sns.despine(ax=ax, left=True)
    ax.axvline(0, c='k')
    ax.set_yticks([])
    ax.set_xlim(-.6, .9)
    if sfx == "_PsiAll":
        ax.set_xlabel("Spearman correlation")
    else:
        ax.set_xticklabels([])
    ax.set_ylim(.5, len(nodal_metrics)+.5)
    ax.invert_yaxis()
    for i in range(0, len(nodal_metrics)):
        c.add_text(short_names_for_stuff[nodal_metrics[i]], Point(-.05, i+1, ("axis_ar1_corr_boxplots"+sfx, "ar1_corr_boxplots"+sfx)), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)
    
    #################### Plot mean AR1 on the brain ####################
    
    mean_ar1s = np.mean(data.get_ar1s(), axis=0)
    VRANGE = (.2, .6)
    fn = f"_cache_brain_ar1_{data.name}_{int(VRANGE[0]*1000)}{int(VRANGE[1]*1000)}{'-nogsr' if GSR == False else ''}.png"
    if not os.path.exists(fn): # No need to regenerate if it exists
        wbplot.pscalar(fn, mean_ar1s, vrange=VRANGE, cmap='viridis', transparent=True)
        
    c.add_image(fn, brain_img_loc[sfx], height=brain_img_height, horizontalalignment="left", verticalalignment="bottom", unitname="brainimage"+sfx)
    if sfx == "_PsiAll":
        c.add_colorbar("brainimage_colorbar3"+sfx, Point(.2, 0, "brainimage"+sfx)+Vector(0, -.4, "cm"), Point(.8, 0, "brainimage"+sfx)+Vector(0, -.2, "cm"), cmap="viridis", bounds=VRANGE)
        c.add_text("TA-$\\Delta_1$", Point(.5, 0, "brainimage"+sfx)+Vector(0, -.9, "cm"), horizontalalignment="center", va="top")

ROW1_TITLEa = Point(0, 6.7, "in")
ROW1_TITLEb = Point(0, 6.4, "in")
c.add_text("Mean regional TA-$\\Delta_1$", Point(.5, 0, "brainimage_LSDall")>>ROW1_TITLEb, ha="center", va="top", **titlestyle)
c.add_text("Correlation with graph metrics", Point(.5, 0, "axis_grid_LSDall")>>ROW1_TITLEa, ha="center", va="top", **titlestyle)
c.add_text("Correlation among\ngraph metrics", Point(.5, 0, "axis_minigrid_LSDall")>>ROW1_TITLEa, ha="center", va="top", **titlestyle)
c.add_text("Correlation with\nregional TA-$\\Delta_1$", Point(.5, 1, "axis_ar1_corr_boxplots_LSDall")>>ROW1_TITLEb, ha="center", va="top", **titlestyle)
c.add_text("Correlation of drug\nvs HCP regional TA-$\\Delta_1$", Point(.5, 1, "hcpar1_LSDall")>>ROW1_TITLEb, ha="center", va="top", **titlestyle)

c.add_text("LSD", Point(.1, .5, ("in", "axis_minigrid_LSDall")), rotation=90, weight="bold", size=8, bbox=dict(facecolor='none', edgecolor=PAL[0], boxstyle="round"))
c.add_text("Psilocybin", Point(.1, .5, ("in", "axis_minigrid_PsiAll")), rotation=90, weight="bold", size=8, bbox=dict(facecolor='none', edgecolor=PAL[1], boxstyle="round"))









from itertools import repeat
import wbplot
_df = []
for timepoint in ["early", "late"]:
    for exp in ["LSD", "Psilocybin", "LSD+Ket"]:
        if "LSD" in exp:
            dsc = cdatasets.LSD("Control", timepoint, gsr=GSR)
            ds = cdatasets.LSD(exp, timepoint, gsr=GSR)
        else:
            dsc = cdatasets.Psilocybin("Control", timepoint if timepoint == "late" else "middle", gsr=GSR)
            ds = cdatasets.Psilocybin("Psilocybin", timepoint if timepoint == "late" else "middle", gsr=GSR)
        _df.extend(zip(np.mean(ds.get_ar1s(), axis=1)-np.mean(dsc.get_ar1s(), axis=1), repeat("meanar1"), repeat(timepoint), repeat(exp), range(0, 100000)))
        _df.extend(zip(ds.get_lmbda()-np.asarray(dsc.get_lmbda()), repeat("lmbda"), repeat(timepoint), repeat(exp), range(0, 100000)))
        _df.extend(zip(ds.get_floor()-np.asarray(dsc.get_floor()), repeat("floor"), repeat(timepoint), repeat(exp), range(0, 100000)))
        if util.plock(f"{timepoint}-{exp.replace('+', '')}{'-nogsr' if GSR == False else ''}-ar1diff.png"):
            wbplot.pscalar(f"{timepoint}-{exp.replace('+', '')}{'-nogsr' if GSR == False else ''}-ar1diff.png", np.mean(ds.get_ar1s(), axis=0)-np.mean(dsc.get_ar1s(), axis=0), vrange=(-.3, .3), cmap="seismic")
        for met in metrics:
            _df.extend(zip(ds.get_metrics()[met]-np.asarray(dsc.get_metrics()[met]), repeat(met), repeat(timepoint), repeat(exp), range(0, 100000)))
        for met in ["meancor", "varcor", "kurtcor"]:
            _df.extend(zip(ds.get_cmstats()[met]-np.asarray(dsc.get_cmstats()[met]), repeat(met), repeat(timepoint), repeat(exp), range(0, 100000)))
        #Regress out movement and recompute meanar1
        _df.extend(zip(ds.get_movement()-np.asarray(dsc.get_movement()), repeat("movement"), repeat(timepoint), repeat(exp), range(0, 100000)))
        resid = sm.OLS(np.concatenate([np.mean(ds.get_ar1s(), axis=1), np.mean(dsc.get_ar1s(), axis=1)]), sm.add_constant(np.concatenate([ds.get_movement(), dsc.get_movement()]))).fit().resid
        _df.extend(zip(resid[0:len(resid)//2]-resid[len(resid)//2:], repeat("regressed_meanar1"), repeat(timepoint), repeat(exp), range(0, 100000)))



df = pandas.DataFrame(_df, columns=["value", "type", "timepoint", "exp", "subject"])



for met in all_metrics+regressed_out:
#for met in ["meanar1"]:
    axname = met+"_box"
    ax = c.ax(axname)
    drug_order = ['LSD', 'Psilocybin', 'LSD+Ket']
    plotdata = df.query("type==@met").groupby(['timepoint', 'exp'])['value'].apply(list).reindex(drug_order, level='exp').reset_index()
    positions = [1, 1.5, 2, 3, 3.5, 4]
    colors = PAL+PAL
    for i in range(0, len(positions)):
        bplot = ax.boxplot(list(plotdata['value'])[i:(i+1)], medianprops={"color": colors[i]}, boxprops={"color": colors[i]}, whiskerprops={"color": colors[i]}, capprops={"color": colors[i]}, vert=True, showfliers=False, positions=positions[i:(i+1)], widths=.35)
    for i,dat in enumerate(list(plotdata['value'])):
        test = scipy.stats.wilcoxon(dat)
        text = "**" if test.pvalue<.01 else "*" if test.pvalue<.05 else ""
        if text:
            c.add_text(text, Point(positions[i], 1, (axname, "axis_"+axname)))
    #sns.boxplot(data=df.query("type==@met"), y="value", x="timepoint", hue="exp", showfliers=False, ax=ax)
    #ax.get_legend().remove()
    ax.axhline(0, c='k')
    sns.despine(bottom=True, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    c.add_text("Early", Point(positions[1], 0, (axname, "axis_"+axname))+Vector(0, -.1, "in"))
    c.add_text("Late", Point(positions[4], 0, (axname, "axis_"+axname))+Vector(0, -.1, "in"))
    if met in names_for_stuff.keys():
        ax.set_title(names_for_stuff[met])

c.add_legend(Point(6.5, 3.0, "in"), [(t, {"color": c, "linestyle": "none", "marker": "s"}) for t,c in zip(drug_order,PAL)],
             line_spacing=Vector(0, 1.3, "Msize"), sym_width=Vector(1, 0, "Msize"))


c.add_text("Change in graph metrics on drug", Point(.5, 1.5, "axis_varcor_box"), ha="center", va="bottom", **titlestyle)
c.add_text("Change in subject SA and TA on drug", Point(.5, 1.5, "axis_lmbda_box"), ha="center", va="bottom", **titlestyle)

c.ax("regressed_meanar1_box").set_title("Change in global TA-$\\Delta_1$\non drug regressing\nout motion")
c.ax("movement_box").set_title("Change in subject \nmotion on drug")

c.add_figure_labels([("a", "minigrid_LSDall"),
                     ("b", "grid_LSDall"),
                     ("c", "brainimage_LSDall", Vector(.13, .1, "in")),
                     ("d", "hcpar1_LSDall", Vector(-.1, .2, "in")),
                     ("e", "ar1_corr_boxplots_LSDall"),
                     ("f", "meanar1_box", Vector(0, 0, "in")),
                     ("g", "lmbda_box", Vector(0, 0, "in")),
                     ("j", "meancor_box", Vector(0, .15, "in")),
                     ("h", "regressed_meanar1_box", Vector(0, .15, "in")),
                     ("i", "movement_box", Vector(0, .15, "in")),
], size=8)

c.save("figure5-supp.pdf")

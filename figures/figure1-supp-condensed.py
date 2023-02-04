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
from figurelib import get_cm_lmbda_params, corplot, icc_full, simplescatter, fingerprint, names_for_stuff, short_names_for_stuff, plot_on_volume_parcellation, trt_bootstrap_icc

# Set this to 0 to make the real figure, or 1 or 2 to make the
# supplements

FILENAME = "figure1-supp-condensed.pdf"

import cdatasets
data = cdatasets.HCP1200()
data_rep = cdatasets.HCP1200(3)
datatrt = cdatasets.HCP1200KindaLikeTRT()
#datatrt = cdatasets.TRT()

titlestyle = {"weight": "bold", "size": 7}
metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity']
graph_metrics = ['assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity', 'meancor', 'varcor', 'kurtcor']
all_metrics = ["lmbda", "floor", "meanar1", "meancor", "varcor", "kurtcor", 'assort', 'cluster', 'lefficiency', 'gefficiency', 'modularity', 'transitivity']
fplabels = ["ar1", "cm", "chance", "gbc", "vbc", "kbc", "degree", "centrality"]
nodal_metrics = ["mean", "var", "kurt", "degree", "centrality"]
nodal_metrics_ar1 = ["ar1", "mean", "var", "kurt", "degree", "centrality"]



#################### Set up the canvas ####################


c = Canvas(7.2, 9.6, "in") # max 9.7
c.set_font("Nimbus Sans", ticksize=5, size=6)
# c.debug_grid(Vector(1, 1, "in"))
# c.debug_grid(Vector(.5, .5, "in"), linewidth=.5)


BOXSIZE = .12
itemheight_reli = .115
itemheight_reg = .16
itemheight = .13
brain_img_loc = {}
for sfx,ROW in [("_camcan", .4)]:
    c.add_axis("minigrid"+sfx, Point(.4, ROW, "in")+Vector(0, .4, "cm"), Point(.4+8*BOXSIZE, ROW+8*BOXSIZE, "in")+Vector(0, .4, "cm"))
    c.add_axis("grid"+sfx, Point(2.2, ROW, "in")+Vector(0, .4, "cm"), Point(2.2+BOXSIZE*5, ROW+.15+BOXSIZE*9, "in"))
    brain_img_loc[sfx] = Point(3.2, ROW, "in")+Vector(0, .4, "cm")
    c.add_axis("ar1_corr_boxplots"+sfx, Point(5.2, ROW, "in"), Point(6.2, ROW+itemheight_reg*len(nodal_metrics), "in"))
    c.add_grid(["nodal_var_ar1_mean"+sfx, "nodal_degree_ar1_mean"+sfx], 2, Point(6.7, ROW+.1, "in"), Point(7.1, ROW+1.1, "in"), size=Vector(.4, .4, "in"))


for sfx,ROW2,ROW3 in [("_trt", 4.1, 2.5), ("_hcpgsr", 7.9, 6.3)]:
    # ROW2
    c.add_axis("minigrid"+sfx, Point(.4, ROW2, "in")+Vector(0, .4, "cm"), Point(.4+8*BOXSIZE, ROW2+8*BOXSIZE, "in")+Vector(0, .4, "cm"))
    c.add_axis("reliabilities"+sfx, Point(2.6, ROW2, "in"), Point(3.15, ROW2+itemheight_reli*len(all_metrics), "in"))
    c.add_grid(["lmbda"+sfx, "floor"+sfx, "meanar1"+sfx], 3, Point(1, .05, "axis_reliabilities"+sfx)+Vector(.45,0,"in"), Point(1, .95, "axis_reliabilities"+sfx)+Vector(.80,0,"in"),size=Vector(.35, .35, "in"))
    c.add_axis("grid"+sfx, Point(4.6, ROW2, "in")+Vector(0, .4, "cm"), Point(4.6+BOXSIZE*5, ROW2+.15+BOXSIZE*9, "in"))
    brain_img_loc[sfx] = Point(5.6, ROW2, "in")+Vector(0, .4, "cm")
    brain_img_height = Vector(0, 1.00, "in")
    
    # ROW3
    c.add_axis("region_reliability"+sfx, Point(0.5, ROW3, "in"), Point(1.7, ROW3+itemheight_reg*len(nodal_metrics_ar1), "in"))
    c.add_axis("fingerprint"+sfx, Point(2.2, ROW3, "in"), Point(3.1, ROW3+itemheight*len(fplabels), "in"))
    c.add_axis("ar1_corr_boxplots"+sfx, Point(3.7, ROW3, "in"), Point(4.7, ROW3+itemheight_reg*len(nodal_metrics), "in"))
    c.add_grid(["nodal_var_ar1_mean"+sfx, "nodal_degree_ar1_mean"+sfx], 2, Point(5.2, ROW3+.1, "in"), Point(5.6, ROW3+1.1, "in"), size=Vector(.4, .4, "in"))
    c.add_axis("reliability_vs_ar1"+sfx, Point(6.2, ROW3, "in"), Point(6.9, ROW3+.7, "in"))
    # c.add_axis("reliability_hist", Point(0, 1, "axis_reliability_vs_ar1"), Point(1, 1, "axis_reliability_vs_ar1")+Vector(0, .2, "in"))


for data, sfx in [(cdatasets.HCP1200(0, gsr=True), "_hcpgsr"), (cdatasets.TRT(), "_trt"), (cdatasets.CamCanFiltered(), "_camcan")]:
    
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
    c.add_colorbar("minigrid_colorbar"+sfx, Point(0, 0, "axis_minigrid"+sfx)-Vector(0, .4, "cm"), Point(1, 0, "axis_minigrid"+sfx)-Vector(0, .2, "cm"), cmap="Blues", bounds=VSCALE)
    c.add_text("Absolute Spearman correlation", Point(.5, 0, "axis_minigrid"+sfx)-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")
    c.add_text("Correlation among graph metrics", Point(.5, 0, "axis_minigrid"+sfx)+Vector(0, 1.2, "in"), ha="center", **titlestyle)
    
    for i in range(0, len(graph_metrics)-1):
        c.add_text(short_names_for_stuff[graph_metrics[i]], Point(i, -.7+i, "minigrid"+sfx), rotation=0, horizontalalignment="left", verticalalignment="bottom", size=5)
        
    for i in range(0, len(graph_metrics)-1):
        c.add_text(short_names_for_stuff[graph_metrics[i+1]], Point(-.7, i, "minigrid"+sfx), horizontalalignment="right", verticalalignment="center", size=5)
        
    for i in range(0, len(graph_metrics)-1):
        for j in range(i, len(graph_metrics)-1):
            c.add_text(labels[i][j+1], Point(i, j+.3, "minigrid"+sfx), horizontalalignment="center", verticalalignment="center", size=10)
            
            
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
        predicted = smf.ols(f"{graph_metrics[j]} ~ {ts_metrics[0]} + floor", data=df_train).fit().predict(df_test)
        spear = scipy.stats.spearmanr(predicted, df_test[graph_metrics[j]])
        print(graph_metrics[j], spear)
        grid[2,j] = spear.correlation
        labels[2][j] = "**" if spear.pvalue < .01/18 else "*" if spear.pvalue < .05/18 else ""
        predicted = smf.ols(f"{graph_metrics[j]} ~ meanar1 + {ts_metrics[0]} + floor", data=df_train).fit().predict(df_test)
        spear = scipy.stats.spearmanr(predicted, df_test[graph_metrics[j]])
        print(graph_metrics[j], spear)
        grid[4,j] = spear.correlation
        labels[4][j] = "**" if spear.pvalue < .01/18 else "*" if spear.pvalue < .05/18 else ""
        
    ax = c.ax("grid"+sfx)
    VSCALE = (0, np.ceil(np.max(grid)*10)/10)
    ax.imshow(grid.T, vmin=VSCALE[0], vmax=VSCALE[1], cmap="Blues", aspect='auto')
    ax.axis('off')
    c.add_colorbar("grid_colorbar"+sfx, Point(0, 0, "axis_grid"+sfx)-Vector(0, .4, "cm"), Point(1, 0, "axis_grid"+sfx)-Vector(0, .2, "cm"), cmap="Blues", bounds=VSCALE)
    c.add_text("Absolute Spearman correlation", Point(.5, 0, "axis_grid"+sfx)-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")
    
    for i in range(0, len(ts_metrics)):
        c.add_text(short_names_for_stuff[ts_metrics[i]], Point(i, -.7, "grid"+sfx), horizontalalignment="left", verticalalignment="bottom", size=5, rotation=30)
        
    for i in range(0, len(graph_metrics)):
        c.add_text(short_names_for_stuff[graph_metrics[i]], Point(-.7, i, "grid"+sfx), horizontalalignment="right", verticalalignment="center", size=5)
        
    for i in range(0, len(ts_metrics)):
        for j in range(0, len(graph_metrics)):
            c.add_text(labels[i][j], Point(i, j+.3, "grid"+sfx), horizontalalignment="center", verticalalignment="center", size=10)
            
            
    c.add_text("Correlation with graph metrics", Point(.5, 0, "axis_grid"+sfx)+Vector(0, 1.4, "in"), ha="center", **titlestyle)
    
    
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
    ax.set_xlabel("Spearman correlation")
    ax.set_ylim(.5, len(nodal_metrics)+.5)
    ax.invert_yaxis()
    c.add_text("Correlation with TA-$\\Delta_1$", Point(.5, 1, "axis_ar1_corr_boxplots"+sfx)+Vector(0,.2, "cm"), ha="center", **titlestyle)
    for i in range(0, len(nodal_metrics)):
        c.add_text(short_names_for_stuff[nodal_metrics[i]], Point(-.05, i+1, ("axis_ar1_corr_boxplots"+sfx, "ar1_corr_boxplots"+sfx)), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)
    
    for nm in ["var", "degree"]:
        axname = f"nodal_{nm}_ar1_mean"+sfx
        ax = c.ax(axname)
        ax.cla()
        mean_nm = np.mean(all_nodal_metrics[nm], axis=0)
        simplescatter(np.mean(data.get_ar1s(), axis=0), mean_nm, ax=ax, s=.45, alpha=.8, c='k', diag=False, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        c.add_text(short_names_for_stuff["ar1"], Point(.5, -.2, "axis_"+axname), ha="center", va="center", size=5)
        c.add_text(short_names_for_stuff[nm], Point(-.2, .5, "axis_"+axname), ha="center", va="center", size=5, rotation=90)
        ind = nodal_metrics.index(nm)
        arrowfrm = Point(1, ind+1, "ar1_corr_boxplots"+sfx)
        arrowto = Point(-.35, .5, f"axis_nodal_{nm}_ar1_mean"+sfx)
        c.add_arrow(arrowfrm, arrowto, lw=1)
        
        
        
    #################### Plot mean AR1 on the brain ####################
    
    mean_ar1s = np.mean(data.get_ar1s(), axis=0)
    if "hcp" in sfx:
        VRANGE = (0, 1)
    elif "cam" in sfx:
        VRANGE = (.8, 1)
    elif "trt" in sfx:
        VRANGE = (.7, 1)
    fn = f"_cache_brain_ar1_{data.name}_{int(VRANGE[0]*1000)}{int(VRANGE[1]*1000)}.png"
    if not os.path.exists(fn): # No need to regenerate if it exists
        if data.name.lower().startswith("hcp"):
            wbplot.pscalar(fn, mean_ar1s, vrange=VRANGE, cmap='viridis', transparent=True)
        elif "cam" in data.name.lower():
            plot_on_volume_parcellation(mean_ar1s, fn, vlim=VRANGE, parcellation="aal")
        elif data.name.lower().startswith("trt"):
            plot_on_volume_parcellation(mean_ar1s, fn, vlim=VRANGE, parcellation="shen")
        
    c.add_image(fn, brain_img_loc[sfx], height=brain_img_height, horizontalalignment="left", verticalalignment="bottom", unitname="brainimage"+sfx)
    c.add_colorbar("brainimage_colorbar3"+sfx, Point(.2, 0, "brainimage"+sfx)+Vector(0, -.4, "cm"), Point(.8, 0, "brainimage"+sfx)+Vector(0, -.2, "cm"), cmap="viridis", bounds=VRANGE)
    c.add_text("Mean regional TA-$\\Delta_1$", Point(.5, 0, "brainimage"+sfx)+Vector(0, -.9, "cm"), horizontalalignment="center", va="top")
    c.add_text("Mean regional TA-$\\Delta_1$", Point(.5, 0, "brainimage"+sfx)+Vector(0, 1.25, "in")-Vector(0, .4, "cm"), ha="center", **titlestyle)
    
    
    
    
    


for datatrt,data,data_rep,sfx in [(cdatasets.HCP1200KindaLikeTRT(gsr=True), cdatasets.HCP1200(0, gsr=True), cdatasets.HCP1200(3, gsr=True), "_hcpgsr"), (cdatasets.TRT(), cdatasets.TRTKindaLikeHCP(0), cdatasets.TRTKindaLikeHCP(5), "_trt")]:
    
    #################### Test retest ####################
    
    ax = c.ax("reliabilities"+sfx)
    
    
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
        
    ax.barh(range(0, len(all_metrics)), [reliabilities[k][0] for k in all_metrics], xerr=np.asarray([np.abs(np.asarray(reliabilities[k][1])-reliabilities[k][0]) for k in all_metrics]).T, color=['r' if k in ['meanar1', 'lmbda', 'floor'] else (.5, .5, .5) if 'cor' in k else 'k' for k in all_metrics], clip_on=False, height=.8)
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
        rs += "#" if lt_lmbda_sim < .01 else "+" if lt_lmbda_sim < .05 else "-"
        rs += "#" if lt_floor_sim < .01 else "+" if lt_floor_sim < .05 else "-"
        rs += "#" if lt_ar1_sim < .01 else "+" if lt_ar1_sim < .05 else "-"
        c.add_text(rs, Point(reliabilities[all_metrics[i]][1][1], i+.4, "reliabilities"+sfx)+Vector(.35, .1, "cm"), horizontalalignment="center", verticalalignment="center", size=5, font="Noto Mono", weight="regular")
        
    for i in range(0, len(all_metrics)):
        c.add_text(names_for_stuff[all_metrics[i]], Point(0, i, "reliabilities"+sfx)+Vector(-.1, 0, "cm"), horizontalalignment="right", verticalalignment="center", size=5)
        
    c.add_text("Test-retest reliability", Point(0, 0, "axis_reliabilities"+sfx)+Vector(0,1.3, "in")+Vector(0, .4, "cm"), ha="left", **titlestyle)
    ax.set_xlabel("Reliability (ICC)")
    
    arrowx = max([reliabilities[v][0] for v in ["lmbda", "floor", "meanar1"]])
    if "trt" in sfx:
        arrowx += .15
    ax = c.ax("lmbda"+sfx)
    ax.cla()
    alph = 1 if "trt" in sfx else .2
    simplescatter(data.get_lmbda(), data_rep.get_lmbda(), s=(.9 if "trt" in sfx else .4), c='k', alpha=alph, rasterized=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ind = all_metrics.index("lmbda")
    arrowfrm = Point(arrowx, ind, "reliabilities"+sfx)
    arrowto = Point(-.1, .5, "axis_lmbda"+sfx)
    c.add_arrow(arrowfrm, arrowto, lw=1)
    ax = c.ax("floor"+sfx)
    ax.cla()
    simplescatter(data.get_floor(), data_rep.get_floor(), s=(.9 if "trt" in sfx else .4), c='k', alpha=alph, rasterized=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ind = all_metrics.index("floor")
    arrowfrm = Point(arrowx, ind, "reliabilities"+sfx)
    arrowto = Point(-.1, .5, "axis_floor"+sfx)
    c.add_arrow(arrowfrm, arrowto, lw=1)
    
    
    ax = c.ax("meanar1"+sfx)
    ax.cla()
    simplescatter(np.mean(data.get_ar1s(), axis=1), np.mean(data_rep.get_ar1s(), axis=1), s=(.9 if "trt" in sfx else .4), c='k', alpha=alph, rasterized=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ind = all_metrics.index("meanar1")
    arrowfrm = Point(arrowx, ind, "reliabilities"+sfx)
    arrowto = Point(-.1, .5, "axis_meanar1"+sfx)
    c.add_arrow(arrowfrm, arrowto, lw=1)
    
    
    
    #################### Fingerprinting ####################
    
    fn = f"_f1_cache_fingerprint_{datatrt.name}.pkl"
    if os.path.exists(fn):
        fingerprintcache_pair = util.pload(fn)
    else:
        inds = np.triu_indices(datatrt.N_regions(), 1)
        mats_reduced = np.asarray([m[inds] for m in datatrt.get_matrices()])
        subjects = np.asarray(datatrt.get_subject_info()['subject'])
        chance = (len(subjects)/len(set(subjects))-1)/(len(subjects)-1)
        if "trt" in sfx:
            run = datatrt.get_subject_info()['run']
            fingerprintcache_pair = {
                "ar1": [fingerprint(subjects[run==(i+1)], datatrt.get_ar1s()[run==(i+1)]) for i in range(0, len(set(run)))],
                "cm": [fingerprint(subjects[run==(i+1)], mats_reduced[run==(i+1)]) for i in range(0, len(set(run)))],
                "gbc": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['mean'][run==(i+1)]) for i in range(0, len(set(run)))],
                "vbc": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['var'][run==(i+1)]) for i in range(0, len(set(run)))],
                "kbc": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['kurt'][run==(i+1)]) for i in range(0, len(set(run)))],
                "meants": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['ts_mean'][run==(i+1)]) for i in range(0, len(set(run)))],
                "varts": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['ts_var'][run==(i+1)]) for i in range(0, len(set(run)))],
                "kurtts": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_cmstats()['ts_kurt'][run==(i+1)]) for i in range(0, len(set(run)))],
                "lefficiency": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['lefficiency'][run==(i+1)]) for i in range(0, len(set(run)))],
                "cluster": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['cluster'][run==(i+1)]) for i in range(0, len(set(run)))],
                "centrality": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['centrality'][run==(i+1)]) for i in range(0, len(set(run)))],
                "degree": [fingerprint(subjects[run==(i+1)], datatrt.get_nodal_metrics()['degree'][run==(i+1)]) for i in range(0, len(set(run)))],
                "chance": [chance for i in range(0, len(set(run)))],
                }
        if "hcp" in sfx:
            scan = datatrt.get_subject_info()['scan']
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
        rng = np.random.RandomState(0)
        util.psave(fn, fingerprintcache_pair)
        
        
    ax = c.ax("fingerprint"+sfx)
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
    c.add_text("Fingerprinting", Point(.4, 0, "axis_fingerprint"+sfx)+Vector(0,1.15, "in"), ha="center", **titlestyle)
    
    
    for i in reversed(range(0, len(fplabels))):
        if fplabels[i] == "": continue
        c.add_text(names_for_stuff[fplabels[i]], Point(0, i, "fingerprint"+sfx)+Vector(-.1, 0, "cm"), horizontalalignment="right", verticalalignment="center", size=5)
        
        
    #################### Region reliabilities ####################
    
    
    fn = f"_cache_brain_reliability_{datatrt.name}.png.pkl"
    if not os.path.exists(fn): # No need to regenerate if it exists
        region_reliabilities = {}
        region_reliabilities['ar1'] = [icc_full(datatrt.get_subject_info()['subject'], datatrt.get_ar1s()[:,i])[0] for i in range(0, datatrt.N_regions())]
        for met in ['mean', 'var', 'kurt']:
            region_reliabilities[met] = [icc_full(datatrt.get_subject_info()['subject'], datatrt.get_nodal_cmstats()[met][:,i])[0] for i in range(0, datatrt.N_regions())]
        for met in ['degree', 'centrality']:
            region_reliabilities[met] = [icc_full(datatrt.get_subject_info()['subject'], datatrt.get_nodal_metrics()[met][:,i])[0] for i in range(0, datatrt.N_regions())]
        util.psave(fn, region_reliabilities)
    else:
        region_reliabilities = util.pload(fn)
        
        
        
    ax = c.ax("reliability_vs_ar1"+sfx)
    corplot(np.mean(datatrt.get_ar1s(), axis=0), region_reliabilities['ar1'], "Regional TA-$\\Delta_1$ reliability (ICC)", "Mean regional TA-$\\Delta_1$", ax=ax, showr2=False, alpha=.5, markersize=2, rasterized=True)
    c.add_text("TA-$\\Delta_1$ by reliability", Point(.5, 1, "axis_reliability_vs_ar1"+sfx)+Vector(0,.2, "cm"), ha="center", **titlestyle)
    
    
    
    
    ax = c.ax("region_reliability"+sfx)
    ax.cla()
    ordered_reliabilities = [region_reliabilities[k] for k in nodal_metrics_ar1]
    bplot = ax.boxplot(ordered_reliabilities, medianprops={"color": "k"}, vert=False, showfliers=False)
    sns.despine(ax=ax, left=True)
    ax.axvline(0, c='k')
    ax.set_yticks([])
    ax.set_ylim(.5, len(nodal_metrics_ar1)+.5)
    ax.set_xlabel("Reliability (ICC)")
    ax.invert_yaxis()
    c.add_text("Reliability of nodal metrics", Point(.5, 0, "axis_region_reliability"+sfx)+Vector(0, 1.15, "in"), ha="center", **titlestyle)
    bplot['boxes'][0].set_color('r')
    bplot['medians'][0].set_color('r')
    bplot['whiskers'][0].set_color('r')
    bplot['whiskers'][1].set_color('r')
    bplot['caps'][0].set_color('r')
    bplot['caps'][1].set_color('r')
    
    for i in range(0, len(nodal_metrics_ar1)):
        c.add_text(short_names_for_stuff[nodal_metrics_ar1[i]], Point(-.05, i+1, "region_reliability"+sfx), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)



c.add_text("HCP-GSR", Point(.05, 9.5, "in"), size=8, weight="bold", ha="left")
c.add_text("Yale Test-Retest", Point(.05, 5.7, "in"), size=8, weight="bold", ha="left")
c.add_text("Cam-CAN", Point(.05, 2.0, "in"), size=8, weight="bold", ha="left")


allnames = []
for sfx in ["_hcpgsr", "_trt"]:
    for axname in ["minigrid", "reliabilities", "grid", "brainimage", "region_reliability", "fingerprint", "ar1_corr_boxplots", "reliability_vs_ar1"]:
        allnames.append((axname, sfx))

for axname in ["minigrid", "grid", "brainimage", "ar1_corr_boxplots"]:
    allnames.append((axname, "_camcan"))


adjustments = {"minigrid": Vector(0, -.05, "cm"),
               "brainimage": Vector(.3, 0, "cm"),
               "fingerprint": Vector(-.2, 0, "cm"),
               }

c.add_figure_labels([(l,n+s,adjustments.get(n, Vector(0,0,"cm"))) for l,(n,s) in zip("abcdefghijklmnopqrst", allnames)], size=8)



c.save(FILENAME)
#c.show()

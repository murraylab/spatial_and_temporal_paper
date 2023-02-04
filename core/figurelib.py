import numpy as np
import scipy.optimize
import scipy.stats
import pandas
import seaborn as sns
import nipype.algorithms.icc
import nibabel
import pingouin
import pyvista
from canvas import Vector, Point
from config import AAL_PATH, SHEN_PATH
import datasets
import vedo
from util import get_cm_lmbda_params
import PIL
from collections import Counter


def corplot(vals1, vals2, name1, name2, title=None, ax=None, diag=False, color=None, alpha=None, markersize=4, showr2="uv", rasterized=False, negsig=False):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    spear = scipy.stats.spearmanr(vals1, vals2)
    pears = scipy.stats.pearsonr(vals1, vals2)[0]
    #R2 = pears**2 * np.sign(pears)
    sig = "**" if spear.pvalue < .01 else "*" if spear.pvalue < .05 else ""
    if not negsig:
        if spear.correlation < 0: sig = ""
    if showr2 is False:
        r2portion = ""
    elif showr2 == "uv":
        UV = np.sum((np.asarray(vals1)-np.asarray(vals2))**2)/np.sum((np.asarray(vals1)-np.mean(np.asarray(vals1)))**2)
        r2portion = f"\nUV$={UV:0.2f}$" if UV < 100 else f"\nUV$={round(UV)}$"
        if UV < .001:
            r2portion = "\nUV$<.001$"
    elif showr2 == "icc":
        ic = icc(vals1, vals2)
        r2portion = f"\nICC$={ic:0.2f}$"
    elif showr2 == "lin":
        ic = lin(vals1, vals2)
        r2portion = f"\nLin$={ic:0.2f}$"
        if "0.00" in r2portion: # Eliminate negative sign in front of 0
            r2portion = f"\nLin$=0.00$"
        # Significance is weird in Lin's concordance
        # pval = lin_significance(vals1, vals2)
        # if pval < .01:
        #     r2portion += "**"
        # elif pval < .05:
        #     r2portion += "*"
    elif showr2 == "r2":
        r2 = pears**2
        r2portion = f"\n$R^2={r2:0.2f}$"
    ax.text(.7, .3, f"$r_s={spear.correlation:0.2f}${sig}{r2portion}", size=6, transform=ax.transAxes,
            bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0, 0.3), fc=(.95, .95, .95, .7)))
    if color is None:
        color = 'k'
        #color = "r" if sig == "**" else "b" if sig == "*" else "k"
    ax.scatter(vals1, vals2, marker='o', s=markersize, c=color, rasterized=rasterized, alpha=alpha, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    sns.despine(ax=ax)
    if diag:
        ax.plot([-500, 500], [-500, 500], c='k', linewidth=.5)
        axlims = [.9*min(np.min(vals1), np.min(vals2)),
                  1.1*max(np.max(vals1), np.max(vals2))]
        ax.set_xlim(*axlims)
        ax.set_ylim(*axlims)


def corplot_c(vals1, vals2, name1, name2, c, title=None, axname=None, diag=False, color=None, offset=0, shift=Vector(0, 0, "in"), noinfo=False, short=False):
    ax = c.ax(axname)
    spear = scipy.stats.spearmanr(vals1, vals2)
    pears = scipy.stats.pearsonr(vals1, vals2)[0]
    #R2 = pears**2 * np.sign(pears)
    R2 = 1 - np.sum((np.asarray(vals1)-np.asarray(vals2))**2)/np.sum((np.asarray(vals1)-np.mean(np.asarray(vals1)))**2)
    sig = "**" if spear.pvalue < .01 else "*" if spear.pvalue < .05 else ""
    if spear.correlation < 0: sig = ""
    if not noinfo:
        r2portion = f"$R^2={R2:0.2f}$" if DI > -100 else f"$R^2={round(DI)}$"
        if not short:
            c.add_text(f"$r_s={spear.correlation:0.2f}${sig}\n{r2portion}", Point(.95, .8, "axis_"+axname)-(Vector(0, .9, "cm")*offset)+shift, bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0, 0.0), fc=tuple(list(color)+[.3])), horizontalalignment="left", zorder=80)
        else:
            c.add_text(f"$r_s={spear.correlation:0.2f}${sig}\t{r2portion}", Point(.95, .8, "axis_"+axname)-(Vector(0, .3, "cm")*offset)+shift, bbox=dict(boxstyle="round,pad=.1,rounding_size=.3", ec=(0.0, 0.0, 0.0, 0.0), fc=tuple(list(color)+[.3])), horizontalalignment="left", zorder=80, size=5)
    if color is None:
        color = "r" if sig == "**" else "b" if sig == "*" else "k"
    orders = np.random.randint(-50, 50, len(vals1))
    if offset == 0:
        orders += 20
    for v1,v2,o in zip(vals1,vals2,orders):
        ax.scatter([v1], [v2], marker='o', s=4, c=[color], zorder=o)
    ax.set_title(title)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    sns.despine(ax=ax)
    if diag:
        prev_limits = ax.axis()
        ax.plot([-500, 500], [-500, 500], c='k', linewidth=.5)
        rng = max(np.max(vals1)-np.min(vals1), np.max(vals2)-np.min(vals2))
        axlims = [min(np.min(vals1), np.min(vals2))-.1*rng,
                  max(np.max(vals1), np.max(vals2))+.1*rng]
        if offset > 0: # Only expand limits if multiple plots on one axis
            axlims = [min(axlims[0], prev_limits[0]),
                      max(axlims[1], prev_limits[1])]
        ax.set_xlim(*axlims)
        ax.set_ylim(*axlims)

def corplot_sig(vals1, vals2, name1, name2, title=None, ax=None, diag=False, color=None, markersize=2):
    if ax is None:
        ax = plt.gca()
    spear = scipy.stats.spearmanr(vals1, vals2)
    pears = scipy.stats.pearsonr(vals1, vals2)[0]
    #R2 = pears**2 * np.sign(pears)
    #DI = np.sum((np.asarray(vals1)-np.asarray(vals2))**2)/np.sum((np.asarray(vals1)-np.mean(np.asarray(vals1)))**2)
    R2 = 1 - np.sum((np.asarray(vals1)-np.asarray(vals2))**2)/np.sum((np.asarray(vals1)-np.mean(np.asarray(vals1)))**2)
    sig = "**" if spear.pvalue < .01 else "*" if spear.pvalue < .05 else ""
    ax.text(.7, .3, f"$r_s$={spear.correlation:0.2f}{sig}\n$R^2={R2:0.2f}$", size=6, transform=ax.transAxes,
            bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0, 0.3), fc=(.95, .95, .95, .7)))
    if color is None:
        color = 'k'
        if sig.startswith("*"):
            color = "r" if spear.correlation > 0 else "b"
    ax.plot(vals1, vals2, marker='o', linestyle='none', markersize=markersize, c=color)
    ax.set_title(title)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    sns.despine(ax=ax)
    if diag:
        ax.plot([-500, 500], [-500, 500], c='k', linewidth=.5)
        axlims = [.9*min(np.min(vals1), np.min(vals2)),
                  1.1*max(np.max(vals1), np.max(vals2))]
        ax.set_xlim(*axlims)
        ax.set_ylim(*axlims)

def simplescatter(vals1, vals2, ax, diag=True, **kwargs):
    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 0
    ax.scatter(vals1, vals2, **kwargs)
    sns.despine(ax=ax)
    if diag:
        ax.plot([-500, 500], [-500, 500], c='k', linewidth=.5)
        axlims = [.9*min(np.min(vals1), np.min(vals2)),
                  1.1*max(np.max(vals1), np.max(vals2))]
        ax.set_xlim(*axlims)
        ax.set_ylim(*axlims)

# Compute ICC
# def icc(test, retest):
#     test = np.asarray(test)
#     retest = np.asarray(retest)
#     assert len(test.shape) == 1
#     assert test.shape == retest.shape
#     return nipype.algorithms.icc.ICC_rep_anova(np.asarray([test, retest]).T)[0]

# def icc_full(subjects, values):
#     all_vals = np.asarray([[v for si,v in zip(subjects,values) if si==s] for s in set(subjects)])
#     return nipype.algorithms.icc.ICC_rep_anova(all_vals)[0]

def icc_full(subjects, values, version="ICC1"):
    counts = Counter(subjects)
    assert len(set(counts.values())) == 1, "Different numbers of subject ratings in ICC"
    df = pandas.DataFrame({"subject": subjects, "value": values})
    df.sort_values("subject", inplace=True, kind="mergesort") # mergesort is only stable sort
    df['rater'] = np.tile(range(0, len(subjects)//len(set(subjects))), len(set(subjects)))
    iccs = pingouin.intraclass_corr(data=df, targets="subject", raters="rater", ratings="value")
    iccs.set_index('Type', inplace=True)
    return (iccs.loc[version]['ICC'], tuple(iccs.loc[version]['CI95%']), iccs.loc[version]['pval'])

def trt_bootstrap_icc(trt, stat, N_shuffles):
    """trt = trt-style dataset, stat = vector of the stat to do icc bootstrap, N_shuffles is obvious"""
    stats = []
    try:
        subjs = np.asarray(list(trt.get_subject_info()['subject']))
    except:
        subjs = np.asarray(list(trt.base.get_subject_info()[trt.inds]['subject']))
    sort = np.argsort(subjs)
    subjs = subjs[sort]
    subjs_unique = np.unique(subjs)
    N_unique = len(subjs_unique)
    N_repeats = len(subjs)//N_unique
    assert len(subjs) == N_repeats*N_unique
    fakenames = np.repeat([range(0, N_unique)], N_repeats)
    for i in range(0, N_shuffles):
        perm = np.random.choice(range(0, N_unique), N_unique)
        permall = np.repeat(perm*N_repeats, N_repeats) + np.tile(range(0, N_repeats), N_unique)
        s = icc_full(fakenames, stat[sort][permall])
        stats.append(s)
    return stats

def fingerprint(subjects, values):
    assert values.shape[0] == len(subjects)
    corrs = np.corrcoef(values)
    np.fill_diagonal(corrs, -1)
    maxcorrs = np.argmax(corrs, axis=0)
    best_match_subject = np.asarray(subjects)[maxcorrs]
    return np.mean(best_match_subject == subjects)

def lin(vals1, vals2):
    """Lin's concordance correlation coefficient"""
    return 2*np.cov(vals1, vals2, ddof=0)[0,1]/(np.var(vals1) + np.var(vals2) + (np.mean(vals1)-np.mean(vals2))**2)

def lin_significance(vals1, vals2, tails=1):
    """Test significance of Lin's concordance using asymptotic normal
    approxmation, see
    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Lins_Concordance_Correlation_Coefficient.pdf"""
    lins = lin(vals1, vals2)
    pearson = np.corrcoef(vals1, vals2)[0,1]
    v = np.abs(np.mean(vals1)-np.mean(vals2))/np.sqrt(np.std(vals1)*np.std(vals2))
    # Mean/Variance of fisher transform of lin's concordance
    try:
        mean = .5 * np.log((1 + lins)/(1 - lins))
    except FloatingPointError:
        return 0
    variance = 1/(len(vals1)-2) * (
        (1-pearson**2)*lins**2 / ((1-lins**2)*pearson**2) +
        2*lins**3*(1-lins)*v**2 / (pearson*(1-lins**2)**2) -
        lins**4*v**4 / (2*pearson**2*(1-lins**2)**2)
        )
    stdevs_from_zero = mean/np.sqrt(variance)
    if tails == 1:
        return 1-scipy.stats.norm.cdf(stdevs_from_zero)
    elif tails == 2:
        raise NotImplementedError

    

# Partial correlation between x and y
def pcorr(x, y, covariate):
    df = pandas.DataFrame(np.asarray([x, y, covariate]).T, columns=["x", "y", "c"])
    pc = pingouin.partial_corr(data=df, x="x", y="y", covar="c", method="pearson")
    return float(pc['r']), float(pc['p-val']), float(pc['CI95%'][0][0]), float(pc['CI95%'][0][1])

# Partial correlation between x and y
def pcorr2(x, y, covariate1, covariate2):
    df = pandas.DataFrame(np.asarray([x, y, covariate1, covariate2]).T, columns=["x", "y", "c1", "c2"])
    pc = pingouin.partial_corr(data=df, x="x", y="y", covar=["c1", "c2"], method="pearson")
    return float(pc['r']), float(pc['p-val']), float(pc['CI95%'][0][0]), float(pc['CI95%'][0][1])

# def plot_on_volume_parcellation(values, filename, clim, cmap="viridis", parcellation="aal"):
#     if parcellation == "aal":
#         d = nibabel.load(AAL_PATH+"/ROI_MNI_V4.nii").get_data()
#         labels = np.asarray(pandas.read_csv(AAL_PATH+"/ROI_MNI_V4.txt", sep="\t", header=None)[2])
#         if len(values) == 115: # We remove one, see datasets.py
#             excluded = datasets._camcan_excluded_regions("AAL")
#             labels = np.delete(labels, excluded)
#         cpos = lambda : [[150*_s*_hc+d.shape[0]/2+d.shape[0]/6*_hc, d.shape[1]/2, d.shape[2]/2-5], [d.shape[0]/2+d.shape[0]/6*_hc, d.shape[1]/2, d.shape[2]/2-5], [0, -.1, 1]]
#         _lh_mask = (d % 10 == 1)
#         _rh_mask = (d % 10 == 2)
#         _vermis_mask = (d % 10 == 0) & (d != 0)
#         _vermis_right_mask = _vermis_mask.copy()
#         _vermis_right_mask[0:(d.shape[0]//2),:,:] = 0
#         _vermis_left_mask = _vermis_mask.copy()
#         _vermis_left_mask[(d.shape[0]//2):,:,:] = 0
#         lh_mask = _lh_mask | _vermis_left_mask
#         rh_mask = _rh_mask | _vermis_right_mask
#     elif parcellation == "shen":
#         d = nibabel.load(SHEN_PATH).get_data()
#         labels = np.arange(1, 269)
#         cpos = lambda : [[290*_s*_hc+d.shape[0]/2-d.shape[0]/30*_hc, d.shape[1]/2, d.shape[2]/2-8], [d.shape[0]/2-d.shape[0]/30*_hc, d.shape[1]/2, d.shape[2]/2-8], [0, -.1, 1]]
#         lh_mask = (d<=(len(labels)//2)) & (d>0)
#         rh_mask = d>(len(labels)//2)
#     dnew = np.zeros(d.shape).astype(float)
#     assert len(labels) == len(values)
#     for num,val in zip(labels, list(values)):
#         dnew[d==num] = val
#     harryplotter = pyvista.Plotter(off_screen=True, shape=(2,2), border=False, border_color='white')
#     harryplotter.background_color = (1,1,1)
#     for i,hemi in enumerate(["L", "R"]):
#         _hc = -1 if hemi == "L" else 1
#         dnewer = dnew.copy()
#         if hemi == "R":
#             dnewer[rh_mask] = 0
#         else:
#             dnewer[lh_mask] = 0
#         for side in [0,1]:
#             _s = 1 if side == 0 else -1
#             # Camera position, focal point, and "view up" (points towards the sky)
#             pvd = pyvista.wrap(dnewer)
#             harryplotter.subplot(side,i)
#             dnewer[dnewer<=clim[0]] = clim[0]+.0001
#             dnewer[dnewer>=clim[1]] = clim[1]-.0001
#             harryplotter.add_volume(pvd, shade=True, clim=clim, cmap=cmap, show_scalar_bar=False, opacity=np.asarray([0]+[1]*(len(set(dnewer.flat))-1)))
#             harryplotter.camera_position = cpos()
#     # harryplotter.enable_anti_aliasing()
#     harryplotter.screenshot(filename, transparent_background=True)

def plot_on_volume_parcellation(values, filename, vlim, cmap="viridis", parcellation="aal"):
    if parcellation == "aal":
        d = nibabel.load(AAL_PATH+"/ROI_MNI_V4.nii").get_data()
        labels = np.asarray(pandas.read_csv(AAL_PATH+"/ROI_MNI_V4.txt", sep="\t", header=None)[2])
        if len(values) == 115: # We remove one, see datasets.py
            excluded = datasets._camcan_excluded_regions("AAL")
            labels = np.delete(labels, excluded)
        cpos = lambda : [[30, 0, -150*_s], [0, 0, 0], [1, -.1, 0]]
        _lh_mask = (d % 10 == 1)
        _rh_mask = (d % 10 == 2)
        _vermis_mask = (d % 10 == 0) & (d != 0)
        _vermis_right_mask = _vermis_mask.copy()
        _vermis_right_mask[0:(d.shape[0]//2),:,:] = 0
        _vermis_left_mask = _vermis_mask.copy()
        _vermis_left_mask[(d.shape[0]//2):,:,:] = 0
        lh_mask = _lh_mask | _vermis_left_mask
        rh_mask = _rh_mask | _vermis_right_mask
    elif parcellation == "shen":
        d = nibabel.load(SHEN_PATH).get_data()
        labels = np.arange(1, 269)
        cpos = lambda : [[30, 0, -300*_s], [-10, 0, 0], [1, -.1, 0]]
        lh_mask = (d<=(len(labels)//2)) & (d>0)
        rh_mask = d>(len(labels)//2)
    dnew = np.ones(d.shape).astype(float)*-999
    assert len(labels) == len(values), str(len(labels))+" "+str(len(values))
    for num,val in zip(labels, list(values)):
        dnew[d==num] = val
    harryplotter = vedo.Plotter(size=(1600, 1200), offscreen=True)
    harryplotter.renderer.UseFXAAOn()
    ims = []
    for i,hemi in enumerate(["L", "R"]):
        _hc = -1 if hemi == "L" else 1
        dnewer = dnew.copy()
        if hemi == "R":
            dnewer[~rh_mask] = -999
        else:
            dnewer[~lh_mask] = -999
        dnewer[(dnewer<vlim[0])&(dnewer>-999)] = vlim[0]+.0000001
        dnewer[dnewer>vlim[1]] = vlim[1]-.00000001
        for side in [0,1]:
            _s = 1 if side == 0 else -1
            # Camera position, focal point, and "view up" (points towards the sky)
            vol = vedo.Volume(dnewer)
            lego = vol.legosurface(vmin=vlim[0], vmax=vlim[1], cmap=cmap)
            lego.pos(np.subtract(*vol.xbounds())/2 - vol.xbounds()[0],
                     np.subtract(*vol.ybounds())/2 - vol.ybounds()[0],
                     np.subtract(*vol.zbounds())/2 - vol.zbounds()[0])
            lego.lighting(ambient=.7, diffuse=.4, specular=0, specularPower=5)
            harryplotter.show(lego, camera={"pos": cpos()[0], "focalPoint": cpos()[1], "viewup": cpos()[2]})
            harryplotter.screenshot(f"_{side}{hemi}.png")
            ims.append(PIL.Image.open(f"_{side}{hemi}.png"))
    concat = PIL.Image.new(ims[0].mode, (ims[0].width * 2, ims[0].height * 2))
    concat.paste(ims[1], (0,0))
    concat.paste(ims[0], (0,ims[0].height))
    concat.paste(ims[2], (ims[0].width,0))
    concat.paste(ims[3], (ims[0].width,ims[0].height))
    concat.save(filename)
    harryplotter.close()


COLOR_CYCLE = sns.color_palette("Greens", 4)[1:]

ARROWSTYLE = "->,head_width=3,head_length=4"

_cp = sns.color_palette('Set2')
MODEL_PAL = {"data": 'k',
             "retest": 'k',
             "Colorless": (1, 0, 0),#_cp[1],
             "ColorlessHet": (1, 0, 0),#_cp[1],
             "ColorlessHom": _cp[1],
             "Colorfull": _cp[7],
             "ColorfullHom": _cp[7],
             "ColorlessTimeonly": _cp[5],
             "ColorfullTimeonly": _cp[5],
             "Colorsurrogate": _cp[7],
             "Spaceonly": _cp[4],
             "phase": _cp[0],
             "degreerand": _cp[2],
             "zalesky": _cp[6], 
             "zalesky2012": _cp[6],
             "eigen": _cp[3],
             }

models_sorted = ["ColorlessHet", "ColorlessHom", "Colorfull", "ColorlessTimeonly", "Spaceonly", "zalesky", "phase", "degreerand", "eigen"]

# model_names = {"data": "Data",
#                "ColorlessHet": "Generative model",
#                "ColorlessHom": "Generative model (homogeneous)",
#                "Colorfull": "Perfect SNR",
#                "Colorsurrogate": "Homogeneous (surrogate)",
#                "ColorfullHom": "Perfect SNR (homogeneous)",
#                "ColorlessTimeonly": "Time only",
#                "Spaceonly": "Spatial only",
#                "phase": "Phase randomization",
#                "degreerand": "Edge reshuffle",
#                "eigen": "Eigensurrogate",
#                "GenTS": "Autoregressive",
#                "zalesky": "Zalesky (2012)",
# }

metric_names = {"assort": "Assortativity", "cluster": "Clustering",
                "lefficiency": "Local eff.", "gefficiency": "Global eff.",
                "modularity": "Modularity", "transitivity": "Transitivity",
                "meancor": "Mean-FC"}

loss_names = {"eigsL2": "eigenvalues",
              "graphmetricsv2": "graph metrics",
              "rawcorrelation": "FC matrix similarity",
              "none": "N/A",
              }

long_name_ds = {"hcp": "Human connectome project",
                "hcpgsr": "Human connectome project w/ GSR",
                "achard": "Human connectome project, AAL parcellation",
                "bsnip": "BSNIP dataset",
                "hcp0": "Human connectome project",
                "hcpgsr0": "Human connectome project w/ GSR",
                "achard0": "Human connectome project, AAL parcellation",
                "bsniphc": "BSNIP dataset"}

short_name_ds = {"hcp": "HCP",
                "hcpgsr": "HCP w/ GSR",
                "achard": "HCP-Termenon",
                "bsnip": "BSNIP",
                "hcp0": "HCP",
                "hcpgsr0": "HCP w/ GSR",
                "achard0": "HCP-Termenon",
                "bsniphc": "BSNIP"}

# long_name = {"assort": "Assortativity", "cluster": "Clustering coef.",
#              "lefficiency": "Local efficiency", "gefficiency": "Global efficiency",
#              "modularity": "Modularity", "transitivity": "Transitivity",
#              "meanar1": "Mean AR1", "meanar1s": "Mean AR1",
#              "direct_lmbda": "$\\lambda$",
#              "direct_lmbdas": "$\\lambda$",
#              "lmbda": "$\\lambda$",
#              "direct_floor": "GC",
#              "direct_floors": "GC",
#              "floor": "GC",
#              "lmbda": "$\\lambda$",
#              "lmbdagen": "$\\lambda_{gen}$",
#              "ar1": "AR1 coefficient",
#              "meancor": "GBC",
#              "varcor": "VBC",
#              "kurtcor": "KBC",
#              "diam": "Diameter",
#              "path": "Mean path length"}

names_for_stuff = {
    # Unweighted graph metrics
    "assort": "Assortativity",
    "cluster": "Clustering coef.",
    "lefficiency": "Local efficiency",
    "gefficiency": "Global efficiency",
    "modularity": "Modularity",
    "transitivity": "Transitivity",
    # Weighted graph metrics
    "meancor": "Mean-FC",
    "varcor": "Var-FC",
    "kurtcor": "Kurt-FC",
    # Nodal graph metrics
    "mean": "Nodal Mean-FC",
    "var": "Nodal Var-FC",
    "kurt": "Nodal Kurt-FC",
    "gbc": "Nodal Mean-FC",
    "vbc": "Nodal Var-FC",
    "kbc": "Nodal Kurt-FC",
    "degree": "Degree",
    "centrality": "Centrality",
    # Timeseries measures
    "lmbda": "SA-λ",
    "loglmbda": "log(SA-λ)",
    "floor": "SA-∞",
    "meanar1": "Global TA-$\\Delta_1$",
    "rawar1": "TA-$\\Delta_1$",
    "ar1gen": "TA-$\\Delta_1^{gen}$",
    "lmbdagen": "SA-$\\lambda^{gen}$",
    "floorgen": "SA-$\\infty^{gen}$",
    "ar1": "Regional TA-$\\Delta_1$",
    "regionalar1": "Mean regional TA-$\\Delta_1$",
    "spatialmetrics": "log(SA-λ) + SA-∞",
    "allmetrics": "All",
    # Datasets
    "hcp": "HCP",
    "hcp0": "HCP",
    "hcp1200": "HCP",
    "hcp12000": "HCP",
    "camcan": "Cam-CAN",
    "hcpgsr": "HCP (w/ GSR)",
    "hcp0gsr": "HCP (w/ GSR)",
    "hcp1200gsr": "HCP (w/ GSR)",
    "hcp12000gsr": "HCP (w/ GSR)",
    "trt": "Yale-TRT",
    "LSDall": "LSD",
    "PsiAll": "Psilocybin",
    # Models
    "data": "Data",
    "retest": "Data (retest)",
    "ColorlessHet": "Spatiotemporal model",
    "ColorlessHom": "Spatiotemporal model (homogeneous)",
    "Colorsurrogate": "Intrinsic timescale + SA",
    "Colorfull": "Intrinsic timescale with SA (fit to eigenvalues)",
    "ColorfullTimeonly": "Intrinsic timescale",
    "ColorfullHom": "Intrinsic timescale (homogeneous)",
    "ColorlessTimeonly": "TA only",
    "Spaceonly": "SA only",
    "phase": "Phase randomization",
    "degreerand": "Edge reshuffle",
    "eigen": "Eigensurrogate",
    "GenTS": "Autoregressive",
    "zalesky": "Zalesky matching",
    "zalesky2012": "Zalesky matching",
    # Other
    "cm": "FC",
    "chance": "Chance",
    "reho": "ReHo",
}

short_names_for_stuff = {
    # Unweighted graph metrics
    "assort": "Assort.",
    "cluster": "Cluster",
    "lefficiency": "Local eff.",
    "gefficiency": "Global eff.",
    "modularity": "Modularity",
    "transitivity": "Transitivity",
    # Weighted graph metrics
    "meancor": "Mean-FC",
    "varcor": "Var-FC",
    "kurtcor": "Kurt-FC",
    # Nodal graph metrics
    "mean": "Nodal mean-FC",
    "var": "Nodal var-FC",
    "kurt": "Nodal kurt-FC",
    "gbc": "Nodal mean-FC",
    "vbc": "Nodal var-FC",
    "kbc": "Nodal kurt-FC",
    "degree": "Degree",
    "centrality": "Centrality",
    # Timeseries measures
    "lmbda": "SA-λ",
    "loglmbda": "log(SA-λ)",
    "floor": "SA-∞",
    "meanar1": "Global TA-$\\Delta_1$",
    "ar1": "Regional TA-$\\Delta_1$",
    "spatialmetrics": "SA-λ + SA-∞",
    "allmetrics": "All",
    # Datasets
    "hcp": "HCP",
    "camcan": "Cam-CAN",
    "hcpgsr": "HCP (w/ GSR)",
    "trt": "Yale-TRT",
    # Other
    "cm": "FC",
    "chance": "Chance",
    # Models
    "data": "Data",
    "retest": "Data (retest)",
    "ColorlessHet": "Spatiotemporal",
    "ColorlessHom": "Spatiotemporal (homogeneous)",
    "Colorsurrogate": "Intrinsic + SA",
    "Colorfull": "Intrinsic + SA (eigenvalues)",
    "ColorfullTimeonly": "Intrinsic",
    "ColorfullHom": "Intrinsic (homogeneous)",
    "ColorlessTimeonly": "TA only",
    "Spaceonly": "SA only",
    "phase": "Phase randomization",
    "degreerand": "Edge reshuffle",
    "eigen": "Eigensurrogate",
    "GenTS": "Autoregressive",
    "zalesky": "Zalesky matching",
    "zalesky2012": "Zalesky matching",
}

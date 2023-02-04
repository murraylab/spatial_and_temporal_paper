# Pharmacology figure
import cdatasets
import numpy as np
import matplotlib.pyplot as plt
from cand import Canvas, Vector, Point
from figurelib import names_for_stuff
import scipy
import seaborn as sns
import util
import pandas

_pal = sns.color_palette("Dark2")
PAL = [_pal[0], _pal[3], _pal[4]]
hueorder = ["LSD", "Psilocybin", "LSD+Ket"]

GSR = True
exps = ["Control", "LSD", "LSD+Ket"]
tps = ["early", "late"]
lsd = {exp : {tp : cdatasets.LSD(exp, tp, gsr=GSR) for tp in tps} for exp in exps}

pexps = ["Control", "Psilocybin"]
ptps = ["early", "late"]
psi = {exp : {tp : cdatasets.Psilocybin(exp, "middle" if tp == "early" else tp, gsr=GSR) for tp in ptps} for exp in pexps}


from itertools import repeat
import wbplot
_df_subj = []
_df_region = []
_all = []
_all_labels = []
VRANGE = (-.26, .26)
for timepoint in ["early", "late"]:
    for exp in ["LSD", "LSD+Ket", "Psilocybin"]:
        if "LSD" in exp:
            dsc = lsd["Control"][timepoint]
            ds = lsd[exp][timepoint]
        else:
            dsc = psi["Control"][timepoint]
            ds = psi[exp][timepoint]
        _df_subj.extend(zip(np.mean(ds.get_ar1s(), axis=1)-np.mean(dsc.get_ar1s(), axis=1), repeat(timepoint), repeat(exp), range(0, 100000)))
        _df_region.extend(zip(np.mean(ds.get_ar1s(), axis=0)-np.mean(dsc.get_ar1s(), axis=0), repeat(timepoint), repeat(exp), range(0, 100000)))
        _all.append(ds.get_ar1s()-dsc.get_ar1s())
        _all_labels.extend([(timepoint.capitalize(), exp)]*ds.get_ar1s().shape[0])
        if util.plock(f"{timepoint}-{exp.replace('+', '')}{'-nogsr' if GSR == False else ''}-ar1diff.png"):
            wbplot.pscalar(f"{timepoint}-{exp.replace('+', '')}{'-nogsr' if GSR == False else ''}-ar1diff.png", np.mean(ds.get_ar1s(), axis=0)-np.mean(dsc.get_ar1s(), axis=0), vrange=VRANGE, cmap="PuOr_r")
        if util.plock(f"{timepoint}-{exp.replace('+', '')}{'-nogsr' if GSR == False else ''}-ar1diff-tmap.png"):
            diffs = (ds.get_ar1s()-dsc.get_ar1s())
            tstat = np.mean(diffs, axis=0)/(np.std(diffs, axis=0)/np.sqrt(len(diffs[:,0])))
            thresh = scipy.stats.t(len(diffs[:,0])).ppf(.05/2)
            tstat[np.abs(tstat)<np.abs(thresh)] = 0
            wbplot.pscalar(f"{timepoint}-{exp.replace('+', '')}{'-nogsr' if GSR == False else ''}-ar1diff-tmap.png", tstat, vrange=(-5, 5), cmap="PuOr_r", orientation="landscape")


df_subj = pandas.DataFrame(_df_subj, columns=["value", "timepoint", "exp", "subject"])
df_region = pandas.DataFrame(_df_region, columns=["value", "timepoint", "exp", "region"])
allar1s = np.concatenate(_all)


titlestyle = {"weight": "bold", "size": 7}
#_pal = sns.color_palette("dark")
#PAL = [_pal[0], _pal[2], _pal[1]]

import sklearn
svd = sklearn.decomposition.TruncatedSVD(n_components=142)
scores = svd.fit_transform(allar1s)

SV1VRANGE = (0, .07)
wbplot.pscalar('component_all.png',  -svd.components_[0], cmap='viridis', vrange=SV1VRANGE)



c = Canvas(88, 185, "mm")
c.set_font("Nimbus Sans", ticksize=5, size=6)

c.add_axis("minigrid", Point(50, 61, "mm"), Point(75, 86, "mm"))

c.add_axis("scores", Point(57, 10, "mm"), Point(83, 20, "mm"))
c.add_axis("scree", Point(57, 30, "mm"), Point(83, 40, "mm"))

c.add_axis("byregion", Point(12, 64, "mm"), Point(34, 86, "mm"))

IMLEFT = 10
IMRIGHT = 50
IMBOT = 108
IMGAP = 24
brain_img_width = Vector(32, 0, "mm")

for i,tp in enumerate(["early", "late"]):
    for j,exp in enumerate(["LSD", "Psilocybin", "LSDKet"]):
        c.add_image(f"{tp}-{exp}{'-nogsr' if GSR == False else ''}-ar1diff.png", Point(IMLEFT if i==0 else IMRIGHT, IMBOT+IMGAP*(2-j), "mm"), width=brain_img_width, horizontalalignment="left", verticalalignment="bottom", unitname="img"+tp+exp)


c.add_colorbar("brainimage_colorbar", Point(.7, 0, "imgearlyLSDKet")+Vector(0, -.4, "cm"), Point(.3, 0, "imglateLSDKet")+Vector(0, -.2, "cm"), cmap="PuOr_r", bounds=VRANGE)
c.add_text("Drug minus control TA-$\\Delta_1$", (Point(0, 0, "imgearlyLSDKet") | Point(1, 0, "imglateLSDKet"))+Vector(0, -.9, "cm"), horizontalalignment="center", va="top")

c.add_text("Early", Point(.5, 1.08, "imgearlyLSD"), ha="center", va="bottom", **titlestyle)
c.add_text("Late", Point(.5, 1.08, "imglateLSD"), ha="center", va="bottom", **titlestyle)

c.add_text("LSD", Point(-.05, .5, "imgearlyLSD"), ha="right", va="center", **titlestyle, rotation=90, bbox=dict(facecolor='none', edgecolor=PAL[0], boxstyle="round"))
c.add_text("Psilocybin", Point(-.05, .5, "imgearlyPsilocybin"), ha="right", va="center", **titlestyle, rotation=90, bbox=dict(facecolor='none', edgecolor=PAL[1], boxstyle="round"))
c.add_text("LSD+Ket", Point(-.05, .5, "imgearlyLSDKet"), ha="right", va="center", **titlestyle, rotation=90, bbox=dict(facecolor='none', edgecolor=PAL[2], boxstyle="round"))

c.add_text("Change in Regional TA-$\\Delta_1$ on drug", (Point(0, 1.2, "imgearlyLSD") | Point(1, 1.2, "imglateLSD")), ha="center", va="bottom", **titlestyle)



c.add_image("component_all.png", Point(1, 15, 'mm'), width=Vector(45, 0, 'mm'), horizontalalignment="left", verticalalignment="bottom", unitname="sv1img")
c.add_text("Serotonergic impact on Regional TA-$\Delta_1$", Point(0.65, 1.08, "sv1img"), ha="center", va="bottom", **titlestyle)


c.add_colorbar("sv1img_colorbar", Point(.3, 0, "sv1img")+Vector(0, -.4, "cm"), Point(.7, 0, "sv1img")+Vector(0, -.2, "cm"), cmap="viridis", bounds=SV1VRANGE)

c.add_text("First singular vector loading", Point(.5, 0, "sv1img")-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")

# ax = c.ax("scree")
# ax.cla()
# ax.plot([1, 2, 3, 4, 5], svd.explained_variance_ratio_[0:5], c='k')
# #ax.plot([1, 2, 3, 4, 5], svd.singular_values_[0:5]**2/np.sum(svd.singular_values_**2), c='k', linestyle=':')
# sns.despine(ax=ax)
# ax.set_xticks([1, 2, 3, 4, 5])
# ax.set_xlabel("Component #")
# ax.set_ylabel("% var explained\nEigenvalue")

ax = c.ax("scree")
ax.cla()
var_exp_by_dataset = np.var(allar1s - (scores[:,[0]] @ svd.components_[[0]]), axis=1)/np.var(allar1s)
labels_df = pandas.DataFrame(_all_labels, columns=["Timepoint", "Experiment"])
labels_df['val'] = var_exp_by_dataset * 100

plotdata = labels_df.groupby(['Timepoint', 'Experiment'])['val'].apply(list).reindex(hueorder, level='Experiment').reset_index()
positions = [1, 1.5, 2, 3, 3.5, 4]
colors = PAL+PAL
for i in range(0, len(positions)):
    bplot = ax.boxplot(list(plotdata['val'])[i:(i+1)], medianprops={"color": colors[i]}, boxprops={"color": colors[i]}, whiskerprops={"color": colors[i]}, capprops={"color": colors[i]}, vert=True, showfliers=False, positions=positions[i:(i+1)], widths=.35)

    #bplot = ax.boxplot(positions[i], list(plotdata['val'])[i:(i+1)], color=colors[i], width=.45)
    #ax.errorbar(positions[i], np.mean(list(plotdata['val'])[i:(i+1)]), yerr=scipy.stats.sem(list(plotdata['val'])[i]), color=(.26, .26, .26))

#sns.barplot(x="Timepoint", hue="Experiment", y="val", data=labels_df, ax=ax, palette=PAL, hue_order=hueorder, errwidth=2)
# ax.get_legend().remove()
ax.set_ylabel("% var exp")
ax.set_xticks([])
ax.axhline(0, c='k')
c.add_text("Early", Point(positions[1], 0, ("scree", "axis_scree"))+Vector(0, -.1, "in"))
c.add_text("Late", Point(positions[4], 0, ("scree", "axis_scree"))+Vector(0, -.1, "in"))
sns.despine(ax=ax, bottom=True)
ax.tick_params(axis='x', which='both',length=0)
ax.set_xlabel("")
ax.set_ylim(0, ax.get_ylim()[1])
c.add_text("Variance explained", Point(.5, 1.13, "axis_scree"), ha="center", va="bottom", **titlestyle)

ax = c.ax("scores")
ax.cla()
labels_df = pandas.DataFrame(_all_labels, columns=["Timepoint", "Experiment"])
labels_df['val'] = -scores[:,0]

plotdata = labels_df.groupby(['Timepoint', 'Experiment'])['val'].apply(list).reindex(hueorder, level='Experiment').reset_index()
positions = [1, 1.5, 2, 3, 3.5, 4]
colors = PAL+PAL
for i in range(0, len(positions)):
    bplot = ax.boxplot(list(plotdata['val'])[i:(i+1)], medianprops={"color": colors[i]}, boxprops={"color": colors[i]}, whiskerprops={"color": colors[i]}, capprops={"color": colors[i]}, vert=True, showfliers=False, positions=positions[i:(i+1)], widths=.35)
    #bplot = ax.bar(positions[i], np.mean(list(plotdata['val'])[i:(i+1)]), color=colors[i], width=.45)
    #ax.errorbar(positions[i], np.mean(list(plotdata['val'])[i:(i+1)]), yerr=scipy.stats.sem(list(plotdata['val'])[i]), color=(.26, .26, .26))

for i,dat in enumerate(list(plotdata['val'])):
    test = scipy.stats.wilcoxon(dat)
    text = "**" if test.pvalue<.01 else "*" if test.pvalue<.05 else ""
    if text:
        c.add_text(text, Point(positions[i], 1, ("scores", "axis_scores")))



# sns.barplot(x="Timepoint", hue="Experiment", y="val", data=labels_df, ax=ax, palette=PAL, hue_order=hueorder, errwidth=2)
# ax.get_legend().remove()
ax.set_ylabel("SV score")
ax.set_xticks([])
ax.axhline(0, c='k')
c.add_text("Early", Point(positions[1], 0, ("scores", "axis_scores"))+Vector(0, -.1, "in"))
c.add_text("Late", Point(positions[4], 0, ("scores", "axis_scores"))+Vector(0, -.1, "in"))
sns.despine(ax=ax, bottom=True)
ax.tick_params(axis='x', which='both',length=0)
ax.set_xlabel("")
c.add_text("Singular vector scores", Point(.5, 1.13, "axis_scores"), ha="center", va="bottom", **titlestyle)
#wilcoxons = labels_df.groupby(["Timepoint", "Experiment"])['val'].apply(list).map(scipy.stats.wilcoxon)

ar1s = {}
for timepoint in ["early", "late"]:
    ar1s[timepoint] = {}
    dsc = lsd["Control"][timepoint]
    for exp in ["LSD", "LSD+Ket"]:
        ds = lsd[exp][timepoint]
        ar1s[timepoint][exp] = np.mean(ds.get_ar1s()-dsc.get_ar1s(), axis=0)
    dsc = psi["Control"][timepoint]
    exp = "Psilocybin"
    ds = psi[exp][timepoint]
    ar1s[timepoint][exp] = np.mean(ds.get_ar1s()-dsc.get_ar1s(), axis=0)


names = [
    "LSD\nearly",
    "LSD\nlate",
    "Psilocybin\nearly",
    "Psilocybin\nlate",
    "LSD+Ket\nearly",
    "LSD+Ket\nlate",
    ]

diffs = [
    ar1s["early"]["LSD"],
    ar1s["late"]["LSD"],
    ar1s["early"]["Psilocybin"],
    ar1s["late"]["Psilocybin"],
    ar1s["early"]["LSD+Ket"],
    ar1s["late"]["LSD+Ket"],
    ]






sigs = util.pload("sigs.pkl")
grid = np.zeros((len(diffs), len(diffs)))*np.nan
labels = grid.tolist()
import statsmodels
_corrected_sigs05 = statsmodels.stats.multitest.multipletests(1-np.asarray(list(sigs.values())), .05, 'holm')
_corrected_sigs01 = statsmodels.stats.multitest.multipletests(1-np.asarray(list(sigs.values())), .01, 'holm')
_corrected_sigs_keys = list(sigs.keys())
adj_p = dict(zip(list(_corrected_sigs_keys), list(_corrected_sigs01[1])))

for i in range(0, len(diffs)):
    for j in range(i+1, len(diffs)):
        spear = scipy.stats.spearmanr(diffs[j], diffs[i])
        grid[i,j] = spear.correlation
        #labels[i][j] = f"{spear.correlation:.2f}"
        pval = adj_p[(min(i,j), max(i,j))]
        labels[i][j] = "**" if pval < .01 else "*" if pval < .05 else ""



ax = c.ax("minigrid")
VSCALEGRID = (-.72, .72)
ax.imshow(grid.T[1:,:-1], vmin=VSCALEGRID[0], vmax=VSCALEGRID[1], cmap="bwr", aspect='auto')
ax.axis('off')
c.add_colorbar("minigrid_colorbar", Point(0, 0, "axis_minigrid")-Vector(0, .4, "cm"), Point(1, 0, "axis_minigrid")-Vector(0, .2, "cm"), cmap="bwr", bounds=VSCALEGRID)
c.add_text("Spearman correlation", Point(.5, 0, "axis_minigrid")-Vector(0, .9, "cm"), horizontalalignment="center", verticalalignment="top")
c.add_text("Similarity of drug effect\non Regional TA-$\\Delta_1$", Point(.5, 1.2, "axis_minigrid"), ha="center", **titlestyle)

colors = [p for p in PAL for _ in range(0, 2)]
for i in range(0, len(names)-1):
    c.add_text(names[i].replace("\n", " "), Point(i, -.6+i, "minigrid"), rotation=0, horizontalalignment="left", verticalalignment="bottom", size=5, color=colors[i])

for i in range(0, len(names)-1):
    c.add_text(names[i+1], Point(-.7, i, "minigrid"), horizontalalignment="right", verticalalignment="center", size=5, color=colors[i+1])

for i in range(0, len(names)-1):
    for j in range(i, len(names)-1):
        c.add_text(labels[i][j+1], Point(i, j+.2, "minigrid"), horizontalalignment="center", verticalalignment="center", size=10)

# ax = c.ax("bysubject")
# ax.cla()
# sns.boxplot(data=df_subj, x="timepoint", hue="exp", y="value", ax=ax, showfliers=False)
# ax.get_legend().remove()
# ax.set_ylabel("SV score")
# sns.despine(ax=ax, bottom=True)
# ax.tick_params(axis='x', which='both',length=0)
# ax.set_xlabel("")
# ax.axhline(0, c='k')

ax = c.ax("byregion")
ax.cla()


plotdata = df_region.groupby(['timepoint', 'exp'])['value'].apply(list).reindex(hueorder, level='exp').reset_index()
positions = [1, 1.5, 2, 3, 3.5, 4]
colors = PAL+PAL
for i in range(0, len(positions)):
    bplot = ax.boxplot(list(plotdata['value'])[i:(i+1)], medianprops={"color": colors[i]}, boxprops={"color": colors[i]}, whiskerprops={"color": colors[i]}, capprops={"color": colors[i]}, vert=True, showfliers=False, positions=positions[i:(i+1)], widths=.35)

for i,dat in enumerate(list(plotdata['value'])):
    test = scipy.stats.wilcoxon(dat)
    text = "**" if test.pvalue<.01 else "*" if test.pvalue<.05 else ""
    if text:
        c.add_text(text, Point(positions[i], 1, ("byregion", "axis_byregion")))

ax.axhline(0, c='k')
sns.despine(bottom=True, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Change in TA-$\Delta_1$")
ax.set_xticks([])
c.add_text("Early", Point(positions[1], 0, ("byregion", "axis_byregion"))+Vector(0, -.1, "in"))
c.add_text("Late", Point(positions[4], 0, ("byregion", "axis_byregion"))+Vector(0, -.1, "in"))
c.add_text("Change in mean\nRegional TA-$\Delta_1$", Point(.5, 1.2, "axis_byregion"), ha="center", **titlestyle)


# c.add_legend(Point(2.5, 3.0, "in"), [(t, {"color": c, "linestyle": "none", "marker": "s"}) for t,c in zip(hueorder,PAL)],
#              line_spacing=Vector(0, 1.3, "Msize"), sym_width=Vector(1, 0, "Msize"), fontsize=5)


# sns.boxplot(data=df_region, x="timepoint", hue="exp", y="value", ax=ax, showfliers=False, palette=PAL, hue_order=hueorder)
# ax.get_legend().remove()
# sns.despine(ax=ax, bottom=True)
# ax.tick_params(axis='x', which='both',length=0)
# ax.set_xlabel("")
# ax.axhline(0, c='k')
# colors = sns.color_palette()
c.add_legend(Point(7, 59, "mm"), [(n,{"color": c, "linewidth": 3}) for n,c in zip(["LSD", "Psilocybin", "LSD+Ket"], PAL)],
             line_spacing=Vector(0, 1.1, "Msize"), sym_width=Vector(0.6, 0, "Msize"), padding_sep=Vector(.5, 0, "Msize"))

c.add_figure_labels([("a", "imgearlyLSD"),
                     ("b", "byregion", Vector(-3, 4, "mm")),
                     ("c", "minigrid", Vector(-5, 4, "mm")),
                     ("d", "sv1img", Vector(6, 3, "mm"))], size=8)

c.save("figure5.pdf")
c.show()

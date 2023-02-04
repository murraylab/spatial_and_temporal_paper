import numpy as np
import seaborn as sns
import scipy
import pandas
from cand import Canvas, Vector, Point
import cdatasets
import util
from figurelib import MODEL_PAL, names_for_stuff, short_names_for_stuff, pcorr, plot_on_volume_parcellation, simplescatter, COLOR_CYCLE, pcorr2
from config import AAL_PATH
import os

#################### Establish the cache ####################

import models
fn = "_f4_cache_perturb.pkl"
if os.path.isfile(fn):
    df_lmbda,df_floor = util.pload(fn)
else:
    cc = cdatasets.CamCanFiltered()
    model = models.Model_Colorless
    regionar1 = np.mean(cc.get_ar1s(), axis=0)
    rows_floor = []
    floors = np.linspace(0, .95, 20)
    lmbdas = [20, 50, 80]
    for seed in range(0, 10):
        for l in lmbdas:
            for f in floors:
                cm = model.generate(distance_matrix=cc.get_dists(), params={"lmbda": l, "floor": f}, ar1vals=regionar1, num_timepoints=cc.N_timepoints(), TR=cc.TR/1000, highpass_freq=cc.highpass, seed=seed)
                metrics = util.graph_metrics_from_cm(cm)
                cmstats = util.cm_metrics_from_cm(cm)
                rows_floor.append(dict([("lmbda", l), ("floor", f), ("seed", seed)] + list(metrics.items()) + list(cmstats.items())))
    
    df_floor = pandas.DataFrame(rows_floor)
    
    rows_lmbda = []
    lmbdas = np.logspace(2, 4.5, 20, base=np.exp(1))
    floors = [.2, .5, .8]
    for seed in range(0, 10):
        for l in lmbdas:
            for f in floors:
                cm = model.generate(distance_matrix=cc.get_dists(), params={"lmbda": l, "floor": f}, ar1vals=regionar1, num_timepoints=cc.N_timepoints(), TR=cc.TR/1000, highpass_freq=cc.highpass, seed=seed)
                metrics = util.graph_metrics_from_cm(cm)
                cmstats = util.cm_metrics_from_cm(cm)
                rows_lmbda.append(dict([("lmbda", l), ("floor", f), ("seed", seed)] + list(metrics.items()) + list(cmstats.items())))
    
    df_lmbda = pandas.DataFrame(rows_lmbda)
    df_lmbda['flr'] = df['floor']
    util.psave(fn, (df_lmbda,df_floor))


#################### Set up the canvas ####################

FILENAME = "figure4.pdf"

cc = cdatasets.CamCanFiltered()
titlestyle = {"weight": "bold", "size": 7}


c = Canvas(3.5, 5.1, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)
# c.debug_grid(Vector(1, 1, "in"))
# c.debug_grid(Vector(.5, .5, "in"), linewidth=.5)

# ROW1 = 2.8
# ROW2 = .5
ROW1 = 2.9
ROW2 = .55

# c.add_grid(["age_vs_meanar1", "age_vs_loglmbda", "age_vs_floor"], 1, Point(4, ROW1, "in"), Point(6.5, ROW1+.7, "in"), size=Vector(.7, .7, "in"))
c.add_axis("metrics_over_age", Point(.5, ROW1, "in"), Point(1.2, ROW1+1.8, "in"))
c.add_grid(["age_vs_gefficiency", "age_vs_varcor", "age_vs_kurtcor", "age_vs_loglmbda", "age_vs_floor"], 5, Point(.5, -.15, "axis_metrics_over_age")+Vector(.6,0,"in"), Point(.5, .95, "axis_metrics_over_age")+Vector(.90,0,"in"),size=Vector(.30, .30, "in"))


img_agecorr_pos = Point(0, ROW2, "in")
img_agecorr_height = Vector(0, 1.5, "in")
c.add_axis("networks", Point(2.6, ROW2+0.95, "in"), Point(3.4, ROW2+1.55, "in"))

c.add_axis("bestcog", Point(2.6, ROW2-.2, "in"), Point(3.25, ROW2+.2, "in"))


#################### Model predictions for age ####################

age_metrics = ["gefficiency", "varcor", "kurtcor"]
ts_metrics = ["floor", "lmbda"]
c.add_grid([f"age_{a}_lmbda" for a in age_metrics], 1, Point(2.1, ROW1+-.3, "in"), Point(3.4, ROW1+.30+-.2, "in"), size=Vector(.35, .35, "in"))
c.add_grid([f"age_{a}_floor" for a in age_metrics], 1, Point(2.1, ROW1+.60, "in"), Point(3.4, ROW1+.55+.30, "in"), size=Vector(.35, .35, "in"))
c.add_grid([f"age_{a}_data" for a in age_metrics], 1, Point(2.1, ROW1+1.42, "in"), Point(3.4, ROW1+.30+1.42, "in"), size=Vector(.35, .35, "in"))

age_metrics_ranges = {"gefficiency": (.35, .50), "varcor": (0, .15), "kurtcor": (-1, 4)}

for age_metric in age_metrics:
    ax = c.ax(f"age_{age_metric}_data")
    c.add_text("Age", Point(.5, -.2, f"axis_age_{age_metric}_data"), ha="center", va="center", size=5)
    c.add_text(short_names_for_stuff[age_metric], Point(.5, 1.1, f"axis_age_{age_metric}_data"), ha="center", va="center", size=5)
    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

c.add_arrow(Point(.2, .8, f"axis_age_gefficiency_data"), Point(.8, .2, f"axis_age_gefficiency_data"), linewidth=1, arrowstyle="->,head_width=3,head_length=4")
c.add_arrow(Point(.2, .2, f"axis_age_varcor_data"), Point(.8, .8, f"axis_age_varcor_data"), linewidth=1, arrowstyle="->,head_width=3,head_length=4")
c.add_arrow(Point(.2, .8, f"axis_age_kurtcor_data"), Point(.8, .2, f"axis_age_kurtcor_data"), linewidth=1, arrowstyle="->,head_width=3,head_length=4")


for age_metric in age_metrics:
    ax = c.ax(f"age_{age_metric}_floor")
    ax.cla()
    c.add_text("Age", Point(.5, -.2, f"axis_age_{age_metric}_floor"), ha="center", va="center", size=5)
    #c.add_text(short_names_for_stuff[age_metric], Point(-.2, .5, f"axis_age_{age_metric}_floor"), ha="center", va="center", rotation=90, size=5)
    c.add_text(short_names_for_stuff[age_metric], Point(.5, 1.1, f"axis_age_{age_metric}_floor"), ha="center", va="center", size=5)
    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    group = df_floor.query(f"lmbda == 50").groupby('floor')
    ax.errorbar(group.mean().index, group[age_metric].mean(), yerr=group[age_metric].sem(), elinewidth=.4, color='k', linewidth=.8)
    ax.set_ylim(*age_metrics_ranges[age_metric])
    ax.invert_xaxis()

for age_metric in age_metrics:
    ax = c.ax(f"age_{age_metric}_lmbda")
    c.add_text("Age", Point(.5, -.2, f"axis_age_{age_metric}_lmbda"), ha="center", va="center", size=5)
    #c.add_text(short_names_for_stuff[age_metric], Point(-.2, .5, f"axis_age_{age_metric}_lmbda"), ha="center", va="center", rotation=90, size=5)
    c.add_text(short_names_for_stuff[age_metric], Point(.5, 1.1, f"axis_age_{age_metric}_lmbda"), ha="center", va="center", size=5)
    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    group = df_lmbda.query(f"flr == .5").groupby('lmbda')
    ax.errorbar(np.log(group.mean().index), group[age_metric].mean(), yerr=group[age_metric].sem(), elinewidth=.4, color='k', linewidth=.8)
    ax.set_ylim(*age_metrics_ranges[age_metric])


c.add_text(f"Data", Point(-.2, 1.45, f"axis_age_{age_metrics[0]}_data"), ha="left", weight="bold")
# c.add_text(f"Model predictions where...", Point(-.2, 1.9, f"axis_age_{age_metrics[0]}_floor"), ha="left", weight="bold")
# c.add_text(f"...{names_for_stuff['lmbda']} mediates age effects", Point(1.1, 1.45, f"axis_age_{age_metrics[2]}_lmbda"), ha="right", weight="bold")
# c.add_text(f"...{names_for_stuff['floor']} mediates age effects", Point(1.1, 1.45, f"axis_age_{age_metrics[2]}_floor"), ha="right", weight="bold")

c.add_text(f"Model predictions if {names_for_stuff['floor']}\nmediates age effects", Point(-.2, 1.3, f"axis_age_{age_metrics[0]}_floor"), ha="left", weight="bold", va="bottom")
c.add_text(f"Model predictions if {names_for_stuff['lmbda']}\nmediates age effects", Point(-.2, 1.3, f"axis_age_{age_metrics[0]}_lmbda"), ha="left", weight="bold", va="bottom")
#################### Plot model results ####################

# age_metrics = ["gefficiency", "varcor", "kurtcor"]
# ts_metrics = ["floor", "lmbda"]
# c.add_grid([f"age_{a}_{b}" for a in age_metrics for b in ts_metrics], 3, Point(2.3, ROW1, "in"), Point(3.4, ROW1+1.8, "in"), size=Vector(.5, .5, "in"))

# ages = cc.get_subject_info()['age']
# movement = cc.get_movement()

# age_metrics_ranges = {"gefficiency": (.35, .50), "varcor": (0, .15), "kurtcor": (-1, 4)}


# for m in age_metrics:
#     ax = c.ax(f"age_{m}_lmbda")
#     for i,f in enumerate(sorted(set(df_lmbda['floor']))):
#         lmbdagroup = df_lmbda.query(f"flr == {f}").groupby('lmbda')
#         ax.errorbar(np.log(lmbdagroup.mean().index), lmbdagroup[m].mean(), yerr=lmbdagroup[m].sem(), color=COLOR_CYCLE[i], elinewidth=1)
#         ax.set_ylim(*age_metrics_ranges[m])
#     ax = c.ax(f"age_{m}_floor")
#     for i,f in enumerate(sorted(set(df_floor['lmbda']))):
#         floorgroup = df_floor.query(f"lmbda == {f}").groupby('floor')
#         ax.errorbar(floorgroup.mean().index, floorgroup[m].mean(), yerr=floorgroup[m].sem(), color=COLOR_CYCLE[i], elinewidth=1)
#         ax.set_ylim(*age_metrics_ranges[m])


# for m in age_metrics:
#     for t in ts_metrics:
#         ax = c.ax(f"age_{m}_{t}")
#         sns.despine(ax=ax)
#         if m != age_metrics[-1]:
#             ax.set_xticks([])
#         else:
#             if t == "lmbda":
#                 ax.set_xlabel("$log(\\lambda_{gen})$")
#             elif t == "floor":
#                 ax.set_xlabel("GC$_{gen}$")
#         if t != ts_metrics[0]:
#             ax.set_yticks([])
#         else:
#             ax.set_ylabel(names_for_stuff[m])


# # Add arrows
# def plot_arrow(axname, up):
#     center = Point(.8, .95, "axis_"+axname)
#     offset = Vector(.15, .15 if up else -.15, "cm")
#     c.add_arrow(center-offset, center+offset, linewidth=1, arrowstyle="->,head_width=2,head_length=4")
#     # c.add_text("Age", center+offset.flipy()/3*(-1 if up else 1)-offset*.4, rotation=(45 if up else -45))
#     c.add_text("Age", center - offset.width()*1.2, ha="right")

# plot_arrow("age_gefficiency_lmbda", False)
# plot_arrow("age_gefficiency_floor", True)
# plot_arrow("age_varcor_lmbda", True)
# plot_arrow("age_varcor_floor", False)
# plot_arrow("age_kurtcor_lmbda", False)
# plot_arrow("age_kurtcor_floor", True)


# # c.add_legend(Point(.3, .8, "axis_age_gefficiency_lmbda"),
# #              [(n, {'color': sns.color_palette()[i]}) for i,n in enumerate(['a', 'b', 'c'])],
# #              line_spacing=Vector(0, 1.2, "Msize")
# #              )


# c.add_text("Model predictions", (Point(.5, 1, "axis_age_gefficiency_lmbda") | Point(.5, 1, "axis_age_gefficiency_floor"))+Vector(0, .10, "in"), ha="center", va="baseline", **titlestyle)

#################### Metrics over age ####################

ax = c.ax("metrics_over_age")
ax.cla()
metrics = ['assort', 'cluster', 'gefficiency', 'lefficiency', 'modularity', 'transitivity']
weighted_metrics = ['meancor', 'varcor', 'kurtcor']
tsmetrics = ['loglmbda', 'floor', 'meanar1']
mlabels = metrics+weighted_metrics+tsmetrics
metric_vals = {}
ages = cc.get_subject_info()['age']
movement = cc.get_movement()
metric_vals.update({m : pcorr(ages, cc.get_metrics()[m], movement) for m in metrics})
metric_vals.update({m : pcorr(ages, cc.get_cmstats()[m], movement) for m in weighted_metrics})
metric_vals['loglmbda'] = pcorr(ages, np.log(cc.get_lmbda()), movement)
metric_vals['floor'] = pcorr(ages, cc.get_floor(), movement)
metric_vals['meanar1'] = pcorr(ages, np.mean(cc.get_ar1s(), axis=1), movement)

barheights = [metric_vals[m][0] for m in mlabels]
barpvalues = [metric_vals[m][1] for m in mlabels]
barcis = [np.asarray(metric_vals[m][2:4])-metric_vals[m][0] for m in mlabels]
ax.barh(range(0, len(mlabels)), barheights, color=['r' if k in ['meanar1', 'loglmbda', 'floor'] else (.5, .5, .5) if 'cor' in k else 'k' for k in mlabels], clip_on=False)
ax.errorbar(barheights, range(0, len(mlabels)), xerr=np.abs(np.asarray(barcis).T), elinewidth=1, c='k', linewidth=0)
ax.invert_yaxis()
sns.despine(ax=ax)
ax.set_yticks([])
ax.set_ylim(-.5, len(mlabels)-.5)
ax.invert_yaxis()
ax.set_xlabel("Partial correlation")
ax.axvline(0, c='k', linewidth=1)
for i in range(0, len(barpvalues)):
    if barpvalues[i] < .05:
        ax.scatter(barheights[i]+.12*np.sign(barheights[i]), i, s=8, marker='*', c='k')

ax.set_xlim(-.25, .28)

for i,metric in enumerate(mlabels):
    c.add_text(short_names_for_stuff[metric], Point(-.05, i, ("axis_metrics_over_age", "metrics_over_age")), rotation=0, horizontalalignment="right", verticalalignment="center", size=5)


c.add_text("Correlation with age", Point(.7, 1.00, "axis_metrics_over_age")+Vector(0, .10, "in"), ha="center", va="baseline", **titlestyle)

#################### Inset for age-metric correlations ####################

# Mini scatter plot sfor ar1, lmbda, and GC
arrowx = max(barheights)
for axname, vals in [("loglmbda", np.log(cc.get_lmbda())),
                     ("floor", cc.get_floor()),
                     ("varcor", cc.get_cmstats()['varcor']),
                     ("kurtcor", cc.get_cmstats()['kurtcor']),
                     ("gefficiency", cc.get_metrics()['gefficiency'])]:
    ax = c.ax("age_vs_"+axname)
    ax.cla()
    # corplot(ages, meanar1, "Age", names_for_stuff["meanar1"], ax=ax)
    # partial_plot(ages, meanar1, movement, "age_vs_meanar1")
    simplescatter(ages, vals, ax, diag=False, s=.4, c='k', alpha=.35, linewidth=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ind = mlabels.index(axname)
    arrowx = max(barheights[ind], 0)
    arrowfrm = Point(arrowx+.03, ind, "metrics_over_age")
    arrowto = Point(-.35, .5 if axname not in ["varcor", "kurtcor"] else .1, "axis_age_vs_"+axname)
    c.add_arrow(arrowfrm, arrowto, lw=1, arrowstyle="->,head_width=2,head_length=4")
    c.add_text("Age", Point(.5, -.2, "axis_age_vs_"+axname), ha="center", va="center", size=5)
    c.add_text(short_names_for_stuff[axname], Point(-.2, .5, "axis_age_vs_"+axname), ha="center", va="center", rotation=90, size=5)

#################### Subnetwork changes in AR1 ####################

corrs = []
for i in range(0, cc.N_regions()):
    corrs.append(pcorr(cc.get_ar1s()[:,i], ages, movement))

aal_parc = pandas.read_csv(AAL_PATH+"/ROI_MNI_V4v2.txt", sep="\t", names=["label", "name", "number", "network", "networkmod"])

# We strip the last element since the missing element in AAL is in the cerebellum, at the end of the parcellation
df = pandas.DataFrame(np.asarray([aal_parc['network'][:-1], aal_parc['networkmod'][:-1], np.mean(cc.get_ar1s(), axis=0), [c[0] for c in corrs]]).T,columns=["network", "networkmod", "val", "ageval"]).query('networkmod != "Other"')

sorted_networks = list(df.astype({"ageval": float}).groupby("networkmod")['ageval'].mean().sort_values().index)
ax = c.ax("networks")
ax.cla()
sns.boxplot(y="networkmod", x="ageval", data=df, ax=ax, color='white', order=sorted_networks, width=.5, linewidth=.5, fliersize=0)
for i,box in enumerate(ax.artists):
    box.set_edgecolor('#000000')
    box.set_facecolor('white')
    # iterate over whiskers and median lines
    for j in range(6*i,6*(i+1)):
         ax.lines[j].set_color('#000000')



ax.axvline(0, c='k', linewidth=1)
ax.set_xlim(-.20, .25)

for i,netw in enumerate(sorted_networks):
    test = scipy.stats.wilcoxon(df[df['networkmod']==netw]['ageval'])
    if test.pvalue*len(sorted_networks)<.01:
        print(netw, "sig", test.pvalue)
        c.add_text("**", Point(.26, i, "networks")+Vector(0, -.11, "cm"), size=12)

ax.tick_params(axis=u'y', which=u'both',length=0)
sns.despine(ax=ax)
ax.set_xlabel("Mean partial correlation")
ax.set_ylabel("")
c.add_text(f"Correlation of {names_for_stuff['ar1']}\nwith age by network", Point(.25, 1.05, "axis_networks")+Vector(0, .15, "in")-Vector(0, .4, "cm")+Vector(0, .2, "cm"), ha="center", va="baseline", **titlestyle)
# df['corrs_pos'] = [c[1]<.05 and c[0] > 0 for c in corrs]
# df['corrs_neg'] = [c[1]<.05 and c[0] < 0 for c in corrs]
# frac_sig = df.groupby('networkmod')[['corrs_pos', 'corrs_neg']].sum()/df.groupby('networkmod')[['corrs_pos', 'corrs_neg']].count()
# for d in ["neg", "pos"]:
#     ax = c.ax("networks_"+d)
#     ax.cla()
#     if d == "neg":
#         ax.invert_xaxis()
#     ax.barh(range(len(frac_sig)), frac_sig['corrs_'+d])
#     ax.set_yticks([])
#     sns.despine(ax=ax, left=True)
#     ax.axvline(0, c='k', linewidth=1)



#################### Plot on the brain ####################

VRANGE = (-.25, .25)
CMAP = "PuOr_r"
plot_on_volume_parcellation(np.asarray([c[0] for c in corrs]), "_cc_ar1_vs_age.png", VRANGE, cmap=CMAP)
c.add_image("_cc_ar1_vs_age.png", img_agecorr_pos, height=img_agecorr_height, ha="left", va="bottom", unitname="img_ar1_age")
c.add_colorbar("img_ar1_age_colorbar", Point(.2, 0, "img_ar1_age")+Vector(0, -.6, "cm"), Point(.8, 0, "img_ar1_age")+Vector(0, -.4, "cm"), cmap=CMAP, bounds=VRANGE)
c.add_text("Partial correlation", Point(.5, 0, "img_ar1_age")+Vector(0, -1.1, "cm"), horizontalalignment="center", va="top")
c.add_text(f"Correlation of {names_for_stuff['ar1']}\nwith age", Point(.5, 1.15, "img_ar1_age")+Vector(0, .05, "in")-Vector(0, .4, "cm"), ha="center", va="bottom", **titlestyle)



#################### Predicting cognitive decline ####################

VRANGE = (-.15, .15)
CMAP = "PuOr_r"

wmnan = np.logical_not(np.isnan(cc.get_subject_info()["ace-r"]))
cs = [pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), cc.get_ar1s()[:,i][wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])[0] for i in range(0, cc.N_regions())]
csp = [pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), cc.get_ar1s()[:,i][wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])[1] for i in range(0, cc.N_regions())]
csci = [list(pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), cc.get_ar1s()[:,i][wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])[2:4]) for i in range(0, cc.N_regions())]


namemap = {
    "0_1lmbda": names_for_stuff["lmbda"],
    "0_2floor": names_for_stuff["floor"],
    "0_3meanar1": names_for_stuff["meanar1"],
    "0_4blank": "",
    "0_5varcor": names_for_stuff['varcor'],
    }

corr_meanar1 = pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), np.mean(cc.get_ar1s(), axis=1)[wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])
corr_lmbda = pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), np.asarray(cc.get_lmbda())[wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])
corr_floor = pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), np.asarray(cc.get_floor())[wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])

corr_varcor = pcorr2(np.asarray(cc.get_subject_info()["ace-r"][wmnan]), np.asarray(cc.get_cmstats()['varcor'])[wmnan], np.asarray(cc.get_subject_info()["age"])[wmnan], cc.get_movement()[wmnan])

ax = c.ax("bestcog")
ax.cla()
rows = pandas.DataFrame([{"parcel": "0_1lmbda", "val": corr_lmbda[0], "ci": list(corr_lmbda[2:4]), "n": 3, "pval": corr_lmbda[1]},
                    {"parcel": "0_2floor", "val": corr_floor[0], "ci": list(corr_floor[2:4]), "n": 3, "pval": corr_floor[1]},
                    {"parcel": "0_3meanar1", "val": corr_meanar1[0], "ci": list(corr_meanar1[2:4]), "n": 3, "pval": corr_meanar1[1]},
                    {"parcel": "0_4blank", "ci": [0, 0]},
                    {"parcel": "0_5varcor", "val": corr_varcor[0], "ci": list(corr_varcor[2:4]), "n": 9, "pval": corr_varcor[1]},
])
rows['parcelname'] = rows['parcel'].map(lambda x : namemap[x])
rows = rows.sort_values("parcel", key=lambda x : [list(namemap.keys()).index(v) for v in x])
ax.cla()
sns.barplot(y="parcelname", x="val", data=rows, ax=ax, color='k')
ax.errorbar(rows['val'], range(0, len(rows['val'])), xerr=np.abs(np.asarray(np.asarray(list(rows['ci']))-np.asarray(rows['val'])[:,None]).T), elinewidth=1, c='#555555', linewidth=0)
ax.axvline(0, c='k', linewidth=1)
#ax.set_xlim(-.15, .15)
ax.tick_params(axis=u'y', which=u'both',length=0)
sns.despine(ax=ax)
ax.set_xlabel("Partial correlation", loc="right")
ax.set_ylabel("")

#TODO
for i,(_,row) in enumerate(rows.iterrows()):
    if np.isnan(row['val']): continue
    if row['pval']*row['n']<.01:
        print(row)
        c.add_text("**", Point(.28, i, "bestcog")+Vector(0, -.10, "cm"), size=12)

c.add_text(f"Correlation of {names_for_stuff['ar1']}\nwith cognitive function", Point(.25, 1.15, ("axis_networks", "axis_bestcog")), ha="center", va="baseline", **titlestyle)

c.add_text("n.s.", Point(.28, 2, "bestcog"), style="italic", size=6)




LABELS_ROW2 = 1.7 + ROW2
c.add_figure_labels([
    ("a", "metrics_over_age"),
    ("b", "age_gefficiency_data"),
    ("c", "img_ar1_age", Point(.05, LABELS_ROW2, "in")),
    ("d", "networks", Point(2.05, LABELS_ROW2, "in")),
    ("e", "bestcog", Point(2.05, 1.1, "in")),
], size=8)




c.save(FILENAME)
c.show()


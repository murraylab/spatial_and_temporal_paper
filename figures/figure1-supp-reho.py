import matplotlib.pyplot as plt
import numpy as np
from cand import Canvas, Point, Vector
import cdatasets
import seaborn as sns
import figurelib
import util


c = Canvas(88, 80, "mm")
c.set_font("Nimbus Sans", ticksize=5, size=6)

c.add_axis("region_corr_hcp", Point(.5, 2.0, "in"), Point(1.0, 2.5, "in"))
c.add_axis("subject_corr_hcp", Point(1.65, 2.0, "in"), Point(2.15, 2.5, "in"))
c.add_axis("finger_hcp", Point(2.8, 2.0, "in"), Point(3.1, 2.5, "in"))

c.add_axis("region_corr_trt", Point(.5, .5, "in"), Point(1.0, 1.0, "in"))
c.add_axis("subject_corr_trt", Point(1.65, .5, "in"), Point(2.15, 1.0, "in"))
c.add_axis("finger_trt", Point(2.8, .5, "in"), Point(3.1, 1.0, "in"))

data_hcp = cdatasets.HCP1200()
datarep_hcp = cdatasets.HCP1200(3) # Not needed just for fingerprint cache
datatrt_hcp = cdatasets.HCP1200KindaLikeTRT()

data_trt = cdatasets.TRTKindaLikeHCP(0) # Not needed just for fingerprint cache
datarep_trt = cdatasets.TRTKindaLikeHCP(5)
datatrt_trt = cdatasets.TRT()


datasets = [
    ("HCP", data_hcp, datarep_hcp, datatrt_hcp, "_hcp", 30),
    ("Yale-TRT", data_trt, datarep_trt, datatrt_trt, "_trt", 10),
]

for (name,data,datarep,datatrt,sfx,nbins) in datasets:
    ar1s = data.get_ar1s()
    rehos = data.get_rehos()
    ax = c.ax("region_corr"+sfx)
    ax.cla()
    figurelib.corplot(np.mean(ar1s, axis=0), np.mean(rehos, axis=0), "Regional TA-$\Delta_1$", "Regional homogeneity (ReHo)", color='k', markersize=1, alpha=1, showr2=False, ax=ax, title="Mean regional\nTA-$\Delta_1$ vs ReHo")
    ax.scatter(np.mean(ar1s, axis=0), np.mean(rehos, axis=0), c='k', s=.1, alpha=1, clip_on=False)
    sns.despine(ax=ax)
    ax.set_title("Mean regional\nTA-$\Delta_1$ vs ReHo")
    ax.set_xlabel("Regional TA-$\Delta_1$")
    ax.set_ylabel("ReHo")
    
    ax = c.ax("subject_corr"+sfx)
    ax.hist([np.corrcoef(ar1s[i], rehos[i])[0,1] for i in range(0, ar1s.shape[0])], bins=nbins, color='k', clip_on=False)
    sns.despine(ax=ax)
    ax.set_title("Correlation between\nTA-$\Delta_1$ and ReHo")
    ax.set_ylabel("# scans")
    ax.set_xlabel("Regional TA-$\Delta_1$ vs\nReHo correlation")
    
    try:
        fn = f"_f1_cache_fingerprint_{datatrt.name}_{data.name}_{datarep.name}.pkl"
        (fingerprintcache, fingerprintcache_pair, same_subject, diff_subject, diff_regions) = util.pload(fn)
    except:
        fn = f"_f1_cache_fingerprint_{datatrt.name}.pkl"
        fingerprintcache_pair = util.pload(fn)
    
    ax = c.ax("finger"+sfx)
    ax.cla()
    subjects = np.asarray(datatrt.get_subject_info()['subject'])
    try:
        scan = datatrt.get_subject_info()['scan']
        fingerprintcache_pair["reho"] = [figurelib.fingerprint(subjects[(scan==(i+1))|(scan==(j+1))], datatrt.get_rehos()[(scan==(i+1))|(scan==(j+1))]) for i in range(0, len(set(scan))) for j in range(0, i)]
    except:
        run = datatrt.get_subject_info()['run']
        fingerprintcache_pair["reho"] = [figurelib.fingerprint(subjects[(run==(i+1))|(run==(j+1))], datatrt.get_rehos()[(run==(i+1))|(run==(j+1))]) for i in range(0, len(set(run))) for j in range(0, i)]
    
    fplabels = ["ar1", "chance", "reho"]
    barheights = [np.mean(fingerprintcache_pair[k]) if k != "" else 0 for k in fplabels]
    ax.barh(range(0, len(fplabels)), barheights, color=['r' if k in ['ar1'] else (.5, .5, .5) if 'bc' in k else 'k' for k in fplabels], clip_on=False)
    points = np.asarray([(k, fpv) for k in range(0, len(fplabels)) for fpv in fingerprintcache_pair[fplabels[k]]])
    ax.scatter(points[:,1], points[:,0]+np.random.uniform(-.3, .3, len(points[:,0])), s=1, c='#555555', zorder=10, clip_on=False)
    
    ax.set_xlim(0, .6)
    ax.set_xticks([0, .4, .8])
    ax.invert_yaxis()
    sns.despine(ax=ax)
    ax.set_yticks([])
    ax.set_ylim(-.5, len(fplabels)-.5)
    ax.invert_yaxis()
    ax.set_xlabel("Subject identification rate")
    ax.set_title("ReHo fingerprinting")
    
    for i in reversed(range(0, len(fplabels))):
        if fplabels[i] == "": continue
        c.add_text(figurelib.names_for_stuff[fplabels[i]], Point(0, i, "finger"+sfx)+Vector(-.1, 0, "cm"), horizontalalignment="right", verticalalignment="center", size=5)
    
    c.add_text(name, Point(-.55, 1.7, "axis_region_corr"+sfx), weight="bold", size=8, ha="left")



c.add_figure_labels([("a", "region_corr_hcp"),
                     ("b", "subject_corr_hcp", Vector(-5, 0, "mm")),
                     ("c", "finger_hcp", Vector(-5, 0, "mm")),
                     ("d", "region_corr_trt"),
                     ("e", "subject_corr_trt", Vector(-5, 0, "mm")),
                     ("f", "finger_trt", Vector(-5, 0, "mm")),
                     ], size=8)

c.save("figure1-supp-reho.pdf")


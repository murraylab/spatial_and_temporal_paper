import datasets
from cand import Canvas, Vector, Point
from viz import corplot
import models
import seaborn as sns
import pandas
import numpy as np
import scipy
import wbplot
import util
from figurelib import corplot, names_for_stuff
import cdatasets


dsets = [cdatasets.HCP1200(0), cdatasets.HCP1200(0, gsr=True), cdatasets.TRT(), cdatasets.CamCanFiltered()]
dsets_clinical = [cdatasets.Join([cdatasets.LSD("LSD", "early", gsr=True),
                                  cdatasets.LSD("LSD", "late", gsr=True),
                                  cdatasets.LSD("LSD+Ket", "early", gsr=True),
                                  cdatasets.LSD("LSD+Ket", "late", gsr=True),
                                  cdatasets.LSD("Control", "early", gsr=True),
                                  cdatasets.LSD("Control", "late", gsr=True)], name="LSDall"),
                  cdatasets.Join([cdatasets.Psilocybin("Psilocybin", "middle", gsr=True),
                                  cdatasets.Psilocybin("Psilocybin", "late", gsr=True),
                                  cdatasets.Psilocybin("Control", "middle", gsr=True),
                                  cdatasets.Psilocybin("Control", "late", gsr=True)], name="PsiAll")


]

c = Canvas(7.2, 7.5, "in")
c.set_font("Nimbus Sans", ticksize=5, size=6)

# c.add_axis("testretest_motion", Point(.65, 3.9, "in"), Point(.65+.6, 3.9+.6, "in"))
fmt = dict(showr2=False, markersize=1)

_axis_names = ["direct_lmbda", "direct_floor", "meanar1", "parcel_size", "age"]
axis_names = [ds.name+n for ds in dsets+dsets_clinical for n in _axis_names]
c.add_grid(axis_names, 6, Point(.65, .45, "in"), Point(6.8, 6.8, "in"), size=Vector(.6, .6, "in"))


for ds in dsets+dsets_clinical:
    if ds == dsets_clinical[0]:
        color = [sns.color_palette("Paired")[num] for num in ds.get_subject_info()['dataset']]
        alpha=.8
    elif ds == dsets_clinical[1]:
        color = [sns.color_palette("Paired")[num+6] for num in ds.get_subject_info()['dataset']]
        alpha=.8
    elif ds == dsets[2]:
        alpha=.6
        color = (0,0,0)
    else:
        color = (0, 0, 0)
        alpha=.2
    ax = c.ax(ds.name+"parcel_size")
    corplot(ds.get_parcelareas(), np.mean(ds.get_ar1s(), axis=0), "Parcel size", names_for_stuff["regionalar1"], ax=ax, title="Effect of parcel\nsize on regional TA-$\\Delta_1$", showr2=False, markersize=.5, color=(0,0,0), negsig=True, alpha=.8)
    
    ax = c.ax(ds.name+"meanar1")
    corplot(ds.get_movement(), np.mean(ds.get_ar1s(), axis=1), "Motion", names_for_stuff["meanar1"], ax=ax, title="Effect of head\nmotion on global TA-$\\Delta_1$", **fmt, negsig=True, color=color, alpha=alpha, rasterized=True)
    
    ax = c.ax(ds.name+"direct_lmbda")
    if 'camcan' in ds.name:
        corplot(ds.get_movement(), np.log(ds.get_lmbda()), "Motion", names_for_stuff["loglmbda"], ax=ax, title="Effect of head\nmotion on SA-λ", **fmt, negsig=True, color=color, alpha=alpha, rasterized=True)
    else:
        corplot(ds.get_movement(), ds.get_lmbda(), "Motion", names_for_stuff["lmbda"], ax=ax, title="Effect of head\nmotion on SA-λ", **fmt, negsig=True, color=color, alpha=alpha, rasterized=True)
    
    ax = c.ax(ds.name+"direct_floor")
    corplot(ds.get_movement(), ds.get_floor(), "Motion", names_for_stuff["floor"], ax=ax, title="Effect of head\nmotion on SA-∞", **fmt, negsig=True, color=color, alpha=alpha, rasterized=True)
    
    ax = c.ax(ds.name+"age")
    corplot(ds.get_subject_info()["Age_in_Yrs" if "hcp" in ds.name else "Age" if ds in dsets_clinical else "age"], ds.get_movement(), "Age", "Motion", ax=ax, title="Effect of age on\nhead motion", **fmt, negsig=True, color=color, alpha=alpha, rasterized=True)

for d in dsets+dsets_clinical:
    for a in _axis_names:
        if d != dsets[0]:
            c.ax(d.name+a).set_title("")
        if d != dsets_clinical[-1]:
            c.ax(d.name+a).set_xlabel("")
        #if a != _axis_names[0]:
        #    c.ax(d+a).set_ylabel("")

c.add_figure_labels(list(zip("abcde", axis_names, [Vector(-.4, .1, "cm")]*6)), size=8)

section_label_offset = Vector(-.5, .1, "in")
initial_label_offset = Vector(0, .2, "in")
c.add_text("HCP:", initial_label_offset+section_label_offset+Point(0, 1, "axis_hcp12000direct_lmbda"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("HCP-GSR:", section_label_offset+Point(0, 1, "axis_hcp12000gsrdirect_lmbda"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("Yale Test-Retest:", section_label_offset+Point(0, 1, "axis_trtdirect_lmbda"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("Cam-CAN:", section_label_offset+Point(0, 1, "axis_camcanfilteredRestAALdirect_lmbda"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("LSD Dataset:", section_label_offset+Point(0, 1, "axis_LSDalldirect_lmbda"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")
c.add_text("Psilocybin Dataset:", section_label_offset+Point(0, 1, "axis_PsiAlldirect_lmbda"), size=7, weight="bold", horizontalalignment="left", verticalalignment="bottom")

def make_legend(names, loc, pal):
    for i in range(0, len(names)):
        c.add_text(names[i], loc+Vector(.15, -.085*i-.045, "in"), va="bottom", ha="left", size=5)
        c.add_marker(loc+Vector(0, -.085*i, "in"), c=pal[i*2], markersize=3, marker='o')
        c.add_marker(loc+Vector(.10, -.085*i, "in"), c=pal[i*2+1], markersize=3, marker='o')
    c.add_text("Early", loc+Vector(-.03, .03, "in"), va="bottom", size=5)
    c.add_text("Late", loc+Vector(.12, .03, "in"), va="bottom", size=5)

make_legend(["LSD", "LSD+Ket", "Control"], Point(1.0, 1.1, "axis_LSDalldirect_lmbda"), sns.color_palette("Paired"))
make_legend(["Psilocybin", "Control"], Point(1.0, 1.0, "axis_PsiAlldirect_lmbda"), sns.color_palette("Paired")[6:])

c.save("figure1-supp-movement.pdf")
c.show()

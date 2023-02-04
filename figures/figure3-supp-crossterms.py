from cand import Canvas, Vector, Point
import seaborn as sns
import pandas
import numpy as np
from figurelib import COLOR_CYCLE, names_for_stuff

c = Canvas(3.5, 3.0, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)

betzelfigs = ["betzel_dist", "betzel_neighbors", "betzel_dist_cross", "betzel_neighbors_cross"]
c.add_grid(betzelfigs, 2, Point(.5, .5, "in"), Point(2.9, 2.6, "in"), size=Vector(.7, .7, "in"))

titlestyle = {"weight": "bold", "size": 7}

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
                    ], size=8)





c.save("figure3-supp-crossterms.pdf")
#c.show()

from canvas import Canvas, Vector, Point
import seaborn as sns
import pandas
import numpy as np

c = Canvas(3.5, 1.6, "in", fontsize=6, fontsize_ticks=5)

betzelfigs = ["betzel_dist", "betzel_neighbors"]
c.add_grid(betzelfigs, 1, Point(.5, .5, "in"), Point(2.9, 1.5, "in"), size=Vector(.7, .7, "in"))


#################### Betzel ####################

ax = c.ax("betzel_dist")
df = pandas.read_pickle("line_dist.pandas.pkl")
df_group = df.groupby(['eta', 'gamma'])['lmbda'].agg(['mean', 'sem']).reset_index().sort_values('eta')
gammas = list(sorted(set(df_group['gamma'])))
for i,gamma in enumerate(gammas):
    df_gamma = df_group.query(f'gamma == {gamma}')
    ax.errorbar(x=df_gamma['eta'], y=df_gamma['mean'], yerr=df_gamma['sem'], color=sns.color_palette()[i], elinewidth=.75)

sns.despine(ax=ax)
ax.set_xlabel("EC distance param")
ax.set_ylabel("Mean SA")
c.add_legend(Point(1, .8, "axis_betzel_dist"),
             list(zip([f"{g}" for g in gammas], [{"color": c} for c in sns.color_palette()])),
             sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
c.add_text("EC cluster param", Point(1.2, 1, "axis_betzel_dist"))

ax = c.ax("betzel_neighbors")
df = pandas.read_pickle("line_nbrs.pandas.pkl")
df_group = df.groupby(['eta', 'gamma'])['ar1'].agg(['mean', 'sem']).reset_index().sort_values('gamma')
etas = list(sorted(set(df_group['eta'])))
for i,eta in enumerate(etas):
    df_eta = df_group.query(f'eta == {eta}')
    ax.errorbar(x=df_eta['gamma'], y=df_eta['mean'], yerr=df_eta['sem'], color=sns.color_palette()[i], elinewidth=.75)

sns.despine(ax=ax)
ax.set_xlabel("EC cluster param")
ax.set_ylabel("Mean TA")
c.add_legend(Point(1.1, .9, "axis_betzel_neighbors"),
             list(zip([f"{e}" for e in etas], [{"color": c} for c in sns.color_palette()])),
             sym_width=Vector(1.5, 0, "Msize"), line_spacing=Vector(0, 1.4, "Msize"))
c.add_text("EC distance param", Point(1.25, 1.1, "axis_betzel_neighbors"))

c.add_figure_labels([("a", "betzel_dist"),
                     ("b", "betzel_neighbors"),
                    ], size=8)

c.save("figure3.pdf")
#c.show()

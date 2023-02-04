from cand import Canvas, Vector, Point
import seaborn as sns
import numpy as np
import scipy
import util
from scipy.ndimage.filters import gaussian_filter
from figurelib import MODEL_PAL, short_names_for_stuff, names_for_stuff
import models
import os.path
import datasets
import fa2
import networkx
import pandas

# df_params = pandas.read_pickle("df_fitparams.pandas.pkl").query(f'dataset == "hcp0" and loss in ["eigsL2", "none"]')

# fn = f"_f2allmodels_cache.pkl"
# if os.path.exists(fn):
#     cache = util.pload(fn)
# else:
#     cache = {}
#     # Data
#     i = 3
#     distance_matrix = datasets.get_hcp_distances()
#     data_timeseries = datasets.get_hcp_timeseries()[i]
#     cache['data_matrix'] = datasets.get_hcp_matrices()[i]
#     TR = .72
#     cache['data_ar1s'] = util.get_ar1s(data_timeseries)
#     cache['data_thresh'] = util.threshold_matrix(cache['data_matrix'], .05)
    
#     # Colorfull
#     cache['colorfull_matrix'] = models.Model_Colorfull.generate(distance_matrix, ar1vals=cache['data_ar1s'], seed=100+i, num_timepoints=len(data_timeseries[0]), TR=TR, params=dict(df_params.query('model == "Colorfull"').iloc[i][['lmbda', 'floor']]))
#     cache['colorfull_thresh'] = util.threshold_matrix(cache['colorfull_matrix'], .05)
#     G = networkx.from_numpy_array(cache['colorfull_thresh'])
#     cache['colorfull_layout'] = fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    
#     # Timeonly
#     cache['timeonly_matrix'] = models.Model_Colorless_Timeonly.generate(distance_matrix, ar1vals=cache['data_ar1s'], seed=100+i, num_timepoints=len(data_timeseries[0]), TR=TR, params={})
#     cache['timeonly_thresh'] = util.threshold_matrix(cache['timeonly_matrix'], .05)
#     G = networkx.from_numpy_array(cache['timeonly_thresh'])
#     cache['timeonly_layout'] = fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    
#     # Spaceonly
#     cache['spaceonly_matrix'] = models.Model_Spaceonly.generate(distance_matrix, seed=100+i, num_timepoints=len(data_timeseries[0]), params=dict(df_params.query('model == "Spaceonly"').iloc[i][['lmbda']]))
#     cache['spaceonly_thresh'] = util.threshold_matrix(cache['spaceonly_matrix'], .05)
#     G = networkx.from_numpy_array(cache['spaceonly_thresh'])
#     cache['spaceonly_layout'] = fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    
#     # Phase scramble model
#     phase_tss = util.phase_randomize(data_timeseries)
#     cache['phase_matrix'] = util.correlation_matrix_pearson(phase_tss)
#     cache['phase_thresh'] = util.threshold_matrix(cache['phase_matrix'], .05)
#     G = networkx.from_numpy_array(cache['phase_thresh'])
#     cache['phase_layout'] = fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    
#     # Eigensurrogate model
#     cache['eigen_matrix'] = models.Model_Eigensurrogate.generate(cache['data_matrix'])
#     cache['eigen_thresh'] = util.threshold_matrix(cache['eigen_matrix'], .05)
#     G = networkx.from_numpy_array(cache['eigen_thresh'])
#     cache['eigen_layout'] = fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    
#     # Zalesky model
#     # cache['zalesky_matrix'] = models.Model_Zalesky2012.generate(cache['data_matrix'], seed=0)
#     # cache['zalesky_thresh'] = util.threshold_matrix(cache['zalesky_matrix'], .05)
    
#     # Degree-preserving randomization
#     cache['degreerand_thresh'] = models.Model_DegreeRandomize.generate(cache['data_thresh'])
#     G = networkx.from_numpy_array(cache['degreerand_thresh'])
#     cache['degreerand_layout'] = fa2.ForceAtlas2(gravity=50, scalingRatio=3).forceatlas2_networkx_layout(G, pos=None, iterations=2000)
#     util.psave(fn, cache)

def make_network_plot(sfx, cachename):
    pass
    # pos = Point(0, 0, "diagram"+sfx) + Vector(4, -.2, "in")
    # c.add_axis("network"+sfx, pos, pos+Vector(1.4, 1.4, "in"))
    # ax = c.ax("network"+sfx)
    # G = networkx.from_numpy_array(cache[cachename+'_thresh'])
    # networkx.draw_networkx_nodes(G, cache[cachename+'_layout'], node_size=2, node_color="k", alpha=0.8, ax=ax, linewidths=0)
    # networkx.draw_networkx_edges(G, cache[cachename+'_layout'], edge_color=MODEL_PAL[sfx[1:]], alpha=.05, ax=ax)
    # ax.axis("off")



c = Canvas(6.0, 6.65, "in")
c.set_font("Nimbus Sans", size=6, ticksize=5)

POS = [Point(.16, 5.4, "in"), Point(3.16, 5.4, "in"), Point(.16, 4.1, "in"), Point(3.16, 4.1, "in"), Point(.16, 2.8, "in"), Point(.16, 1.5, "in"), Point(.16, .2, "in")]
SCALE = Vector(.528, .528, "in")

#################### Spatial only ####################

arrowstyle = dict(lw=1, arrowstyle="->,head_width=2,head_length=4")

name = "Spaceonly"

sfx = "_"+name

c.add_unit("diagram"+sfx, SCALE, POS[0])

c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
tss = np.random.RandomState(2).randn(8, 50)

axis_names = ["dia_timeseries"+sfx, "dia_spatial"+sfx, "dia_output"+sfx]
c.add_grid(axis_names, 1, Point(.25, .25), Point(4.25, 1.25), size=Vector(1, 1))

# Make the timeseries
ax = c.ax("dia_timeseries"+sfx)
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')

# c.add_text(f"i=k", Point(0, 0, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(f"i=1", Point(0, tss.shape[0]-2, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(". . .", Point(0, (tss.shape[0]-1)/2, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center", rotation=90, size=10)
# c.add_text(f"i=0", Point(0, tss.shape[0]-1, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")

ax.axis('off')

# Show spatial correlation
ax = c.ax("dia_spatial"+sfx)
ax.imshow(gaussian_filter(np.random.RandomState(4).randn(100,100), sigma=10), cmap="gray_r")
ax.axis("off")


# Final timeseries
ax = c.ax("dia_output"+sfx)
ax.cla()
tss_spatial = scipy.linalg.sqrtm(np.eye(8) + (1-np.eye(8))*.6) @ tss
for i in range(0, tss.shape[0]):
    ax.plot(i+tss_spatial[i]*.3, clip_on=False, linewidth=.5, c='k')

ax.axis('off')

# Add titles
title_height = Vector(0, .05)
c.add_text("Timeseries without TA", Point(.5, 0, "axis_dia_timeseries"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Spatial embedding", Point(.5, 0, "axis_dia_spatial"+sfx) - title_height, size=5, verticalalignment="top")


c.add_text("Gaussian timeseries", Point(.5, 1, "axis_dia_timeseries"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("SA-λ and SA-∞", Point(.5, 1, "axis_dia_spatial"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate timeseries", Point(.5, 1, "axis_dia_output"+sfx) + title_height, size=6, verticalalignment="bottom")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_arrow(Point(1, .5, "axis_dia_timeseries"+sfx)+spacing, Point(0, .5, "axis_dia_spatial"+sfx)-spacing, **arrowstyle)
c.add_arrow(Point(1, .5, "axis_dia_spatial"+sfx)+spacing, Point(0, .5, "axis_dia_output"+sfx)-spacing, **arrowstyle)

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(4.5, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text(names_for_stuff["Spaceonly"], Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")


# Plot of nodes and edges
make_network_plot(sfx, "spaceonly")

#################### Colorfull diagram ####################

name = "Colorfull"

sfx = "_"+name

c.add_unit("diagram"+sfx, SCALE, POS[2])
c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
import models
dist = (1-np.eye(8)) * 1000
tss = models.Model_Colorfull.generate_timeseries(dist, ar1vals=np.random.RandomState(14).rand(8), num_timepoints=1100, TR=.72, params={"lmbda": 100, "floor": 0}, seed=0)[:,100:150]

axis_names = ["dia_timeseries"+sfx, "dia_spatial"+sfx, "dia_output"+sfx]
c.add_grid(axis_names, 1, Point(.25, .25), Point(4.25, 1.25), size=Vector(1, 1))

# Make the timeseries
ax = c.ax("dia_timeseries"+sfx)
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')

# c.add_text(f"i=k", Point(0, 0, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(f"i=1", Point(0, tss.shape[0]-2, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(". . .", Point(0, (tss.shape[0]-1)/2, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center", rotation=90, size=10)
# c.add_text(f"i=0", Point(0, tss.shape[0]-1, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")

ax.axis('off')

# Show spatial correlation
from scipy.ndimage.filters import gaussian_filter
ax = c.ax("dia_spatial"+sfx)
ax.imshow(gaussian_filter(np.random.RandomState(12).randn(100,100), sigma=10), cmap="gray_r")
ax.axis("off")


# Final timeseries
import models
ax = c.ax("dia_output"+sfx)
ax.cla()
tss_spatial = scipy.linalg.sqrtm(np.eye(8) + (1-np.eye(8))*.6) @ tss
for i in range(0, tss.shape[0]):
    ax.plot(i+tss_spatial[i]*.3, clip_on=False, linewidth=.5, c='k')

ax.axis('off')

# Add titles
title_height = Vector(0, .05)
c.add_text("Filtered 1/f$^\\alpha$ noise with\nexponents matching TA", Point(.5, 0, "axis_dia_timeseries"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Correlate according\nto SA-λ and SA-∞", Point(.5, 0, "axis_dia_spatial"+sfx) - title_height, size=5, verticalalignment="top")


c.add_text("Smooth timeseries", Point(.5, 1, "axis_dia_timeseries"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Spatial embedding", Point(.5, 1, "axis_dia_spatial"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate timeseries", Point(.5, 1, "axis_dia_output"+sfx) + title_height, size=6, verticalalignment="bottom")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_text("×", Point(1, .5, "axis_dia_timeseries"+sfx) | Point(0, .5, "axis_dia_spatial"+sfx), size=16)
c.add_arrow(Point(1, .5, "axis_dia_spatial"+sfx)+spacing, Point(0, .5, "axis_dia_output"+sfx)-spacing, **arrowstyle)

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(4.5, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text(names_for_stuff["Colorsurrogate"], Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")


make_network_plot(sfx, "colorfull")

#################### Time only ####################

name = "ColorlessTimeonly"

sfx = "_"+name

c.add_unit("diagram"+sfx, SCALE, POS[1])
c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
tss = np.cumsum(np.random.RandomState(1).randn(8, 50)*.8, axis=1)

axis_names = ["dia_timeseries"+sfx, "dia_whitenoise"+sfx, "dia_output"+sfx]
c.add_grid(axis_names, 1, Point(.25, .25), Point(4.25, 1.25), size=Vector(1, 1))

# Make the timeseries
ax = c.ax("dia_timeseries"+sfx)
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')

# c.add_text(f"i=k", Point(0, 0, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(f"i=1", Point(0, tss.shape[0]-2, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(". . .", Point(0, (tss.shape[0]-1)/2, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center", rotation=90, size=10)
# c.add_text(f"i=0", Point(0, tss.shape[0]-1, "dia_timeseries"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")

ax.axis('off')


# Show adding random gaussian noise
ax = c.ax("dia_whitenoise"+sfx)
tss_white = np.random.RandomState(5).random((8,1)) * np.random.RandomState(2).randn(8, 50) * 1.5
for i in range(0, tss_white.shape[0]):
    ax.plot(tss_white[i]*.3+i, clip_on=False, linewidth=.5, c='k')

ax.axis('off')


# Final timeseries
import models
ax = c.ax("dia_output"+sfx)
ax.cla()
for i in range(0, tss.shape[0]):
    ax.plot(i+util.timeseries_from_spectrum(util.make_noisy_spectrum(50, .72, 2, .01, .6), i)/8, clip_on=False, linewidth=.5, c='k')

ax.axis('off')

# Add titles
title_height = Vector(0, .05)
c.add_text("Filtered random\nwalk timeseries", Point(.5, 0, "axis_dia_timeseries"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Noise w/ heterogeneous\nvariance, matches TA-$\Delta_1$", Point(.5, 0, "axis_dia_whitenoise"+sfx) - title_height, size=5, verticalalignment="top")


c.add_text("Smooth timeseries", Point(.5, 1, "axis_dia_timeseries"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Regional noise", Point(.5, 1, "axis_dia_whitenoise"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate timeseries", Point(.5, 1, "axis_dia_output"+sfx) + title_height, size=6, verticalalignment="bottom")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_text("+", Point(1, .5, "axis_dia_timeseries"+sfx) | Point(0, .5, "axis_dia_whitenoise"+sfx), size=16)
c.add_arrow(Point(1, .5, "axis_dia_whitenoise"+sfx)+spacing, Point(0, .5, "axis_dia_output"+sfx)-spacing, **arrowstyle)

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(4.5, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text(names_for_stuff["ColorlessTimeonly"], Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")


make_network_plot(sfx, "timeonly")


#################### Eigensurrogate ####################

name = "eigen"

sfx = "_"+name


c.add_unit("diagram"+sfx, SCALE, POS[5])

c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
tss = np.random.RandomState(2).randn(8, 50)

axis_names = ["cmat"+sfx, "equation"+sfx, "flow"+sfx, "cm2"+sfx]
c.add_grid(axis_names, 1, Point(.25, .25), Point(5.75, 1.25), size=Vector(1, 1))
garbage_pos = Point(1.8, .15, "axis_equation"+sfx)
dice_pos = Point(1.2, .85, "axis_equation"+sfx)

c.ax("equation"+sfx).axis("off")
c.add_text("$Q \\Lambda Q^{-1}$", Point(.5, .5, "axis_equation"+sfx), size=12, weight="bold")

# Make the images
c.add_image("garbage.png", garbage_pos, height=Vector(0, .35))
c.add_image("dice.png", dice_pos, height=Vector(0, .35))

# Make the arrows to/from the images
c.ax("flow"+sfx).axis("off")
spacing_img = Vector(.2, 0)
spacing_text = Vector(0, .02)
arrow_start = Point(.2, .3, "axis_equation"+sfx)
# c.add_line(arrow_start, arrow_start >> garbage_pos, linewidth=1, color='k')
# c.add_arrow(arrow_start >> garbage_pos, garbage_pos-spacing_img, color='k', **arrowstyle)
c.add_text("Eigenvectors", (arrow_start >> garbage_pos)+spacing_text+Vector(.15, 0), va="bottom", ha="left", size=5)
c.add_arrow(arrow_start, garbage_pos-spacing_img, color='k', connectionstyle="angle,angleA=90,angleB=0", **arrowstyle)


c.add_arrow(Point(1, .5, "axis_equation"+sfx), Point(0, .5, "axis_cm2"+sfx),  **arrowstyle)
c.add_text("Eigenvalues", spacing_text+(Point(1, .5, "axis_equation"+sfx) | Point(0, .5, "axis_cm2"+sfx)), va="bottom", size=5)

c.add_arrow(dice_pos+spacing_img, (Point(0, 0, "axis_cm2"+sfx) >> dice_pos), **arrowstyle)
c.add_text("Random eigenvectors", spacing_text+(dice_pos | (Point(0, 0, "axis_cm2"+sfx) >> dice_pos)), va="bottom", size=5)


# Make the correlation matrix
ax = c.ax("cmat"+sfx)
cm = np.corrcoef(np.random.RandomState(0).randn(8, 10))
ax.imshow(cm)
ax.axis("off")

# Make the other correlation matrix matrix
ax = c.ax("cm2"+sfx)
cm = np.corrcoef(np.random.RandomState(1).randn(8, 10))
ax.imshow(cm)
ax.axis("off")

# Add titles
title_height = Vector(0, .05)

c.add_text("FC matrix", Point(.5, 1, "axis_cmat"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Eigendecomposition", Point(.5, 1, "axis_equation"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate FC matrix", Point(.5, 1, "axis_cm2"+sfx) + title_height, size=6, verticalalignment="bottom")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_text("=", Point(.5, 1, "axis_cmat"+sfx) | Point(.5, 0, "axis_equation"+sfx), size=16)

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(6, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text("Eigensurrogate", Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")


make_network_plot(sfx, "eigen")


#################### Zalesky ####################

name = "zalesky2012"

sfx = "_"+name


c.add_unit("diagram"+sfx, SCALE, POS[6])

c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
tss = np.random.RandomState(2).randn(4, 50)

axis_names1 = ["cmat"+sfx, None, None, "cm2"+sfx]
axis_names2 = [None, "timeseries"+sfx, "global"+sfx, None]
c.add_grid(axis_names1, 1, Point(.25, .25), Point(5.75, 1.25), size=Vector(1, 1))
c.add_grid(axis_names2, 1, Point(.25, .25), Point(5.75, .75), size=Vector(1, .5))

tss2 = np.random.RandomState(2).randn(4, 50)

# Make the timeseries
ax = c.ax("timeseries"+sfx)
ax.axis("off")
for i in range(0, tss.shape[0]):
    ax.plot(np.linspace(0, 29, 30), tss[i,:30]*.3+i, clip_on=False, linewidth=.5, c='k')
    ax.plot(np.linspace(30, 49, 20), tss[i,30:]*.3+i, clip_on=False, linewidth=.5, c=(.6, .6, .6))


ax = c.ax("global"+sfx)
ax.axis("off")
gs = np.random.RandomState(2).randn(50)
ax.plot(np.linspace(0, 29, 30), gs[:30], clip_on=False, linewidth=.5, c='k')
ax.plot(np.linspace(30, 49, 20), gs[30:], clip_on=False, linewidth=.5, c=(.6, .6, .6))
ax.set_ylim(-3, 3)



# # Make the arrows to/from the images
# c.ax("flow"+sfx).axis("off")
# spacing_img = Vector(.2, 0)
# spacing_text = Vector(0, .02)
# arrow_start = Point(.2, .3, "axis_equation"+sfx)
# c.add_line(arrow_start, arrow_start >> garbage_pos, linewidth=1, color='k')
# c.add_arrow(arrow_start >> garbage_pos, garbage_pos-spacing_img, color='k', **arrowstyle)
# c.add_text("Eigenvectors", (arrow_start >> garbage_pos)+spacing_text+Vector(.15, 0), va="bottom", ha="left", size=5)


# c.add_arrow(Point(1, .5, "axis_equation"+sfx), Point(0, .5, "axis_cm2"+sfx),  **arrowstyle)
# c.add_text("Eigenvalues", spacing_text+(Point(1, .5, "axis_equation"+sfx) | Point(0, .5, "axis_cm2"+sfx)), va="bottom", size=5)

# c.add_arrow(dice_pos+spacing_img, (Point(0, 0, "axis_cm2"+sfx) >> dice_pos), **arrowstyle)
# c.add_text("Random eigenvectors", spacing_text+(dice_pos | (Point(0, 0, "axis_cm2"+sfx) >> dice_pos)), va="bottom", size=5)


# Make the correlation matrix
ax = c.ax("cmat"+sfx)
cm = np.corrcoef(np.random.RandomState(0).randn(8, 10))
ax.imshow(cm)
ax.axis("off")

# Make the other correlation matrix matrix
ax = c.ax("cm2"+sfx)
cm = np.corrcoef(np.random.RandomState(1).randn(8, 10))
ax.imshow(cm)
ax.axis("off")

# Add titles
title_height = Vector(0, .05)

c.add_text("FC matrix", Point(.5, 1, "axis_cmat"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Gaussian timeseries", Point(.5, 1, ("axis_timeseries"+sfx, "axis_cmat"+sfx)) + title_height, size=6, verticalalignment="bottom")
c.add_text("Global signal", Point(.5, 1, ("axis_global"+sfx, "axis_cmat"+sfx)) + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate FC matrix", Point(.5, 1, "axis_cm2"+sfx) + title_height, size=6, verticalalignment="bottom")

c.add_text("Original data", Point(.5, 0, "axis_cmat"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Surrogate data", Point(.5, 0, "axis_cm2"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Match timeseries\nlength to Var-FC", Point(.5, 0, "axis_timeseries"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Match global signal\nstrength to Mean-FC", Point(.5, 0, "axis_global"+sfx) - title_height, size=5, verticalalignment="top")


# Add arrows
spacing = Vector(.05, 0)
c.show() # Thre is some bug in either CanD or matplotlib that requires this line.  Don't know why but no time to debug right now.
c.add_text("+", Point(1, .5, "axis_timeseries"+sfx) | Point(0, .5, "axis_global"+sfx), size=16)
# c.add_arrow(Point(1, .9, "axis_cmat"+sfx)+spacing, Point(0, .9, "axis_cm2"+sfx)-spacing, **arrowstyle)
c.add_arrow(Point(1, .5, "axis_global"+sfx)+spacing, Point(0, .5, ("axis_cm2"+sfx, "axis_global"+sfx))-spacing, **arrowstyle)

c.add_arrow(Point(0, 5, "timeseries"+sfx), Point(30, 5, "timeseries"+sfx), shrinkA=0, shrinkB=0, arrowstyle="|-|,widthA=2,widthB=2", color='k', linewidth=1)
c.add_text("# points", (Point(0, 5, "timeseries"+sfx) | Point(30, 5, "timeseries"+sfx)) + Vector(0, .1, "cm"), ha="center", va="bottom")
c.add_arrow(Point(1, 5, ("axis_cmat"+sfx, "timeseries"+sfx))+spacing, Point(0, 5, ("axis_timeseries"+sfx, "timeseries"+sfx)), **arrowstyle)
c.add_text("Var-FC", Point(1, 5, ("axis_cmat"+sfx, "timeseries"+sfx))+spacing | Point(0, 5, ("axis_timeseries"+sfx, "timeseries"+sfx)) + Vector(0, .3, "cm"), size=5)
arrow_start = Point(1, .88, "axis_cmat"+sfx)+spacing
arrow_end = Point(.5, .8, "axis_global"+sfx)
# c.add_line(arrow_start, arrow_start << arrow_end, linewidth=1, color='k', solid_capstyle="projecting")
c.add_arrow(arrow_start, arrow_end, color='k', capstyle="projecting", connectionstyle="angle,angleA=0,angleB=90", **arrowstyle)
#c.add_text("Mean-FC", (arrow_start << arrow_end) | arrow_end + Vector(.15, 0, "cm"), ha="left", size=5)
c.add_text("Mean-FC", arrow_start + Vector(0.01, 0.1, "cm"), ha="left", size=5)


# Box it all in with the correct color
c.add_box(Point(0, 0), Point(6, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text(names_for_stuff["zalesky2012"], Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")


make_network_plot(sfx, "eigen")


#################### Phase randomize ####################

name = "phase"

sfx = "_"+name


c.add_unit("diagram"+sfx, SCALE, POS[4])

c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
tss = np.random.RandomState(2).randn(8, 50)

axis_names = ["timeseries1"+sfx, "equation"+sfx, "flow"+sfx, "timeseries2"+sfx]
c.add_grid(axis_names, 1, Point(.25, .25), Point(5.75, 1.25), size=Vector(1, 1))
garbage_pos = Point(1.8, .15, "axis_equation"+sfx)
dice_pos = Point(1.2, .85, "axis_equation"+sfx)

c.ax("equation"+sfx).axis("off")
c.add_text("$\\sum f(x) e^{-i \\omega k t}$", Point(.5, .5, "axis_equation"+sfx), size=10, weight="bold")

# Make the images
c.add_image("garbage.png", garbage_pos, height=Vector(0, .35))
c.add_image("dice.png", dice_pos, height=Vector(0, .35))

# Make the arrows to/from the images
c.ax("flow"+sfx).axis("off")
spacing_img = Vector(.2, 0)
spacing_text = Vector(0, .02)
arrow_start = Point(.2, .3, "axis_equation"+sfx)
c.add_arrow(arrow_start, garbage_pos-spacing_img, color='k', connectionstyle="angle,angleA=90,angleB=0", **arrowstyle)
c.add_text("Phases", (arrow_start >> garbage_pos)+spacing_text+Vector(.15, 0), va="bottom", ha="left", size=5)

c.add_arrow(Point(1.15, .5, "axis_equation"+sfx), Point(0, .5, "axis_timeseries2"+sfx), **arrowstyle)
c.add_text("Amplitudes", spacing_text+(Point(1, .5, "axis_equation"+sfx) | Point(0, .5, "axis_timeseries2"+sfx)), va="bottom", size=5)

c.add_arrow(dice_pos+spacing_img, (Point(0, 0, "axis_timeseries2"+sfx) >> dice_pos), **arrowstyle)
c.add_text("Random phases", spacing_text+(dice_pos | (Point(0, 0, "axis_timeseries2"+sfx) >> dice_pos)), va="bottom", size=5)


# Make the correlation matrix
ax = c.ax("timeseries1"+sfx)
tss = np.random.RandomState(0).randn(8, 50)
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')

# c.add_text(f"i=k", Point(0, 0, "timeseries1"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(f"i=1", Point(0, tss.shape[0]-2, "timeseries1"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")
# c.add_text(". . .", Point(0, (tss.shape[0]-1)/2, "timeseries1"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center", rotation=90, size=10)
# c.add_text(f"i=0", Point(0, tss.shape[0]-1, "timeseries1"+sfx)+Vector(-.1, 0, "in"), horizontalalignment="right", verticalalignment="center")

ax.axis('off')

# Make the other correlation matrix matrix
ax = c.ax("timeseries2"+sfx)
tss = np.random.RandomState(1).randn(8, 50)
for i in range(0, tss.shape[0]):
    ax.plot(tss[i]*.3+i, clip_on=False, linewidth=.5, c='k')


ax.axis('off')

# Add titles
title_height = Vector(0, .05)

c.add_text("BOLD timeseries", Point(.5, 1, "axis_timeseries1"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Fourier transform", Point(.5, 1, "axis_equation"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Surrogate timeseries", Point(.5, 1, "axis_timeseries2"+sfx) + title_height, size=6, verticalalignment="bottom")

c.add_text("Original data", Point(.5, 0, "axis_timeseries1"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Surrogate data", Point(.5, 0, "axis_timeseries2"+sfx) - title_height, size=5, verticalalignment="top")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_arrow(Point(1, .5, "axis_timeseries1"+sfx) + spacing, Point(0, .5, "axis_equation"+sfx) - spacing, **arrowstyle)

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(6, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text(names_for_stuff["phase"], Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")

make_network_plot(sfx, "phase")


#################### Degree-preserving randomization ####################


name = "degreerand"

sfx = "_"+name


c.add_unit("diagram"+sfx, SCALE, POS[3])

c.set_default_unit("diagram"+sfx)

# Set up timeseries we will use for this diagram
tss = np.random.RandomState(2).randn(8, 50)

axis_names = ["graph"+sfx, "graphcut"+sfx, "newgraph"+sfx]
c.add_grid(axis_names, 1, Point(.25, .25), Point(4.25, 1.25), size=Vector(1, 1))


# Draw graph whole
ax = c.ax("graph"+sfx)
base = Point(.5, .5, "axis_graph"+sfx)
unit = Vector(0, .4, "axis_graph"+sfx)
node_loc = [base + unit,
            base + unit @ 72,
            base + unit @ 144,
            base + unit @ 216,
            base + unit @ 288,
            ]

edge_list = [(0,1), (0,3), (0,4),
             (1,2),
             (3,4)
             ]

# Draw edges
for i,j in edge_list:
    c.add_line(node_loc[i], node_loc[j], c='k')

# Draw nodes
text_offset = Vector(0, -.01)
for i in range(0, 5):
    c.add_marker(node_loc[i], marker="o", markersize=6, c='k')
    #c.add_text(str(i), node_loc[i]+text_offset, color='w', size=6, horizontalalignment="center", verticalalignment="center")

ax.axis("off")


# Draw cut graph
ax = c.ax("graphcut"+sfx)
base = Point(.5, .5, "axis_graphcut"+sfx)
node_loc = [base + unit,
            base + unit @ 72,
            base + unit @ 144,
            base + unit @ 216,
            base + unit @ 288,
            ]

# Draw cut edges
for i,j in edge_list:
    a = node_loc[i]
    b = node_loc[j]
    v_atob = c.convert_to_absolute_length(b-a)
    v_atob = v_atob/np.sqrt(v_atob.x**2 + v_atob.y**2)
    c.add_line(a, a+v_atob*.09, c='k')
    c.add_line(b, b-v_atob*.09, c='k')

# Draw nodes
text_offset = Vector(0, -.01)
for i in range(0, 5):
    c.add_marker(node_loc[i], marker="o", markersize=6, c='k')
    #c.add_text(str(i), node_loc[i]+text_offset, color='w', size=6, horizontalalignment="center", verticalalignment="center")


# Let's actually do a scramble because why not.
np.random.seed(0)
for i in range(0, 100):
    ns = np.random.permutation(5)[0:2]
    e1 = edge_list[ns[0]]
    e2 = edge_list[ns[1]]
    if e1[0] == e2[1]:
        continue
    if e1[1] == e2[0]:
        continue
    newe1 = (e1[0], e2[1])
    newe2 = (e1[1], e2[0])
    if newe1 in edge_list or newe2 in edge_list:
        continue
    if (newe1[1], newe1[0]) in edge_list or (newe2[1], newe2[0]) in edge_list:
        continue
    edge_list[ns[0]] = newe1
    edge_list[ns[1]] = newe2
    edge_list.sort()

ax.axis("off")


# Draw scrambled graph
ax = c.ax("newgraph"+sfx)
base = Point(.5, .5, "axis_newgraph"+sfx)
# Could substitute this octagon for a circle, would allow for changing N_timeseries
node_loc = [base + unit,
            base + unit @ 72,
            base + unit @ 144,
            base + unit @ 216,
            base + unit @ 288,
            ]
    
# Draw edges
for i,j in edge_list:
    c.add_line(node_loc[i], node_loc[j], c='k')

# Draw nodes
text_offset = Vector(0, -.01)
for i in range(0, 5):
    c.add_marker(node_loc[i], marker="o", markersize=6, c='k')
    #c.add_text(str(i), node_loc[i]+text_offset, color='w', size=6, horizontalalignment="center", verticalalignment="center")

ax.axis("off")

# Add arrows to connect them
spacing = Vector(.05, 0)
c.add_arrow(Point(1, .5, "axis_graph"+sfx)+spacing, Point(0, .5, "axis_graphcut"+sfx)-spacing, **arrowstyle)
c.add_arrow(Point(1, .5, "axis_graphcut"+sfx)+spacing, Point(0, .5, "axis_newgraph"+sfx)-spacing, **arrowstyle)



scissors_pos = Point(.12, .84, "axis_graphcut"+sfx)
c.add_image("edit-cut-2.png", scissors_pos, height=Vector(0, .27, "in"))


# Add titles
title_height = Vector(0, .05)

c.add_text("FC graph", Point(.5, 1, "axis_graph"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Keep node degree", Point(.5, 1, "axis_graphcut"+sfx) + title_height, size=6, verticalalignment="bottom")
c.add_text("Randomly rewire", Point(.5, 1, "axis_newgraph"+sfx) + title_height, size=6, verticalalignment="bottom")

c.add_text("Original data", Point(.5, 0, "axis_graph"+sfx) - title_height, size=5, verticalalignment="top")
c.add_text("Surrogate data", Point(.5, 0, "axis_newgraph"+sfx) - title_height, size=5, verticalalignment="top")

# Box it all in with the correct color
c.add_box(Point(0, 0), Point(4.5, 1.6), boxstyle="Round", fill=True, alpha=.3, zorder=-10, linewidth=None, color=MODEL_PAL[name])
c.add_text(names_for_stuff["degreerand"], Point(0, 1.6), weight="bold", size=6, horizontalalignment="left")


make_network_plot(sfx, "degreerand")


c.save("figure2-supp-models.pdf")
c.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import util
import seaborn as sns
import pandas
import networkx
import fitting
import bct
import scipy

# toplot can be "cm", "graph", "eig"
def model_gui(model, distance_matrix, lossfunc, subjmat, toplot="cm", *args, **kwargs):
    # This function was updated to display the loss between the current model and some
    # given sample. It now requires two new parameters: a loss function, and a benchmark matrix
    
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)
    #fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, gridspec_kw={})
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    axcolor = 'lightgoldenrodyellow'
    slider_axes = [plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor, visible=False),
                   plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor, visible=False),
                   plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor, visible=False),
                   plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor, visible=False)]
    
    param_names = list(sorted(model.params.keys()))
    sliders = []
    for i,p in enumerate(param_names):
        sliders.append(Slider(slider_axes[i], p, model.params[p][0], model.params[p][1]))
        slider_axes[i].set_visible(True)
    
    metrics = ["meancor", "varcor", "kurtcor", "cluster", "assort", "gefficiency", "lefficiency", "modularity", "transitivity"]
    data_ms = util.graph_metrics_from_cm(subjmat)
    data_ms['transitivity'] *= 50
    data_ms['varcor'] *= 10
    data_thresh = util.threshold_matrix(subjmat)
    data_mss = [(m, data_ms[m], 'data') for m in metrics]
    def update(val):
        params = {k: s.val for k,s in zip(param_names, sliders)}
        mat = model.generate(distance_matrix, params, *args, **kwargs)
        plt.suptitle('loss: %s' % str(lossfunc(subjmat,mat)))
        ax1.cla()
        ax2.cla()
        ax3.cla()
        if toplot == "cm":
            ax1.imshow(util.threshold_matrix(mat), cmap="binary", vmin=0, vmax=1)
        elif toplot == "eig":
            ax1.plot(range(1, mat.shape[0]+1), np.sort(np.real(np.linalg.eig(mat)[0]))[::-1], )
            ax1.plot(range(1, mat.shape[0]+1), np.sort(np.real(np.linalg.eig(subjmat)[0]))[::-1], c='k')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
        elif toplot == "degree":
            ax1.hist(np.sum(util.threshold_matrix(mat), axis=0), alpha=.5)
            ax1.hist(np.sum(data_thresh, axis=0), alpha=.5)
        ax2.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ms = util.graph_metrics_from_cm(mat)
        ms['transitivity'] *= 50
        ms['varcor'] *= 10
        mss = [(m, ms[m], 'model') for m in metrics]
        sns.barplot(x="metric", y="val", hue="version", data=pandas.DataFrame(mss+data_mss, columns=["metric", "val", "version"]), ax=ax3)
        ax3.set_ylim(0, 1)
        fig.canvas.draw_idle()
    for s in sliders:
        s.on_changed(update)
    update(None)
    plt.show()

# `matrices` is a dict: keys are names, vals are correlation matrices
def model_comparison_plot(matrices):
    N_plots = 2
    plt.figure(figsize=(5, (2+len(matrices))*2.5))
    matrix_names = list(sorted(matrices.keys()))
    limit_metric_names = ["meancor", "varcor", "cluster", "assort", "gefficiency", "lefficiency", "modularity"]
    metrics = [(met_name, met_val, mat_name)
                 for mat_name,mat_val in matrices.items()
                 for met_name,met_val in util.graph_metrics_from_cm(mat_val).items()
                   if met_name in limit_metric_names]
    plt.subplot(len(matrices)+3, 1, 1)
    sns.barplot(x="metric", y="val", hue="matrix",
                data=pandas.DataFrame(metrics, columns=["metric", "val", "matrix"]),
                ax=plt.gca(), hue_order=matrix_names)
    plt.xlabel("")
    
    plt.subplot(len(matrices)+3, 1, 2)
    for mn in matrix_names:
        eigs = sorted(np.real(np.linalg.eig(matrices[mn])[0]), reverse=True)
        plt.plot(range(1, len(eigs)+1), eigs)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Log eigs")
    plt.subplot(len(matrices)+3, 1, 3)
    for mn in matrix_names:
        sns.distplot(list(matrices[mn].flat), ax=plt.gca(), hist=False)
    plt.title("Degree histogram")
    for i,(name,cm) in enumerate(matrices.items()):
        plt.subplot(len(matrices)+3, 3, 1+3*(i+3))
        cm_thresh = util.threshold_matrix(cm)
        plt.imshow(cm_thresh, cmap="binary", vmin=0, vmax=1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.ylabel(name)
        plt.subplot(len(matrices)+3, 3, 2+3*(i+3))
        plt.imshow(cm, cmap="RdYlBu_r", vmin=-1, vmax=1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.subplot(len(matrices)+3, 3, 3+3*(i+3))
        networkx.draw(networkx.Graph(cm_thresh), ax=plt.gca(), node_size=5)
    plt.tight_layout()

def gmetrics_scatterplots(matrices):
    # Compare the values of model graph metrics and sample graph metrics
    # across multiple subjects

    matrix_names = list(sorted(matrices.keys()))
    limit_metric_names = ["meancor", "cluster", "assort", "gefficiency", "lefficiency", "modularity"]
    num_subjects = len(matrices[list(matrices.keys())[0]])
    metrics = pandas.DataFrame([(subj_id,met_name,met_val,mat_name)
               for subj_id in range(num_subjects)
               for mat_name, mat_val in matrices.items()
               for met_name, met_val in util.graph_metrics_from_cm(mat_val[subj_id]).items()
               if met_name in limit_metric_names],
                               columns=["subj_id","met_name","met_val","model"])

    def abline(slope, intercept, axes):
        # Plot a line from slope and intercept
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        axes.plot(x_vals, y_vals, '--')
        
    def r_squared_arrays(x, y, rd=False, all=False):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        if rd:
            rsq = round(r_value ** 2, 4)
            p_value = round(p_value, 4)
        else:
            rsq = r_value ** 2
        if all:
            return slope, intercept, r_value, p_value, std_err
        else:
            return rsq, p_value

    cols = int(np.round(len(limit_metric_names) / 3))
    rows = 3
    fig, axes = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axes[row][col].scatter(metrics["met_val"][metrics["met_name"]==limit_metric_names[cols*row+col]][metrics["model"]=="models"],
            metrics["met_val"][metrics["met_name"] == limit_metric_names[cols * row + col]][metrics["model"] == "subjs"])
            abline(1, 0, axes=axes[row][col])
            axes[row][col].set_title(str(limit_metric_names[cols * row + col]) + ', rsq: %s' % r_squared_arrays(x,y,rd=True)[0])
    plt.show()

def diagnostic_plot_on_hcp_brain(vec):
    from mpl_toolkits.mplot3d import Axes3D
    import datasets
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poss = datasets.get_hcp_positions()
    ax.scatter(poss[:,0], poss[:,1], poss[:,2], c=vec)
    plt.show()

def corplot(vals1, vals2, name1, name2, title=None, ax=None, diag=False, color=None):
    if ax is None:
        ax = plt.gca()
    spear = scipy.stats.spearmanr(vals1, vals2)
    pears = scipy.stats.pearsonr(vals1, vals2)[0]
    R2 = pears**2 * np.sign(pears)
    sig = "**" if spear.pvalue < .01 else "*" if spear.pvalue < .05 else ""
    if spear.correlation < 0: sig = ""
    ax.text(.7, .3, f"$r_s$={spear.correlation:.2}{sig}\n$R^2$={R2:.2}", size=7, transform=ax.transAxes)
    if color is None:
        color = "r" if sig == "**" else "b" if sig == "*" else "k"
    ax.scatter(vals1, vals2, marker='o', s=4, c=color)
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
        

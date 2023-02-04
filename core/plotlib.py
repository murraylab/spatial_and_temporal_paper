import seaborn as sns
from canvas import Point, Vector
import numpy as np
import matplotlib.ticker as ticker
import scipy

SCATTER_SIZE = Vector(1.0, 1.0, "inches")
MODEL_COLORS = {"data": 'k', "HRF:graphmetrics": (.5, .8, .5), "HRFNodata:graphmetrics": (.7, 1, .7), "degreerand:none": (.3, .3, .7), "phase:none": (.7, .2, .7), "zalesky:none": (.7, .5, .7)}

# TODO make colors based on HSLuv


# Formatting functions


def metric_name(name):
    names = {"assort": "Assortativity",
             "cluster": "Clustering",
             "gefficiency": "Efficiency",
             "lefficiency": "Local efficiency",
             "modularity": "Modularity",
             "transitivity": "Transitivity",
             "centrality": "Betweenness centrality",
             "degree": "Degree",
             "mean": "GBC",
             "var": "VBC",
             "std": "SBC",
             "tau": "Temporal AC"}
    if name in names.keys():
        return names[name]
    return "Unknown metric"


# Plotting functions

def hist_correlations(ax, data1, data2, full=True):
    bins = np.linspace(-1 if full else 0, 1, 21)
    ax.hist(data1, bins=bins, density=True, alpha=.85)
    ax.hist(data2, bins=bins, density=True, alpha=.85)
    if full:
        ax.set_xticks([-1, 0, 1])
    else:
        ax.set_xticks([0, .5, 1])
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    ax.axvline(0, c='k')

def eigenplot(ax, matrices):
    for m in matrices:
        eigs = np.sort(np.abs(np.real(np.linalg.eig(m)[0])))[::-1][0:100]
        ax.plot(np.log10(list(range(1, len(eigs)+1))), np.log10(eigs), c='k', linewidth=.5)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["$10^%i$" % k for k in [0, 1, 2]])
    ax.set_yticklabels(["$10^%i$" % k for k in [0, 1, 2]])
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue magnitude")
    sns.despine(ax=ax)

#def scatter_compare_metrics(name, c, pos, df, x, y, error_upper=None, error_lower=None):

def scatter_compare_metrics(c, pos, models, metric, df):
    """df should have "modelloss", "subject", "metric", "baseline", "median", "qupper", "qlower" """
    name = metric+"_scatter"
    c.add_axis(name, pos, pos+SCATTER_SIZE)
    ax = c.ax(name)
    axlims = [np.inf, -np.inf, np.inf, -np.inf]
    for i,model in enumerate(models):
        color = MODEL_COLORS[model]
        rows = df.query(f"modelloss == '{model}' and metric == '{metric}'")
        ax.scatter(rows['baseline'], rows['median'], color=color, s=10)
        ax.errorbar(rows['baseline'], rows['median'], linestyle='none', yerr=(rows['median']-rows['qlower'], rows['qupper']-rows['median']), color=color, elinewidth=.4)
        corr = scipy.stats.spearmanr(rows['baseline'], rows['median']).correlation
        if   i == 0:
            c.add_text("r=%.2f" % corr, Point(-.1, 1.05, "axis_"+name), color=color, horizontalalignment="left", verticalalignment="top")
        elif i == 1:
            c.add_text("r=%.2f" % corr, Point(.5, 1.05, "axis_"+name), color=color, horizontalalignment="center", verticalalignment="top")
        elif i == 2:
            c.add_text("r=%.2f" % corr, Point(1.1, 1.05, "axis_"+name), color=color, horizontalalignment="right", verticalalignment="top")
        elif i == 3:
            c.add_text("r=%.2f" % corr, Point(.5, .95, "axis_"+name), color=color, horizontalalignment="center", verticalalignment="top")
        axlims = [min(axlims[0], rows['baseline'].min()), max(axlims[1], rows['baseline'].max()),
                  min(axlims[2], rows['median'].min()), max(axlims[3], rows['median'].max())]
    axlims = [min(axlims[0], axlims[2])*1.1, max(axlims[1], axlims[3])*1.1,
              min(axlims[0], axlims[2])*1.1, max(axlims[1], axlims[3])*1.1]
    ax.set_xlabel("Data")
    ax.set_ylabel("Model")
    ax.axis("square")
    axsize = ax.axis()
    ax.plot([-100, 100], [-100, 100], linestyle="--", c='k')
    ax.axis(axlims)
    sns.despine(ax=ax)
    c.add_text(metric_name(metric), Point(.5, 1.1, "axis_"+name))
    ticloc = ticker.MaxNLocator(nbins=3, min_n_ticks=3)
    ax.xaxis.set_major_locator(ticloc)
    ax.yaxis.set_major_locator(ticloc)

def nodal_correlation_hist(ax, metric, df, compare_to='tau'):
    piv = df.pivot_table(index='subject', columns='node', values=[metric, compare_to])
    spearman_corrs = piv.apply(lambda row : scipy.stats.spearmanr(row[metric].dropna(), row[compare_to].dropna())[0], axis=1)
    bins = np.linspace(-1, 1, 21)
    ax.hist(spearman_corrs, bins=bins, density=True, color='k')
    ax.set_xticks([-1, 0, 1])
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    ax.axvline(0, c='k')
    ax.set_xlabel("Correlation")
    ax.set_title(metric_name(metric))

def nodal_correlation_scatter(ax, metric, df, compare_to='tau'):
    piv = df.pivot_table(index='subject', columns='node', values=[metric, compare_to])
    spearman_corrs = piv.apply(lambda row : scipy.stats.spearmanr(row[metric].dropna(), row[compare_to].dropna())[0], axis=1)
    bins = np.linspace(-1, 1, 21)
    #best_subj = dnodalhd[dnodalhd['subject'] == spearman_corrs.abs().argmax()]
    #worst_subj = dnodalhd[dnodalhd['subject'] == spearman_corrs.abs().argmin()]
    median_subj = df[df['subject'] == spearman_corrs.abs().argsort()[len(spearman_corrs)//2]]
    #ax.scatter(best_subj[compare_to], best_subj[metric])
    #ax.scatter(worst_subj[compare_to], worst_subj[metric])
    ax.scatter(median_subj[compare_to], median_subj[metric], c='k', s=10)
    #ax.set_title(f"{metric_name(compare_to)} vs {metric_name(metric)},\n{min(spearman_corrs):.2}--{max(spearman_corrs):.2}"); plt.show()
    ax.set_xlabel(metric_name(compare_to) + " (data)")
    ax.set_ylabel(metric_name(metric) + " (data)")
    #ax.set_title("%.3f" % scipy.stats.spearmanr(median_subj[metric], median_subj[compare_to])[0])
    sns.despine(ax=ax)

def side_tau_example_subject(ax, lhs, rhs):
    ax.scatter(lhs, rhs, s=5, c='k')
    ax.set_xlabel("Tau (LHS)")
    ax.set_ylabel("Tau (RHS)")
    sns.despine(ax=ax)
    ax.axis("square")
    axsize = ax.axis()
    ax.plot([-10, 10], [-10, 10], linestyle="--", c='k')
    ticloc = ticker.MaxNLocator(nbins=2, min_n_ticks=2)
    ax.xaxis.set_major_locator(ticloc)
    ax.yaxis.set_major_locator(ticloc)
    ax.set_xlim(axsize[0:2])
    ax.set_ylim(axsize[2:4])
    

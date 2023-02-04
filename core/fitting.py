import numpy as np
import scipy.optimize
from util import graph_metrics_from_cm, threshold_matrix, get_eigenvalues


#################### Fit to eigenvalues ####################

def loss_eigenvalues_l1(m, eigs, **kwargs):
    """Eigenvalue L1 norm loss function"""
    eigs_model = get_eigenvalues(m)
    return np.sum(np.abs(eigs_model-eigs))

def loss_eigenvalues_l2(m, eigs, **kwargs):
    """Eigenvalue L2 norm loss function"""
    eigs_model = get_eigenvalues(m)
    return np.sum(np.square(eigs_model-eigs))

def loss_log_eigenvalues_l2(m, eigs, **kwargs):
    """Eigenvalue L2 norm loss function"""
    eigs_model = np.log(get_eigenvalues(m))
    return np.sum(np.square(eigs_model-np.log(eigs)))

#################### Fit to raw correlation values ####################

# def loss_rawcorrelation(m, , dist=None):
#     return np.sum(np.abs(m1-m2))

# def loss_correlation_in_rawcorrelation(m1, m2, dist=None):
#     return -np.corrcoef(m1.flatten(), m2.flatten())[0,1]

#################### Fit to spatial autocorrelation ####################

# import pandas
# import scipy.optimize
# def get_binned_dists(cm, dist, discretization=1):
#     cm_flat = cm.flatten()
#     dist_flat = dist.flatten()
#     df = pandas.DataFrame(np.asarray([dist_flat, cm_flat]).T, columns=["dist", "corr"])
#     df['dist_bin'] = np.round(df['dist']/discretization)*discretization
#     df_binned = df.groupby('dist_bin').mean().reset_index().sort_values('dist_bin')
#     binned_cm_flat = df_binned['corr']
#     return binned_cm_flat

# def loss_SA(m1, m2, dist):
#     binned1 = get_binned_dists(m1, dist)
#     binned2 = get_binned_dists(m2, dist)
#     return np.sum(np.square(binned1-binned2))


#################### Fit to mean/std ####################

# def loss_meanstd(m1, m2, dist=None):
#     meandiff = np.mean(m1)-np.mean(m2)
#     stddiff = np.std(m1)-np.std(m2)
#     return meandiff**2 + stddiff**2

#################### Fit to graph metrics (Betzel) ####################

# Commented out because the rest of the code doesn't currently support passing distance matrices to loss functions
# def loss_betzel(m1, m2, dists):
#     observed_network = threshold_matrix(m1)
#     model_network = threshold_matrix(m2)
#     degree = lambda network : np.sum(network, axis=1)
#     cluster = lambda network : np.asarray(bct.clustering_coef_bu(network))
#     centrality = lambda network : np.asarray(bct.betweenness_bin(network))
#     def edge_length(network, dist_mat):
#         dist_mat_prime = (np.triu(network, 1) * dist_mat).flatten() # dist if connected, 0 if unconnected
#         return dist_mat_prime[np.nonzero(dist_mat_prime)]
#     KS_k, p_k = ks_2samp(degree(observed_network), degree(model_network))
#     KS_c, p_c = ks_2samp(cluster(observed_network), cluster(model_network))
#     KS_b, p_b = ks_2samp(centrality(observed_network), centrality(model_network))
#     #KS_e, p_e = ks_2samp(edge_length(observed_network, dists), edge_length(model_network, dists))
#     print(KS_k, KS_c, KS_b)#, KS_e)
#     return max(KS_k, KS_c, KS_b)#, KS_e)

#################### Fit to graph metrics ####################

def loss_graphmetrics(m, metrics, **kwargs):
    mets_model = graph_metrics_from_cm(m)
    mets_model['transitivity'] *= 10
    eval_on = ['lefficiency', 'assort', 'gefficiency', 'cluster']
    # This was changed to include modularity and clustering, along with the other 3
    print([(mets_model[e]-metrics[e])**2 for e in eval_on])
    return np.sum([(mets_model[e]-metrics[e])**2 for e in eval_on])

# Set up the cache
loss_graphmetrics._cache_matrix = -1
loss_graphmetrics._cache_metrics = None

def loss_graphmetrics_v2(m, metrics, **kwargs):
    mets_model = graph_metrics_from_cm(m)
    mets_model['transitivity'] *= 10
    eval_on = ['lefficiency', 'transitivity', 'gefficiency', 'cluster']
    # This was changed to include modularity and clustering, along with the other 3
    print([(mets_model[e]-metrics[e])**2 for e in eval_on])
    return np.sum([(mets_model[e]-metrics[e])**2 for e in eval_on])

# Set up the cache
loss_graphmetrics_v2._cache_matrix = -1
loss_graphmetrics_v2._cache_metrics = None

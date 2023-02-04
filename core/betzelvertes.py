import numpy as np
import math
import datasets
import bct
from util import threshold_matrix
from scipy.stats import ks_2samp
from scipy.sparse.csgraph import minimum_spanning_tree
import pickle


# define K functions
def K_dist(u, v, network):
    return dist_mat[u,v]

def K_neighbors(u, v, network):
    return np.dot(network[u,:], network[:,v])

def K_neighbors_mat(network):
    network = network.astype(np.float32)
    K = network@network
    np.fill_diagonal(K, 0)
    return K

def degree(network):
    return np.sum(network, axis=1)

def cluster(network):
    return np.asarray(bct.clustering_coef_bu(network))

def centrality(network):
    return np.asarray(bct.betweenness_bin(network))

def edge_length(network, dist_mat):
    dist_mat_prime = (np.triu(network, 1) * dist_mat).flatten() # dist if connected, 0 if unconnected
    return dist_mat_prime[np.nonzero(dist_mat_prime)]

def graph_metrics_thresh(network):
    metrics = dict()
    metrics['cluster'] = np.mean(bct.clustering_coef_bu(network))
    metrics['assort'] = bct.assortativity_bin(network, 0)
    distmat = bct.distance_bin(network)
    metrics['path'] = np.mean(distmat)
    metrics['diam'] = np.max(distmat)
    metrics['gefficiency'] = np.mean(1/distmat[np.triu_indices(len(network), 1)])
    comms = bct.community_louvain(network)
    metrics['modularity'] = comms[1]
    metrics['n_modules'] = len(set(comms[0]))
    metrics['lefficiency'] = float(np.mean(bct.efficiency_bin(network, local=True)))
    metrics['transitivity'] = bct.transitivity_bu(network)
    return metrics

def compute_energy(observed_network, model_network, dist_mat):
    # compute KS statistics
    KS_k, p_k = ks_2samp(degree(observed_network), degree(model_network))
    KS_c, p_c = ks_2samp(cluster(observed_network), cluster(model_network))
    KS_b, p_b = ks_2samp(centrality(observed_network), centrality(model_network))
    KS_e, p_e = ks_2samp(edge_length(observed_network, dist_mat), edge_length(model_network, dist_mat))

    #print(KS_k, KS_c, KS_b, KS_e)
    return max(KS_k, KS_c, KS_b, KS_e)

def betzelvertes_generate(observed_mat, dist_mat, eta, gamma, seed=0):
    """Implementation of the economical clustering model.

    dist-mat = distance matrix, observed_mat = subject matrix (for constructing
    the MST only), and of course eta and gamma are parameters for the
    generative process.

    Note we implemented this from scratch since the version in bctpy had lots
    of bugs.
    """

    n = len(dist_mat) # so we are dealing with n by n matrices

    observed_network = threshold_matrix(observed_mat)

    # generate starting MST for model
    observed_mat_new = (1-np.eye(n, dtype='int'))*observed_mat + -1*np.eye(n)
    model_network_start = (np.asarray(minimum_spanning_tree(np.sqrt(2*(1-observed_mat_new+1e-6))).todense()) != 0).astype(int)
    model_network_start = model_network_start + model_network_start.T

    # add M connections to each model network
    # compute M = (connections in observed) - (connections in MST)
    M = np.count_nonzero(np.triu(observed_network, 1)) - np.count_nonzero(np.triu(model_network_start, 1))

    dist_mat_filled = (np.eye(n) + dist_mat).astype(np.float32)
    dist_mat_power = (dist_mat_filled**eta - np.eye(n)).astype(np.float32)

    # get current state
    r_state = np.random.get_state()

    np.random.seed(seed)
    model_network = model_network_start.copy()

    u,v = np.where(np.triu(np.ones((n,n)), 1))

    for connect_ctr in range(M):
        # compute probabilities
        K_mat = K_neighbors_mat(model_network)
        prob_mat = dist_mat_power * K_mat**gamma * np.logical_not(model_network).astype(np.float32)

        C = np.append(0, np.cumsum(prob_mat[u,v]))[:-1]
        r = np.sum(np.random.random_sample()*C[-1] >= C)
        uu = u[r-1]
        vv = v[r-1]

        model_network[uu,vv] = 1
        model_network[vv,uu] = 1
        #print(connect_ctr)

    np.random.set_state(r_state)

    return model_network

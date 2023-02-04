import numpy as np
import paranoid as pns
import bct
import scipy.sparse
import scipy.stats
pns.settings.Settings.set(namespace={"np": np})
from paranoidtypes import CorrelationMatrix, SymmetricMatrix, Graph, DistanceMatrix
import statsmodels.tsa.api
import pickle
import pandas

@pns.accepts(SymmetricMatrix, pns.Range(0, 1))
@pns.returns(Graph)
def threshold_matrix(M, threshold_density=.1):
    assert type(M) == np.ndarray, "Wrong type"
    # M = np.absolute(M) # We want negative correlations to be edges, too.
    # Set diagonal to -1.  This makes sure it is added last, and thus
    # will not screw up the algorithm below for adding edges.  We have
    # ndarrays, so this is component-wise multiplication, not actual
    # matrix multiplication.
    M = (1-np.eye(M.shape[0], dtype='int'))*M + -1*np.eye(M.shape[0])
    # Now we threshold it.  We start with a minimum spanning tree to
    # ensure there are no disconnected nodes.  The cryptic square root
    # stuff gives it ultrametricity.
    MST = (np.asarray(scipy.sparse.csgraph.minimum_spanning_tree(np.sqrt(2*(1-M+1e-6))).todense()) != 0).astype(int)
    MST = MST + MST.T # Numpy routine doesn't make the result symmetric
    M = (1-MST)*M + -1*MST # Set MST to -1, like above w/ diagonal.
    Tri = np.triu(M) # Remove duplicate elements for symmetric matrix
    Rks = np.reshape(scipy.stats.rankdata(Tri, method='min'), Tri.shape) #Order them
    Rks = 1+np.max(Rks)-Rks # +1 so they start at 1, not 0
    # Correct for MST edges, the absence of a diagonal.  Divide by 2
    # for undirected graph, since we are only considering a triangular
    # matrix.
    conns_to_add = (M.shape[0]*(M.shape[0]-1)/2)*threshold_density - (M.shape[0]-1)
    newconns = Rks<=conns_to_add
    newconns = np.logical_or(newconns, newconns.T)
    Gmat = np.logical_or(MST, newconns)
    return np.asarray(Gmat)

@pns.accepts(pns.NDArray(t=pns.Range(-1, 1)))
@pns.returns(pns.NDArray(t=pns.Number))
def fisher(m):
    return np.arctanh(m)

@pns.accepts(pns.NDArray(t=pns.Number, d=2))
@pns.requires("np.allclose(M, M.T)")
@pns.returns(SymmetricMatrix)
def make_perfectly_symmetric(M):
    return np.maximum(M, M.T)

@pns.accepts(CorrelationMatrix)
@pns.returns(pns.Dict(k=pns.String, v=pns.Number))
def cm_metrics_from_cm(c):
    metrics = dict()
    c = make_perfectly_symmetric(c)
    ctri = c[np.triu_indices(len(c), 1)]
    metrics["meancor_fish"] = np.mean(fisher(ctri))
    metrics["varcor_fish"] = np.var(fisher(ctri))
    metrics["kurtcor_fish"] = scipy.stats.kurtosis(fisher(ctri))
    metrics["meancor"] = np.mean(ctri)
    metrics["varcor"] = np.var(ctri)
    metrics["kurtcor"] = scipy.stats.kurtosis(ctri)
    return metrics

@pns.accepts(CorrelationMatrix, pns.Range(0, 1))
@pns.returns(pns.Dict(k=pns.String, v=pns.Number))
def graph_metrics_from_cm(c, density=.1):
    c = make_perfectly_symmetric(c)
    metrics = cm_metrics_from_cm(c)
    c_thresh = threshold_matrix(c, threshold_density=density)
    metrics.update(graph_metrics_from_adj(c_thresh))
    return metrics

@pns.accepts(Graph)
@pns.returns(pns.Dict(k=pns.String, v=pns.Number))
def graph_metrics_from_adj(c_thresh):
    metrics = dict()
    metrics['cluster'] = np.mean(bct.clustering_coef_bu(c_thresh))
    metrics['assort'] = bct.assortativity_bin(c_thresh, 0)
    distmat = bct.distance_bin(c_thresh)
    metrics['path'] = np.mean(distmat)
    metrics['diam'] = np.max(distmat)
    metrics['gefficiency'] = np.mean(1/distmat[np.triu_indices(c_thresh.shape[0], 1)])
    comms = bct.community_louvain(c_thresh)
    metrics['modularity'] = comms[1]
    metrics['n_modules'] = len(set(comms[0]))
    metrics['lefficiency'] = float(np.mean(bct.efficiency_bin(c_thresh, local=True)))
    metrics['transitivity'] = bct.transitivity_bu(c_thresh)
    return metrics

@pns.accepts(pns.NDArray(d=2, t=pns.Number))
@pns.returns(pns.Dict(k=pns.String, v=pns.NDArray(d=1, t=pns.Number)))
def nodal_graph_metrics_from_tss(tss):
    metrics = dict()
    metrics["ar1"] = get_ar1s(tss)
    metrics["tsmean"] = np.mean(tss, axis=1)
    metrics["tsstd"] = np.std(tss, axis=1)
    metrics["tsvar"] = np.var(tss, axis=1)
    metrics["tskurt"] = scipy.stats.kurtosis(tss, axis=1)
    cm = correlation_matrix_pearson(tss)
    metrics.update(nodal_graph_metrics_from_cm(cm))
    return metrics

@pns.accepts(CorrelationMatrix)
@pns.returns(pns.Dict(k=pns.String, v=pns.NDArray(d=1, t=pns.Number)))
def nodal_graph_metrics_from_cm(c):
    metrics = dict()
    nodiag = c[~np.eye(c.shape[0],dtype=bool)].reshape(c.shape[0],-1)
    metrics["mean"] = np.mean(nodiag, axis=1)
    metrics["std"] = np.std(nodiag, axis=1)
    metrics["var"] = np.var(nodiag, axis=1)
    metrics["kurt"] = scipy.stats.kurtosis(nodiag, axis=1)
    metrics["mean_fish"] = np.mean(fisher(nodiag), axis=1)
    metrics["std_fish"] = np.std(fisher(nodiag), axis=1)
    metrics["var_fish"] = np.var(fisher(nodiag), axis=1)
    metrics["kurt_fish"] = scipy.stats.kurtosis(fisher(nodiag), axis=1)
    c_thresh = threshold_matrix(c, threshold_density=.1)
    metrics.update(nodal_graph_metrics_from_adj(c_thresh))
    return metrics

@pns.accepts(Graph)
@pns.returns(pns.Dict(k=pns.String, v=pns.NDArray(d=1, t=pns.Number)))
def nodal_graph_metrics_from_adj(c_thresh):
    metrics = dict()
    metrics['degree'] = np.sum(c_thresh, axis=1)
    metrics['cluster'] = np.asarray(bct.clustering_coef_bu(c_thresh))
    metrics['lefficiency'] = np.asarray(bct.efficiency_bin(c_thresh, local=True))
    metrics['centrality'] = np.asarray(bct.betweenness_bin(c_thresh))
    return metrics

@pns.accepts(pns.NDArray(d=2, t=pns.Number))
@pns.returns(pns.Dict(k=pns.String, v=pns.NDArray(d=1, t=pns.Number)))
#@pns.requires("ts.shape[0] < ts.shape[1]") # More timepoints than neurons
@pns.ensures("len(return['ar1']) == ts.shape[0]")
def nodal_stats_from_ts(ts):
    metrics = dict()
    metrics["ts_mean"] = np.mean(ts, axis=1)
    metrics["ts_std"] = np.std(ts, axis=1)
    metrics["ts_var"] = np.var(ts, axis=1)
    metrics["ts_kurt"] = scipy.stats.kurtosis(ts, axis=1)
    a = ts[:,:-1]; b = ts[:,1:]
    #metrics["ar1"] = np.mean((a-np.mean(a,axis=1).reshape(-1,1))*(b-np.mean(b,axis=1).reshape(-1,1))/(np.std(a,axis=1)*np.std(b,axis=1)).reshape(-1,1), axis=1)
    return metrics

def nodal_ts_and_cm_metrics(ts):
    metrics = dict()
    metrics["ts_mean"] = np.mean(ts, axis=1)
    metrics["ts_std"] = np.std(ts, axis=1)
    metrics["ts_var"] = np.var(ts, axis=1)
    metrics["ts_kurt"] = scipy.stats.kurtosis(ts, axis=1)
    c = correlation_matrix_pearson(ts)
    nodiag = c[~np.eye(c.shape[0],dtype=bool)].reshape(c.shape[0],-1)
    metrics["mean"] = np.mean(nodiag, axis=1)
    metrics["std"] = np.std(nodiag, axis=1)
    metrics["var"] = np.var(nodiag, axis=1)
    metrics["kurt"] = scipy.stats.kurtosis(nodiag, axis=1)
    return metrics

def get_cm_lmbda_params(cm, dist, discretization=1):
    cm_flat = cm.flatten()
    dist_flat = dist.flatten()
    df = pandas.DataFrame(np.asarray([dist_flat, cm_flat]).T, columns=["dist", "corr"])
    df['dist_bin'] = np.round(df['dist']/discretization)*discretization
    df_binned = df.groupby('dist_bin').mean().reset_index().sort_values('dist_bin')
    binned_dist_flat = df_binned['dist_bin']
    binned_cm_flat = df_binned['corr']
    binned_dist_flat[0] = 1
    spatialfunc = lambda v : np.exp(-binned_dist_flat/v[0])*(1-v[1])+v[1]
    with np.errstate(all='warn'):
        res = scipy.optimize.minimize(lambda v : np.sum((binned_cm_flat-spatialfunc(v))**2), [10, .3], bounds=[(.1, 100), (-1, 1)])
    return (res.x[0], res.x[1])

@pns.accepts(pns.NDArray(d=2, t=pns.Number))
@pns.returns(CorrelationMatrix)
def correlation_matrix_pearson(ts):
    """Find the correlation matrix from the timeseries ts.  

    Each region should be a separate row, and each timepoint should be
    a separate column.
    """
    return make_perfectly_symmetric(np.corrcoef(ts))

@pns.accepts(pns.NDArray(d=2, t=pns.Number))
@pns.requires("distances.shape[1] == 3")
@pns.returns(DistanceMatrix)
def distance_matrix_euclidean(distances):
    """Returns a Euclidean distance matrix. 

    `distances` should be a list of xyz coordinates
    """
    return scipy.spatial.distance.cdist(distances, distances)

def cube_sampling_distance_matrix_euclidean(num_regions, seed=0):
    #centroids of parcels chosen randomly from within a cube
    r = np.random.get_state()
    np.random.seed(seed)
    x = []
    for i in range(num_regions):
        x.append(np.random.rand(3))
    x = np.array(x)
    np.random.set_state(r)
    return scipy.spatial.distance.cdist(x, x)

def spherical_sampling_distance_matrix_euclidean(num_regions, seed=0):
    #centroids of parcels chosen randomly on surface of a sphere
    r = np.random.get_state()
    np.random.seed(seed)
    x = []
    for i in range(num_regions):
        theta = 2 * np.pi * np.random.rand()
        phi = np.arccos(1 - 2 * np.random.rand())
        x.append([np.sin(phi)*np.cos(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(phi)])
    x = np.array(x)
    np.random.set_state(r)
    return scipy.spatial.distance.cdist(x, x)

def square_lattice_sampling_2d_euclidean(num_regions, seed=0):
    #centroids of parcels as square lattice
    assert num_regions > 0
    x = []
    n = int(np.ceil(np.sqrt(num_regions)))
    for i in range(n):
        for j in range(n):
            x.append([i,j])
    for i in range(n**2-num_regions):
        np.random.shuffle(x)
        x.pop()
    x = np.array(x)
    assert len(x)==num_regions
    return scipy.spatial.distance.cdist(x,x)

def circular_lattice_sampling_2d_euclidean(num_regions, seed=0):
    #centroids of parcels as circular lattice
    assert num_regions > 0
    x = []
    n = int(np.ceil(np.sqrt(num_regions/np.pi)))
    bound = int(np.ceil(np.sqrt(2*num_regions))/2)
    for i in range(-bound,bound):
        for j in range(-bound,bound):
            if np.sqrt(i**2+j**2)<=n:
                x.append([i,j])
    for i in range(len(x)-num_regions):
        np.random.shuffle(x)
        x.pop()
    x = np.array(x)
    assert len(x)==num_regions
    return scipy.spatial.distance.cdist(x,x)

def degree_preserving_randomization(A, seed=None):
    r = np.random.get_state()
    np.random.seed(seed)
    randomized = bct.randmio_und_connected(A, 5)[0]
    np.random.set_state(r)
    return randomized

@pns.accepts(pns.NDArray(d=2, t=pns.Number), pns.Maybe(pns.Natural0))
@pns.returns(pns.NDArray(d=2, t=pns.Number))
@pns.ensures("tss.shape == return.shape")
def phase_randomize(tss, seed=None):
    """Phase-randomization which doesn't preserve amplitude.

    Phase randomizes by row, i.e. it operates serially, one row at a
    time, where each row is a timeseries and timepoints are given by
    the columns.

    See: T. Schreiber and A. Schmitz. “Surrogate time series”. In
    Physica D vol. 142 (no. 3-4), p346-382 (2000)
    doi:10.1016/S0167-2789(00)00043-9
    """
    surrogates = np.fft.rfft(tss, axis=1)
    (N, n_time) = tss.shape
    len_phase = surrogates.shape[1]
    # Generate random phases uniformly distributed in the
    # interval [0, 2*Pi]
    phases = np.random.RandomState(seed).uniform(low=0, high=2 * np.pi, size=(N, len_phase))
    # Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates *= np.exp(1j * phases)
    # Calculate IFFT and take the real part, the remaining imaginary
    # part is due to numerical errors.
    return np.real(np.fft.irfft(surrogates, n=n_time, axis=1))

@pns.accepts(pns.NDArray(d=2, t=pns.Number), pns.Maybe(pns.Natural0))
@pns.returns(pns.NDArray(d=2, t=pns.Number))
@pns.ensures("tss.shape == return.shape")
def phase_randomize_preserve_amplitude(tss, seed=None):
    """Phase-randomization preserving amplitude.

    See: T. Schreiber and A. Schmitz. “Surrogate time series”. In
    Physica D vol. 142 (no. 3-4), p346-382 (2000)
    doi:10.1016/S0167-2789(00)00043-9
    """
    
    #  Create sorted Gaussian reference series
    gaussian = np.random.randn(*tss.shape)
    gaussian.sort(axis=1)
    
    #  Rescale data to Gaussian distribution
    ranks = tss.argsort(axis=1).argsort(axis=1)
    rescaled_data = np.zeros(tss.shape)
    
    for i in range(tss.shape[0]):
        rescaled_data[i, :] = gaussian[i, ranks[i, :]]
        
    #  Phase randomize rescaled data
    phase_randomized_data = phase_randomize(rescaled_data, seed)
    
    #  Rescale back to amplitude distribution of original data
    sorted_original = tss.copy()
    sorted_original.sort(axis=1)
    
    ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)
    for i in range(tss.shape[0]):
        rescaled_data[i, :] = sorted_original[i, ranks[i, :]]
    return rescaled_data


@pns.accepts(pns.NDArray(d=2, t=pns.Number))
@pns.returns(pns.NDArray(d=2, t=pns.Number))
@pns.ensures("tss.shape == return.shape")
def fit_and_sim_var(tss):
    """Fit a vector auto-regressive model to the timeseries and then generate surrogate timeseries"""
    err_params = np.geterr()
    with np.errstate(all='warn'):
        model = statsmodels.tsa.api.VAR(tss.T).fit(maxlags=1)
        new_tss = model.simulate_var(tss.shape[1])
    return new_tss.T

def psave(filename, data, overwrite=True):
    if overwrite == False:
        import os.path
        assert os.path.isfile(filename) == False, "File already exists, can't overwrite"
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def pload(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def plock(filename, data=()):
    try:
        with open(filename, "xb") as f:
            pickle.dump(data, f)
            return True
    except FileExistsError:
        return False

def get_ar1s(x):
    """Compute AR1 from the timeseries.
    
    This is the biased estimator, but it is the default in numpy.
    This is what we use throughout.  The purpose of this function is
    to standardize computing the AR1 coefficient for this model.
    """
    if isinstance(x[0], (list, np.ndarray)):
        return np.asarray([get_ar1s(xe) for xe in x])
    return np.corrcoef(x[0:-1], x[1:])[0,1]

def get_long_memory(x, minscale, multivariate=False):
    """Multivariate is extremely slow for any reasonably sized parcellation.

    x is the matrix of timeseries (rows are regions, columns are timepoints),
    and minscale is the minimum wavelet scale used to perform the estimation.
    As a rule of thumb, if data are low pass filtered, minscale should be the
    multiple of nyquist corresponding to the filter frequency, e.g. 2 if
    filtering is performed at half nyquist.
    """
    import rpy2.robjects.packages
    import rpy2.robjects.numpy2ri
    x = x.transpose()
    rpy2.robjects.numpy2ri.activate()
    try:
        multiwave = rpy2.robjects.packages.importr('multiwave')
    except Exception as e:
        print("Please install the multiwave package in R")
        raise e
    filt = multiwave.scaling_filter("Daubechies", 8).rx2('h')
    if multivariate:
        res = list(multiwave.mww(x, filt, np.asarray([minscale,11])).rx2('d'))
    else:
        res = [multiwave.mww(x[:,i], filt, np.asarray([minscale,11])).rx2('d')[0] for i in range(0, x.shape[1])]
    rpy2.robjects.numpy2ri.deactivate()
    return res


def get_eigenvalues(m):
    """Find the eigenvalues of the correlation matrix m.

    They will always be real and non-negative since correlation matrices are
    positive semidefinite
    """
    return scipy.linalg.eigvalsh(m)

#################### Frequency spectrum functions ####################

@pns.accepts(pns.Natural1(), pns.RangeOpen(0, 10), pns.Range(0, 2), pns.Positive0())
@pns.returns(pns.NDArray(t=pns.Positive0(), d=1))
def make_spectrum(tslen, TR, alpha, highpass_freq):
    """Create a 1/f^alpha spectrum with the given length, TR, and alpha.
    Return the fourier spectrum (amplitude spectrum).

    This function also applies a high pass filter at .01 hz, as is common in
    fMRI preprocessing pipelines.
    """
    freqs = np.fft.rfftfreq(tslen, TR)
    with np.errstate(all="warn"):
        spectrum = freqs**(-alpha/2)
    if highpass_freq > 0:
        butter = scipy.signal.iirfilter(ftype='butter', N=4, Wn=highpass_freq, btype='highpass', fs=1/TR, output='ba')
        butterresp = scipy.signal.freqz(*butter, fs=1/TR, worN=len(freqs), include_nyquist=True)
        assert np.all(np.isclose(freqs, butterresp[0]))
        spectrum = spectrum * np.abs(butterresp[1])
    spectrum[0] = 0
    return spectrum

@pns.accepts(pns.Natural1(), pns.RangeOpen(0, 10), pns.Range(0, 2), pns.Positive0(), pns.Range(0, 1))
@pns.returns(pns.NDArray(t=pns.Positive0(), d=1))
def make_noisy_spectrum(tslen, TR, alpha, highpass_freq, target_ar1):
    """Similar to make_spectrum, except adds white noise to the spectrum
    (i.e. uniform distribution).  Returns the fourier spectrum (amplitude
    spectrum).

    This also applies the same filter as make_spectrum.

    """
    noiseless_spectrum = make_spectrum(tslen, TR, alpha, highpass_freq)
    N = len(noiseless_spectrum)
    noise = how_much_noise(noiseless_spectrum, target_ar1)
    noisy_spectrum = np.sqrt(noiseless_spectrum**2 + noise**2 * N)
    noisy_spectrum[0] = 0
    return noisy_spectrum

@pns.accepts(pns.NDArray(t=pns.Positive0(), d=1))
@pns.returns(pns.Range(-1, 1))
def get_spectrum_ar1(spectrum):
    """Given a fourier spectrum, return the expected AR1 of a timeseries generated
    with that spectrum."""
    N = len(spectrum)
    weightedsum = np.sum(spectrum**2*np.cos(np.pi*np.arange(0, N)/N))
    return weightedsum/np.sum(spectrum**2)

@pns.accepts(pns.NDArray(t=pns.Positive0(), d=1))
@pns.returns(pns.Positive0)
def get_spectrum_variance(spectrum):
    """Given a fourier spectrum, return the expected variance of a timeseries generated
    with that spectrum."""
    N = (len(spectrum)-1)*2
    return np.sum(2*spectrum**2/N)/len(spectrum)

@pns.accepts(pns.NDArray(t=pns.Positive0(), d=1), pns.RangeOpenClosed(0, 1))
@pns.returns(pns.Positive0())
def how_much_noise(spectrum, target_ar1):
    """Determine the standard deviation of noise to add to achieve a target AR1.

    `spectrum` is the power spectrum to generate from, and `target_ar1` is the
    AR1.  This function answers the following question: If I generate
    timeseries with frequency spectrum (amplitude spectrum) `spectrum`, and
    then add white noise to the generated timeseries, what should the standard
    deviation of this white noise be if I want the timeseries to have the AR1
    coefficient `target_ar1`?
    """

    N = len(spectrum)
    weightedsum = np.sum(spectrum[1:]**2*np.cos(np.pi*np.arange(1, N)/N))
    #sigma = np.sqrt((weightedsum/target_ar1 - np.sum(spectrum**2))/(2*N**2))
    try:
        sigma = np.sqrt((weightedsum - np.sum(spectrum**2)*target_ar1)/(target_ar1*N**2))
    except FloatingPointError:
        sigma = 0
    return sigma

@pns.accepts(pns.Natural1(), pns.RangeOpen(0, 10), pns.Positive0(), pns.Range(0, 1))
@pns.returns(pns.Positive())
def ar1_to_alpha(tslen, TR, highpass_freq, target_ar1):
    """Compute the alpha which would give, noiseless, the given AR1.

    Generate timeseries with get_spectrum_ar1, i.e. high pass filtered.  Return
    a value of alpha such that the filtered pink noise with this exponent has
    AR1 coefficient `target_ar1`.
    """
    objfunc = lambda alpha : (get_spectrum_ar1(make_spectrum(tslen, TR, alpha[0], highpass_freq)) - target_ar1)**2
    x = scipy.optimize.minimize(objfunc, 1.5, bounds=[(0, 2)])
    return float(x.x[0])

ar1_to_alpha_cache = {}
def ar1_to_alpha_fast(tslen, TR, highpass_freq, target_ar1):
    """Identical to `ar1_to_alpha`, except discretizes to increase speed with a minor loss in precision."""
    global ar1_to_alpha_cache
    ar1round = round(target_ar1, 2)
    key = (tslen, TR, highpass_freq, ar1round)
    if key in ar1_to_alpha_cache.keys():
        return ar1_to_alpha_cache[key]
    val = ar1_to_alpha(*key)
    ar1_to_alpha_cache[key] = val
    return val

@pns.accepts(pns.NDArray(t=pns.Positive0(), d=1), pns.Natural0())
@pns.returns(pns.NDArray(t=pns.Number(), d=1))
def timeseries_from_spectrum(spectrum, seed=0):
    """Given a frequency spectrum (amplitude spectrum) `spectrum`, generate a timeseries with that spectrum"""
    randstate = np.random.RandomState(seed)
    N = (len(spectrum)-1)*2
    reals = randstate.randn(len(spectrum)) * spectrum
    ims = randstate.randn(len(spectrum)) * spectrum
    reals[0] = 0
    ims[0] = 0
    ims[-1] = 0
    ts = np.fft.irfft(reals + 1J*ims, n=N)
    return ts


def spectra_from_timeseries(tss):
    tss = np.asarray(tss)
    if len(tss.shape) == 1:
        tss = np.asarray([tss])
    spec = np.abs(np.fft.rfft(tss))**2
    spec[...,0] = 0
    return spec

@pns.accepts(pns.NDArray(t=pns.Range(0,1), d=2), pns.NDArray(t=pns.Positive0(), d=2), pns.Natural0())
@pns.returns(pns.NDArray(t=pns.Number(), d=2))
def spatial_temporal_timeseries(cm, spectra, seed=0):
    """Generate timeseries with given amplitude spectra and correlation matrices

    Generate timeseries according to fourier (amplitude) spectra `spectra` (n
    by k) which have a given correlation matrix `cm` (n by n).
    """
    N_regions = cm.shape[0]
    N_freqs = len(spectra[0])
    N_timepoints = (N_freqs-1)*2
    assert spectra.shape == (N_regions, N_freqs)
    sum_squares = np.sum(spectra**2, axis=1, keepdims=True)
    cosine_similarity = (spectra @ spectra.T)/np.sqrt(sum_squares @ sum_squares.T)
    covmat = cm / cosine_similarity
    #import matplotlib.pyplot as plt
    #plt.imshow(covmat); plt.colorbar(); plt.show()
    # if np.min(np.linalg.eigvalsh(covmat)) < -1e-8:
    #     print("Error with eigs")
    #     print(np.linalg.eigvalsh(covmat)[0:3])
    #     print(np.mean(cm))
    #     return np.random.randn(N_regions, N_timepoints)
    if np.min(np.linalg.eigvalsh(covmat)) < -1e-8:
        raise PositiveSemidefiniteError("Correlation matrix is not possible with those spectra using this method!")
    randstate = np.random.RandomState(seed)
    rvs = randstate.multivariate_normal(np.zeros(N_regions), cov=covmat, size=N_freqs*2)
    reals = rvs[0:N_freqs].T * spectra
    ims = rvs[N_freqs:].T * spectra
    # Since the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    ims[:,-1] = 0
    # The DC component must be zero and real
    reals[:,0] = 0
    ims[:,0] = 0
    tss = np.fft.irfft(reals + 1J*ims, n=N_timepoints, axis=-1)
    return tss

def spatial_exponential_floor(distances, lmbda, floor):
    return np.exp(-distances/lmbda)*(1-floor)+floor

class PositiveSemidefiniteError(Exception):
    pass

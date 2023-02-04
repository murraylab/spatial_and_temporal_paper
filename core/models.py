import numpy as np
import paranoid as pns
from paranoidtypes import DistanceMatrix, CorrelationMatrix, ParamsList, SymmetricMatrix
import scipy.optimize
import util

class Model:
    def __init__(self, *args, **kwargs):
        raise Exception("Methods should be called statically")
    @classmethod
    def generate(cls, *args, **kwargs):
        """Default implementation which uses the "generate_timeseries" method if it exists"""
        return util.correlation_matrix_pearson(cls.generate_timeseries(*args, **kwargs))
    @classmethod
    def quickfit(cls, dataset, method, seed, fixed_params={}):
        params = []
        for i in range(0, dataset.N_subjects()):
            p = cls.fit(distance_matrix=dataset.get_dists(),
                        method=method,
                        eigs=dataset.get_eigenvalues()[i],
                        metrics=dataset.get_metrics(False)[i],
                        ar1vals=dataset.get_ar1s()[i],
                        num_timepoints=dataset.N_timepoints(),
                        TR=dataset.TR/1000,
                        highpass_freq=dataset.highpass,
                        seed=seed,
                        fixed_params=fixed_params,
                        )
            params.append(p)
        return params
    @classmethod
    def fit(cls, distance_matrix, method, eigs, metrics, *args, seed=0, optimizer="differential_evolution", verbose=True, fixed_params={}, **kwargs):
        assert hasattr(cls, "params"), "Must specify parameters in 'params' class variable"
        assert cls.params in pns.Dict(pns.String, pns.Tuple(pns.Number, pns.Number)), \
            "Invalid format for parameters: must be {'name': (minval, maxval)}"
        # Set up the fit with differential evolution
        param_names = [p for p in sorted(cls.params.keys()) if p not in fixed_params.keys()]
        param_bounds = [cls.params[i] for i in param_names]
        if isinstance(seed, int):
            seed = [seed]
        def obj_func(params):
            #params = [pmin if p < pmin else pmax if p > pmax else p for p,(pmin,pmax) in zip(params,param_bounds)]
            for p,(pmin,pmax) in zip(params,param_bounds):
                if p < pmin or p > pmax:
                    return np.inf
            all_params = {n: params[i] for i,n in enumerate(param_names)}
            all_params.update(fixed_params)
            obj_func_values = []
            try:
                for s in seed:
                    m = cls.generate(distance_matrix,
                                        all_params,
                                        *args, seed=s, **kwargs)
                    obj_func_values.append(method(m, eigs=eigs, metrics=metrics, distance_matrix=distance_matrix))
            except util.PositiveSemidefiniteError:
                obj_func_values = [np.inf]
            res = np.mean(obj_func_values)
            if verbose:
                print(" \t".join([f"{k}={v:.3}" for k,v in all_params.items()])+" \t"+f"obj_func={res:.3}")
            return res
        if optimizer == "differential_evolution":
            res = scipy.optimize.differential_evolution(obj_func, bounds=param_bounds,
                                                        disp=True, polish=False, maxiter=100)
        else:
            res = scipy.optimize.minimize(obj_func, bounds=param_bounds, method=optimizer, x0=[np.random.random()*(p[1]-p[0])+p[0] for p in param_bounds])
        return {n: v for n,v in zip(param_names, res.x)}


#################### Generative model (Colorless) ####################

class Model_Colorless(Model):
    """Spatially autocorrelated Brownian motion with region-specific Gaussian noise to bring down the region's AR1."""
    name = "Colorless"
    params = {'lmbda': (0.01, 80), 'floor': (0, 1)}
    fixed_power = 2 # Brownian motion
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, ar1vals, num_timepoints, TR, highpass_freq, seed=0):
        assert num_timepoints % 2 == 0, "Must be even timeseries length"
        # Filtered brown noise spectrum
        spectrum = util.make_spectrum(tslen=num_timepoints, TR=TR, alpha=cls.fixed_power, highpass_freq=highpass_freq)
        spectra = np.asarray([spectrum]*len(ar1vals))
        # Spatial autocorrelation matrix
        corr = cls.transform_dists(distance_matrix, params)
        # Create spatially embedded timeseries with the given spectra
        tss = util.spatial_temporal_timeseries(corr, spectra, seed=seed)
        # Compute the standard deviation of nosie we need to add to get the desired AR1
        noises = [util.how_much_noise(spectrum, max(.001, ar1)) for ar1 in ar1vals]
        # Add noise to the timeseries
        rng = np.random.RandomState(seed+100000)
        tss += rng.randn(tss.shape[0], tss.shape[1]) * np.asarray(noises).reshape(-1,1)
        return tss
    @staticmethod
    def transform_dists(distances, params):
        return util.spatial_exponential_floor(distances, params['lmbda'], params['floor'])

class Model_ColorlessCor(Model):
    """Spatially autocorrelated Brownian motion with region-specific correlated Gaussian noise to bring down the region's AR1."""
    name = "ColorlessCor"
    params = {'lmbda': (0.01, 80), 'floor': (0, 1)}
    fixed_power = 2 # Brownian motion
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, ar1vals, num_timepoints, TR, highpass_freq, seed=0):
        assert num_timepoints % 2 == 0, "Must be even timeseries length"
        # Filtered brown noise spectrum
        spectrum = util.make_spectrum(tslen=num_timepoints, TR=TR, alpha=cls.fixed_power, highpass_freq=highpass_freq)
        spectra = np.asarray([spectrum]*len(ar1vals))
        # Spatial autocorrelation matrix
        corr = cls.transform_dists(distance_matrix, params)
        # Create spatially embedded timeseries with the given spectra
        tss = util.spatial_temporal_timeseries(corr, spectra, seed=seed)
        # Compute the standard deviation of nosie we need to add to get the desired AR1
        noises = [util.how_much_noise(spectrum, max(.001, ar1)) for ar1 in ar1vals]
        # Add noise to the timeseries
        rng = np.random.RandomState(seed+100000)
        tss += rng.multivariate_normal(mean=[0]*tss.shape[0], cov=corr,  size=tss.shape[1]).T * np.asarray(noises).reshape(-1,1)
        return tss
    @staticmethod
    def transform_dists(distances, params):
        return util.spatial_exponential_floor(distances, params['lmbda'], params['floor'])


class Model_ColorlessCorGS(Model):
    """Spatially autocorrelated Brownian motion with region-specific correlated Gaussian noise to bring down the region's AR1."""
    name = "ColorlessCor"
    params = {'lmbda': (0.01, 140), 'floor': (0, 1), 'gs': (0, 2)}
    fixed_power = 2 # Brownian motion
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, ar1vals, num_timepoints, TR, highpass_freq, seed=0):
        assert num_timepoints % 2 == 0, "Must be even timeseries length"
        # Filtered brown noise spectrum
        spectrum = util.make_spectrum(tslen=num_timepoints, TR=TR, alpha=cls.fixed_power, highpass_freq=highpass_freq)
        spectra = np.asarray([spectrum]*len(ar1vals))
        # Spatial autocorrelation matrix
        corr = cls.transform_dists(distance_matrix, params)
        # Create spatially embedded timeseries with the given spectra
        tss = util.spatial_temporal_timeseries(corr, spectra, seed=seed)
        # Compute the standard deviation of nosie we need to add to get the desired AR1
        noises = [util.how_much_noise(spectrum, max(.001, ar1)) for ar1 in ar1vals]
        # Global signal
        gs = util.timeseries_from_spectrum(spectrum)
        # Add noise to the timeseries
        rng = np.random.RandomState(seed+100000)
        tss += rng.multivariate_normal(mean=[0]*tss.shape[0], cov=corr,  size=tss.shape[1]).T * np.asarray(noises).reshape(-1,1)
        tss += gs*params['gs']
        return tss
    @staticmethod
    def transform_dists(distances, params):
        return util.spatial_exponential_floor(distances, params['lmbda'], params['floor'])

class Model_ColorlessCorRatio(Model):
    """Spatially autocorrelated Brownian motion with region-specific correlated Gaussian noise to bring down the region's AR1."""
    name = "ColorlessCor"
    params = {'lmbda': (0.01, 80), 'floor': (0, 1), 'ratio': (0, 1)}
    fixed_power = 2 # Brownian motion
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, ar1vals, num_timepoints, TR, highpass_freq, seed=0):
        assert num_timepoints % 2 == 0, "Must be even timeseries length"
        # Filtered brown noise spectrum
        spectrum = util.make_spectrum(tslen=num_timepoints, TR=TR, alpha=cls.fixed_power, highpass_freq=highpass_freq)
        spectra = np.asarray([spectrum]*len(ar1vals))
        # Spatial autocorrelation matrix
        corr = cls.transform_dists(distance_matrix, params)
        # Create spatially embedded timeseries with the given spectra
        tss = util.spatial_temporal_timeseries(corr, spectra, seed=seed)
        # Compute the standard deviation of nosie we need to add to get the desired AR1
        noises = [util.how_much_noise(spectrum, max(.001, ar1)) for ar1 in ar1vals]
        # Add noise to the timeseries
        rng = np.random.RandomState(seed+100000)
        tss += (1-params['ratio']) * rng.multivariate_normal(mean=[0]*tss.shape[0], cov=corr,  size=tss.shape[1]).T * np.asarray(noises).reshape(-1,1)
        tss += params['ratio'] * rng.randn(tss.shape[0], tss.shape[1]) * np.asarray(noises).reshape(-1,1)
        return tss
    @staticmethod
    def transform_dists(distances, params):
        return util.spatial_exponential_floor(distances, params['lmbda'], params['floor'])

class Model_Colorless_Hom(Model_Colorless):
    """Spatially autocorrelated Brownian motion with region-specific Gaussian noise to bring down the region's AR1."""
    name = "ColorlessHom"
    params = {'lmbda': (0.01, 80), 'floor': (0, 1), 'ar1': (.01, .9)}
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, num_timepoints, TR, seed=0, **kwargs):
        ar1vals = np.asarray([params['ar1']]*distance_matrix.shape[0])
        return super().generate_timeseries(distance_matrix=distance_matrix,
                                           params=params, ar1vals=ar1vals,
                                           num_timepoints=num_timepoints,
                                           TR=TR, seed=seed)

class Model_Colorless_Timeonly(Model_Colorless):
    """Brownian motion with region-specific Gaussian noise to bring down the region's AR1."""
    name = "ColorlessTimeonly"
    params = {}
    @staticmethod
    def transform_dists(distances, params):
        return (distances==0).astype(float)


class Model_Spaceonly(Model):
    """White noise with spatial correlations"""
    name = "Spaceonly"
    params = {"lmbda": (0.01, 80), "floor": (0, 1)}
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, num_timepoints, seed=0, **kwargs):
        rng = np.random.RandomState(seed+1000)
        tss = rng.randn(distance_matrix.shape[0], num_timepoints)
        # Turn distances into a covariance matrix
        dist_cm = util.spatial_exponential_floor(distance_matrix, params['lmbda'], params['floor'])
        # Take the matrix square root.  This is a bottleneck.
        dist_cm_sqrt = np.real(scipy.linalg.sqrtm(dist_cm))
        # Apply the template correlation matrix to the timeseries.
        timeseries = np.matmul(dist_cm_sqrt, tss)
        return timeseries

class Model_Colorfull(Model):
    """Spatially autocorrelated 1/f noise to determine AR1."""
    name = "Colorfull"
    # Floor bound is .5 because most higher values give a
    # non-positive-semidefinite covariance matrix, so this speeds things up.
    params = {'lmbda': (0.01, 80), 'floor': (0, .4)}
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, ar1vals, num_timepoints, TR, highpass_freq, seed=0):
        assert num_timepoints % 2 == 0, "Must be even timeseries length"
        # Determine the pink noise exponent alpha from the AR1
        alphas = [util.ar1_to_alpha_fast(TR=TR, tslen=num_timepoints, highpass_freq=highpass_freq, target_ar1=max(0,ar1)) for ar1 in ar1vals]
        # Use these alpha values to construct desired frequency spectra
        spectra = np.asarray([util.make_spectrum(tslen=num_timepoints, TR=TR, alpha=alpha, highpass_freq=highpass_freq) for alpha in alphas])
        # Spatial autocorrelation matrix
        corr = cls.transform_dists(distance_matrix, params)
        # Compute timeseries from desired correlation matrix and frequency spectra
        tss = util.spatial_temporal_timeseries(cm=corr, spectra=spectra, seed=seed)
        return tss
    @staticmethod
    def transform_dists(distances, params):
        return np.exp(-distances/params['lmbda'])*(1-params['floor'])+params['floor']


class Model_Colorfull_Hom(Model_Colorfull):
    """Spatially autocorrelated 1/f noise to set the AR1."""
    name = "ColorfullHom"
    params = {'lmbda': (0.01, 80), 'floor': (0, 1), 'ar1': (.01, .9)}
    @classmethod
    def generate_timeseries(cls, distance_matrix, params, num_timepoints, TR, seed=0, **kwargs):
        ar1vals = np.asarray([params['ar1']]*distance_matrix.shape[0])
        return super().generate_timeseries(distance_matrix=distance_matrix,
                                           params=params, ar1vals=ar1vals,
                                           num_timepoints=num_timepoints,
                                           TR=TR, seed=seed)


class Model_Colorfull_Timeonly(Model_Colorfull):
    name = "ColorfullTimeonly"
    params = {}
    @staticmethod
    def transform_dists(distances, params):
        return (distances==0).astype(float)

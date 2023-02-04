import datasets
import util
import os
import numpy as np
import models
import pandas
import scipy.linalg
import util
import hashlib
from config import CACHE_DIR, HCP_GEODESIC_R_FILENAME, HCP_GEODESIC_L_FILENAME, HCP_DEMOGRAPHICS


#################### Base ####################

class Dataset:
    name = "" # Must be unique and non-empty for each subclass
    disc = 1 # Discretization for computing lmbda/floor
    highpass = .01 # Avoid extremely low frequencies during model fitting
    def __init__(self, *args, **kwargs):
        self._setup(*args, **kwargs)
        self._init_cache()
        assert self.name != "", "Please specify a name"
    # Implement these functions when you subclass Dataset!
    def _setup(self):
        raise NotImplementedError
    def _get_timeseries(self):
        raise NotImplementedError
    def _get_positions(self):
        raise NotImplementedError
    def _get_movement(self):
        raise NotImplementedError
    def _get_subject_info(self):
        raise NotImplementedError
    def _get_parcelareas(self):
        raise NotImplementedError
    def _get_dropped_frames(self):
        raise NotImplementedError
    # Cache management functions
    _cache_keys_memory_only = ['timeseries', 'dists']
    _cache_keys_compress = ['matrices', 'thresh']
    _cache_keys_special = ['n'] # Not stored as a list of subjects
    def _init_cache(self):
        self._memory_cache = {}
        self._cache_dir = os.path.join(CACHE_DIR, self.name)
        if not os.path.isdir(self._cache_dir):
            os.mkdir(self._cache_dir)
    def _cache_path(self, key):
        cachepath = os.path.join(self._cache_dir, key)
        if any(key.startswith(k) for k in self._cache_keys_compress):
            cachepath += ".npz"
        return cachepath
    def _in_cache(self, key):
        if key in self._memory_cache.keys():
            return True
        if os.path.isfile(self._cache_path(key)):
            return True
        return False
    def _from_cache(self, key):
        assert self._in_cache(key), f"Key {key} not in cache"
        if key not in self._memory_cache.keys():
            if any(key.startswith(k) for k in self._cache_keys_compress):
                fh = np.load(self._cache_path(key))
                cacheval = fh.get('v')
                fh.close()
            else:
                cacheval = util.pload(self._cache_path(key))
            self._memory_cache[key] = cacheval
        return self._memory_cache[key]

        raise ValueError("Invalid key")
    def _set_cache(self, key, value):
        if any(key.startswith(k) for k in self._cache_keys_compress):
            np.savez_compressed(self._cache_path(key), v=value)
        elif key not in self._cache_keys_memory_only:
            util.psave(self._cache_path(key), value)
        self._memory_cache[key] = value
    # Functions to do heavy lifting
    def _cache_lmbdaparams(self, disc):
        matrices = self.get_matrices()
        lmbdaparams = [util.get_cm_lmbda_params(m, self.get_dists(), disc) for m in matrices]
        self._set_cache('lmbda'+str(disc), [lp[0] for lp in lmbdaparams])
        self._set_cache('floor'+str(disc), [lp[1] for lp in lmbdaparams])
    def _cache_Ns(self):
        # Workaround because a few timeseries are shorter than the rest
        tss = self.get_timeseries()
        assert len(set([ts.shape[0] for ts in tss])) == 1, "Different number of regions"
        try:
            dropped_frames = self._get_dropped_frames()
            has_dropped_frames = True
        except NotImplementedError:
            has_dropped_frames = False
        if not has_dropped_frames:
            tslens = [ts.shape[1] for ts in tss]
            assert len(set(tslens)) == 1, "Dataset not all the same size"
            shape = (len(tss), tss[0].shape[0], tss[0].shape[1])
        else:
            tslens = [ts.shape[1]+len(df) for ts,df in zip(tss,dropped_frames)]
            assert len(set(tslens)) == 1, "Dataset not all the same size"
            shape = (len(tss), tss[0].shape[0], tslens[0])
        self._set_cache('n', shape)
    # Functions to get interesting properties of the dataset
    def get_timeseries(self):
        if not self._in_cache('timeseries'):
            self._set_cache('timeseries', self._get_timeseries())
        return self._from_cache('timeseries')
    def get_matrices(self):
        return [util.correlation_matrix_pearson(ts) for ts in self.get_timeseries()]
    def get_positions(self):
        if not self._in_cache('positions'):
            self._set_cache('positions', self._get_positions())
        return self._from_cache('positions')
    def get_dists(self):
        return util.distance_matrix_euclidean(self.get_positions())
    def _get_ar1s(self):
        # If scrubbing was not applied, we can directly compute the AR1 of the
        # timeseries
        if not self.has_dropped_frames():
            return util.get_ar1s(self.get_timeseries())
        # If scrubbing was applied, we break the timeseries into segments based
        # on the indices where there were dropped frames.
        else:
            ar1s = []
            timeseries = self.get_timeseries()
            dropped_frames = self.get_dropped_frames()
            for subj in range(0, self.N_subjects()):
                subj_dropped_frames = np.asarray([-1]+dropped_frames[subj]+[self.N_timepoints()])
                segment_lengths = subj_dropped_frames[1:]-subj_dropped_frames[:-1]-1
                inds = [sum(segment_lengths[0:i]) for i in range(0, len(segment_lengths))]+[self.N_timepoints()]
                segments = [timeseries[subj][:,inds[i-1]:inds[i]] for i in range(1, len(inds))]
                segments_valid = [s for s in segments if s.shape[1] > 2]
                sum_valid_lengths = sum([s.shape[1] - 1 for s in segments_valid])
                # print(inds)
                # print(segment_lengths)
                # print([s.shape[1] - 1 for s in segments_valid])
                # if np.all(segment_lengths == [352,   0,   0,  24,  20]):
                #     import matplotlib.pyplot as plt
                #     plt.imshow(segments_valid[1], aspect='auto')
                #     plt.show()
                #     plt.imshow(segments_valid[2], aspect='auto')
                #     plt.show()
                ar1s_segments = np.sum([util.get_ar1s(s) * (s.shape[1]-1) / sum_valid_lengths for s in segments_valid], axis=0)
                ar1s.append(ar1s_segments)
            return np.asarray(ar1s)
    def get_ar1s(self):
        if not self._in_cache('ar1'):
            self._set_cache('ar1', self._get_ar1s())
        return self._from_cache('ar1')
    def get_lmbda(self, disc=None):
        disc = disc or self.disc
        if not self._in_cache('lmbda'+str(disc)):
            self._cache_lmbdaparams(disc)
        return self._from_cache('lmbda'+str(disc))
    def get_floor(self, disc=None):
        disc = disc or self.disc
        if not self._in_cache('floor'+str(disc)):
            self._cache_lmbdaparams(disc)
        return self._from_cache('floor'+str(disc))
    def get_thresh(self, density=10):
        if not self._in_cache('thresh'+str(density)):
            matrices = self.get_matrices()
            thresh = np.asarray([util.threshold_matrix(m, density/100) for m in matrices])
            self._set_cache('thresh'+str(density), thresh)
        return self._from_cache('thresh'+str(density))
    def get_metrics(self, by_metric=True, density=10):
        if not self._in_cache('metrics'+str(density)):
            thresh = self.get_thresh(density)
            metrics = [util.graph_metrics_from_adj(m) for m in thresh]
            self._set_cache('metrics'+str(density), metrics)
        cached = self._from_cache('metrics'+str(density))
        if by_metric:
            return {k : [c[k] for c in cached] for k in cached[0].keys()}
        return cached
    def get_cmstats(self, by_metric=True):
        if not self._in_cache('cmstats'):
            matrices = self.get_matrices()
            metrics = [util.cm_metrics_from_cm(m) for m in matrices]
            self._set_cache('cmstats', metrics)
        cached = self._from_cache('cmstats')
        if by_metric:
            return {k : [c[k] for c in cached] for k in cached[0].keys()}
        return cached
    def get_nodal_metrics(self, density=10):
        if not self._in_cache('nodal_metrics'+str(density)):
            nodal = [util.nodal_graph_metrics_from_adj(m) for m in self.get_thresh(density)]
            nodal_dict = {k : np.asarray([v[k] for v in nodal]) for k in nodal[0].keys()}
            self._set_cache('nodal_metrics'+str(density), nodal_dict)
        return self._from_cache('nodal_metrics'+str(density))
    def get_nodal_cmstats(self):
        if not self._in_cache('nodal_cmstats'):
            nodal = [util.nodal_ts_and_cm_metrics(m) for m in self.get_timeseries()]
            nodal_dict = {k : np.asarray([v[k] for v in nodal]) for k in nodal[0].keys()}
            self._set_cache('nodal_cmstats', nodal_dict)
        return self._from_cache('nodal_cmstats')
    def get_subject_info(self):
        if not self._in_cache('subject_info'):
            self._set_cache('subject_info', self._get_subject_info())
        return self._from_cache('subject_info')
    def get_movement(self):
        if not self._in_cache('movement'):
            self._set_cache('movement', self._get_movement())
        return self._from_cache('movement')
    def get_parcelareas(self):
        if not self._in_cache('parcelareas'):
            self._set_cache('parcelareas', self._get_parcelareas())
        return self._from_cache('parcelareas')
    def get_eigenvalues(self):
        if not self._in_cache('eigenvalues'):
            eigenvalues = np.asarray([util.get_eigenvalues(cm) for cm in self.get_matrices()])
            self._set_cache('eigenvalues', eigenvalues)
        return self._from_cache('eigenvalues')
    def get_dropped_frames(self):
        if not self._in_cache('dropped_frames'):
            try:
                df = self._get_dropped_frames()
            except NotImplementedError:
                df = [[]]*self.N_subjects()
            self._set_cache('dropped_frames', df)
        return self._from_cache('dropped_frames')
    def has_dropped_frames(self):
        return not all([len(fs) == 0 for fs in self.get_dropped_frames()])
    def get_long_memory(self):
        if self.has_dropped_frames():
            raise NotImplementedError("Cannot compute long memory constant with dropped frames")
        if not self._in_cache('long_memory'):
            longmem = [util.get_long_memory(self.get_timeseries()[i], minscale=self.minwave) for i in range(0, self.N_subjects())]
            self._set_cache('long_memory', longmem)
        return self._from_cache('long_memory')
    def N_subjects(self):
        if not self._in_cache('n'):
            self._cache_Ns()
        return self._from_cache('n')[0]
        return len(self.get_thresh())
    def N_regions(self):
        if not self._in_cache('n'):
            self._cache_Ns()
        return self._from_cache('n')[1]
    def N_timepoints(self):
        if not self._in_cache('n'):
            self._cache_Ns()
        return self._from_cache('n')[2]
    def get_valid(self):
        return np.ones(self.N_subjects()).astype('bool')
    # Convenience methods
    def cache_all(self):
        def tryfn(f):
            try:
                f()
            except NotImplementedError:
                pass
        tryfn(self.N_timepoints)
        tryfn(self.get_ar1s)
        tryfn(self.get_lmbda)
        tryfn(self.get_eigenvalues)
        tryfn(self.get_positions)
        tryfn(self.get_thresh)
        tryfn(self.get_cmstats)
        tryfn(self.get_metrics)
        tryfn(self.get_nodal_metrics)
        tryfn(self.get_nodal_cmstats)
        tryfn(self.get_subject_info)
        tryfn(self.get_movement)
        tryfn(self.get_parcelareas)


#################### Actual datasets ####################

class HCP(Dataset):
    TR = 720
    highpass = .01
    minwave = 1
    def _setup(self, scan=0, gsr=False):
        self.name = f"hcp{scan}{'gsr' if gsr else ''}"
        self.scan = scan
        self.gsr = gsr
    def _get_timeseries(self):
        if self.gsr:
            return datasets.get_hcpgsr_timeseries(self.scan)
        else:
            return datasets.get_hcp_timeseries(self.scan)
    def _get_positions(self):
        return datasets.get_hcp_positions()
    def _get_movement(self):
        return datasets.get_hcp_movement(self.scan)
    def _get_subject_info(self):
        return datasets.get_hcp_demographics()
    def _get_parcelareas(self):
        return np.asarray(datasets.get_hcp_parcelstats()['areas'])

class HCP1200(Dataset):
    TR = 720
    highpass = .01
    minwave = 1
    def _setup(self, scan=0, gsr=False):
        self.name = f"hcp1200{scan}{'gsr' if gsr else ''}"
        self.scan = scan
        self.gsr = gsr
    def _get_timeseries(self):
        return datasets.get_hcp1200_timeseries(self.scan, gsr=self.gsr)
    def _get_positions(self):
        return datasets.get_hcp1200_positions()
    def _get_movement(self):
        return datasets.get_hcp1200_movement(scan=self.scan, gsr=self.gsr)
    def _get_subject_info(self):
        return datasets.get_hcp1200_demographics(gsr=self.gsr)
    def _cache_Ns(self):
        # Workaround because a few timeseries in the GSR version are shorter than 1200
        tss = self.get_timeseries()
        shape = (len(tss), *tss[0].shape)
        self._set_cache('n', shape)
    def _get_parcelareas(self):
        return np.asarray(datasets.get_hcp_parcelstats()['areas'])
    def get_rehos(self):
        return datasets.get_hcp1200_rehos(scan=self.scan)

class HCP1200Geo(HCP1200):
    def _setup(self, scan=0, gsr=False, hemi="R"):
        assert hemi in ["R", "L"]
        super()._setup(scan=scan, gsr=gsr)
        self.name += "geo" + hemi
        self.hemi = hemi
        if hemi != "R":
            raise NotImplementedError("Not yet available for left hemisphere")
    def _get_timeseries(self):
        tss = np.asarray(super()._get_timeseries())
        if self.hemi == "R":
            return tss[:,:180]
        else:
            return tss[:,180:]
    def get_dists(self):
        if self.hemi == "R":
            return np.loadtxt(HCP_GEODESIC_R_FILENAME)
        elif self.hemi == "L":
            return np.loadtxt(HCP_GEODESIC_L_FILENAME)
        return util.distance_matrix_euclidean(self.get_positions())
    def _get_positions(self):
        raise NotImplementedError("Positions not valid for geodesic")
    def _get_parcelareas(self):
        pas = super()._get_parcelareas()
        if self.hemi == "R":
            return pas[:180]
        elif self.hemi == "L":
            return pas[180:]


class TRT(Dataset):
    disc = 5
    TR = 1000
    highpass = .01
    minwave = 2
    def _setup(self):
        self.name = f"trt"
    def _cache_Ns(self):
        # Workaround because a few timeseries seem shorter than the others in this dataset
        tss = self.get_timeseries()
        shape = (len(tss), *tss[0].shape)
        self._set_cache('n', shape)
    def _get_timeseries(self):
        return datasets.get_trt_timeseries()
    def _get_positions(self):
        return datasets.get_trt_positions()
    def _get_movement(self):
        return datasets.get_trt_movement()
    def _get_subject_info(self):
        subjinfo = datasets._get_trt_scaninfo()
        return pandas.DataFrame(subjinfo, columns=["subject", "session", "scanner", "run", "sex", "age"])
    def _get_parcelareas(self):
        return datasets.get_trt_parcelareas()
    def get_rehos(self):
        return datasets.get_trt_rehos()

class CamCan(Dataset):
    disc = 5
    highpass = 0
    minwave = 1
    def _setup(self, dataset="Rest", parcellation="AAL"):
        assert dataset in ["Rest", "SMT", "Movie"]
        assert parcellation in ["AAL", "OSL", "HOA116", "Craddock"]
        self.name = f"camcan{dataset}{parcellation}"
        self.dataset = dataset
        self.parcellation = parcellation
        self.TR = 1970 if dataset in ["Rest", "SMT"] else 2470
    def _get_timeseries(self):
        # We cut off the last time point, because we want all even length timeseries
        return datasets.get_camcan_timeseries(dataset=self.dataset, parcellation=self.parcellation)[:,:,:-1]
    def _get_positions(self):
        return datasets.get_camcan_positions(parcellation=self.parcellation)
    def _get_movement(self):
        return datasets.get_camcan_movement(dataset=self.dataset, parcellation=self.parcellation)
    def _get_subject_info(self):
        subjinfo = datasets.get_camcan_behavior()
        return pandas.DataFrame(subjinfo)
    def _get_parcelareas(self):
        return datasets.get_camcan_parcelareas(self.parcellation)


class CamCanFiltered(CamCan):
    disc = 5
    minwave = 2
    def _setup(self, dataset="Rest", parcellation="AAL"):
        super()._setup(dataset=dataset, parcellation=parcellation)
        self.name = f"camcanfiltered{dataset}{parcellation}"
        nyquist = 1/(self.TR/1000)/2
        butter = scipy.signal.iirfilter(ftype='butter', N=2, Wn=nyquist/2, btype='lowpass', fs=1000/self.TR, output='sos')
        self.filt = lambda x, butter=butter : scipy.signal.sosfiltfilt(butter, x)
    def _get_timeseries(self):
        # We cut off the last time point, because we want all even length timeseries
        tss = datasets.get_camcan_timeseries(dataset=self.dataset, parcellation=self.parcellation)[:,:,:-1]
        return self.filt(tss)


class LSD(Dataset):
    TR = 2500
    #highpass = .01
    #minwave = 1
    def _setup(self, exp, timepoint, gsr):
        self.name = f"lsd{exp}_{timepoint}_{'gsr' if gsr else 'nogsr'}"
        self.exp = exp
        self.timepoint = timepoint
        self.gsr = gsr
    def _get_timeseries(self):
        return datasets.get_lsd_timeseries(exp=self.exp, timepoint=self.timepoint, gsr=self.gsr)
    def _get_dropped_frames(self):
        return datasets.get_lsd_dropped_frames(exp=self.exp, timepoint=self.timepoint)
    # Uses same parcellation as HCP
    def _get_positions(self):
        return datasets.get_hcp_positions()
    def _get_parcelareas(self):
        return np.asarray(datasets.get_hcp_parcelstats()['areas'])
    def _get_movement(self):
        return datasets.get_lsd_movement(exp=self.exp, timepoint=self.timepoint)
    def _get_subject_info(self):
        return datasets.get_lsd_subjectinfo()

class Psilocybin(Dataset):
    TR = 2430
    #highpass = .01
    #minwave = 1
    def _setup(self, exp, timepoint, gsr):
        self.name = f"psilocybin{exp}_{timepoint}_{'gsr' if gsr else 'nogsr'}"
        self.exp = exp
        self.timepoint = timepoint
        self.gsr = gsr
    def _get_timeseries(self):
        return datasets.get_psilocybin_timeseries(exp=self.exp, timepoint=self.timepoint, gsr=self.gsr)
    def _get_dropped_frames(self):
        return datasets.get_psilocybin_dropped_frames(exp=self.exp, timepoint=self.timepoint)
    # Uses same parcellation as HCP
    def _get_positions(self):
        return datasets.get_hcp_positions()
    def _get_parcelareas(self):
        return np.asarray(datasets.get_hcp_parcelstats()['areas'])
    def _get_movement(self):
        return datasets.get_psilocybin_movement(exp=self.exp, timepoint=self.timepoint)
    def _get_subject_info(self):
        return datasets.get_psilocybin_subjectinfo()


#################### Derivatives ####################

# class HCPLikeTRT(Dataset):
#     TR = 720
#     highpass = .01
#     def _setup(self, gsr=False):
#         self.name = f"hcpliketrt{'gsr' if gsr else ''}"
#         self.gsr = gsr
#     def _get_timeseries(self):
#         alltss = []
#         for scan in [0, 1, 2, 3]:
#             if self.gsr:
#                 tss = datasets.get_hcpgsr_timeseries(scan)
#             else:
#                 tss = datasets.get_hcp_timeseries(scan)
#             for ts in tss:
#                 # I know this is the wrong length by 1, but the parcellation
#                 # size is also 360 and so making the timeseries length 361
#                 # instead will help make sure there are no matrix transposition
#                 # errors passed silently
#                 alltss.append(ts[:,0:361])
#                 alltss.append(ts[:,369:730])
#                 alltss.append(ts[:,739:1100])
#         return alltss
#     def _get_positions(self):
#         return datasets.get_hcp_positions()
#     def _get_subject_info(self):
#         demo = datasets.get_hcp_demographics()
#         subjs = np.tile(np.repeat(list(demo.index), 3), 4)
#         run = np.tile([1, 2, 3], 4*len(demo))
#         scan = np.repeat([1, 2, 3, 4], 3*len(demo))
#         return pandas.DataFrame({"subject": subjs, "run": run, "scan": scan})

class TRTKindaLikeHCP(Dataset):
    TR = 1000
    highpass = .01
    disc = 5
    def _setup(self, run=0):
        self.run = run
        self.name = f"trt{self.run}kindalikehcp"
    def _init_cache(self):
        super()._init_cache()
        self.trt = TRT()
        self.subject_info_cache = None
        self.inds = np.asarray(self.trt.get_subject_info()['run']==(self.run+1))
    def _in_cache(self, key):
        return True
    def _from_cache(self, key):
        if key == 'n':
            return (self.trt.N_subjects()//6, self.trt.N_regions(), self.trt.N_timepoints())
        if not self.trt._in_cache(key):
            self.trt.cache_all()
        if 'nodal' in key:
            d = self.trt._from_cache(key)
            return {k : np.asarray(v)[self.inds] for k,v in d.items()}
        else:
            vals = self.trt._from_cache(key)
            return np.asarray(vals)[self.inds]
    def _get_positions(self):
        return self.trt.get_positions()
    def get_subject_info(self):
        trt_info = self.trt.get_subject_info()
        return trt_info[np.asarray(trt_info['run']==1)].reset_index(drop=True)
    def get_rehos(self):
        orig_rehos = self.trt.get_rehos()
        trt_info = self.trt.get_subject_info()
        inds = np.asarray(self.trt.get_subject_info()['run'] == self.run+1)
        return orig_rehos[inds]

class Join(Dataset):
    def _setup(self, bases, name=None):
        for b in bases:
            assert b.N_subjects() == bases[0].N_subjects()
            assert b.N_regions() == bases[0].N_regions()
            assert b.N_timepoints() == bases[0].N_timepoints()
            assert np.all(b.get_positions() == bases[0].get_positions())
        self.bases = bases
        if name:
            self.name = name
        else:
            self.name = "_".join([b.name for b in bases])
    def _init_cache(self):
        pass
    def _in_cache(self, key):
        return True
    def _from_cache(self, key):
        for b in self.bases:
            if not b._in_cache(key):
                b.cache_all()
        if 'nodal' in key:
            ds = [b._from_cache(key) for b in self.bases]
            return {k : np.asarray(sum([list(d[k]) for d in ds], [])) for k,_ in ds[0].items()}
        else:
            return sum([list(b._from_cache(key)) for b in self.bases], []) # Concatenate lists
    def _set_cache(self, key, value):
        print(f"No cache to set")
    def get_positions(self):
        return self.bases[0].get_positions()
    def N_regions(self):
        return self.bases[0].N_regions()
    def N_timepoints(self):
        return self.bases[0].N_timepoints()
    def N_subjects(self):
        return sum([b.N_subjects() for b in self.bases])
    def get_subject_info(self):
        dfs = [b.get_subject_info() for b in self.bases]
        for i,df in enumerate(dfs):
            df['dataset'] = i
            df['dataset_name'] = self.bases[i].name
            #df = df.reset_index("subject_order_id")
        return pandas.concat(dfs)
    def get_parcelareas(self):
        return self.bases[0].get_parcelareas()

class Subset(Dataset):
    def _setup(self, name, base, inds):
        self.base = base
        self.name = name+"_"+base.name
        self.inds = inds
        self.disc = base.disc
        assert sum(self.inds) > 0, "No scans selected"
    def _init_cache(self):
        self._memory_cache = {}
    def _in_cache(self, key):
        return self.base._in_cache(key)
    def _from_cache(self, key):
        if not self.base._in_cache(key):
            self.base.cache_all()
        if 'nodal' in key:
            d = self.base._from_cache(key)
            return {k : np.asarray(v)[self.inds] for k,v in d.items()}
        else:
            vals = self.base._from_cache(key)
            return [v for i,v in enumerate(vals) if self.inds[i]]
    def _set_cache(self, key, value):
        print(f"Warning, memory cache only for {key}")
        self._memory_cache[key] = value
    def get_positions(self):
        return self.base.get_positions()
    def N_regions(self):
        return self.base.N_regions()
    def N_timepoints(self):
        return self.base.N_timepoints()
    def N_subjects(self):
        return np.sum(self.inds)
    def get_valid(self):
        valid_base = self.base.get_valid()
        return valid_base[self.inds]
    def get_timeseries(self):
        return self.base.get_timeseries()[self.inds]


class HCPKindaLikeTRT(Dataset):
    TR = 720
    highpass = .01
    version = ""
    def _setup(self, gsr=False, hemi=None):
        self.name = f"hcp{self.version}kindaliketrt{'gsr' if gsr else ''}"
        self.gsr = gsr
        self.hemi = hemi
    def _init_cache(self):
        super()._init_cache()
        if self.version == "":
            self.hcps = [HCP(0, gsr=self.gsr), HCP(1, gsr=self.gsr),
                         HCP(2, gsr=self.gsr), HCP(3, gsr=self.gsr)]
        elif self.version == "1200":
            self.hcps = [HCP1200(0, gsr=self.gsr), HCP1200(1, gsr=self.gsr),
                         HCP1200(2, gsr=self.gsr), HCP1200(3, gsr=self.gsr)]
        elif self.version == "1200Geo":
            self.hcps = [HCP1200Geo(0, gsr=self.gsr, hemi=self.hemi), HCP1200Geo(1, gsr=self.gsr, hemi=self.hemi),
                         HCP1200Geo(2, gsr=self.gsr, hemi=self.hemi), HCP1200Geo(3, gsr=self.gsr, hemi=self.hemi)]
        self.subject_info_cache = None
    def _in_cache(self, key):
        return True
    def _from_cache(self, key):
        if key == 'n':
            return (self.hcps[0].N_subjects()*4, self.hcps[0].N_regions(), self.hcps[0].N_timepoints())
        if not all(hcp._in_cache(key) for hcp in self.hcps):
            if key in ["timeseries", "matrices"]:
                for hcp in self.hcps:
                    hcp.get_timeseries()
            else:
                for hcp in self.hcps:
                    hcp.cache_all()
        if 'nodal' in key:
            dicts = [hcp._from_cache(key) for hcp in self.hcps]
            vals = {}
            for k in dicts[0].keys():
                vals[k] = []
            for d in dicts:
                for k,v in d.items():
                    vals[k].extend(v)
            for k in dicts[0].keys():
                vals[k] = np.asarray(vals[k])
            return vals
        else:
            vals = []
            for hcp in self.hcps:
                vals.extend(hcp._from_cache(key))
            # Because HCP-GSR has some timeseries of different lengths
            if key == "timeseries":
                return vals
            else:
                return np.asarray(vals)
    def _get_positions(self):
        return datasets.get_hcp_positions()
    def _get_subject_info(self):
        demo = self.hcps[0].get_subject_info()
        subjs = np.tile(list(demo.index), 4)
        scan = np.repeat([1, 2, 3, 4], len(demo))
        run = np.repeat([1, 2, 1, 2], len(demo))
        session = np.repeat([1, 1, 2, 2], len(demo))
        return pandas.DataFrame({"subject": subjs, "run": run, "scan": scan})
    def get_subject_info(self):
        if not super()._in_cache('subject_info'):
            super()._set_cache('subject_info', self._get_subject_info())
        return super()._from_cache('subject_info')
    def get_rehos(self):
        assert np.all(np.asarray(self.get_subject_info()['subject'][(2*883):(3*883)]) == self.get_subject_info()['subject'][(1*883):(2*883)])
        assert np.all(np.asarray(self.get_subject_info()['subject'][(0*883):(1*883)]) == self.get_subject_info()['subject'][(1*883):(2*883)])
        assert np.all(np.asarray(self.get_subject_info()['subject'][(3*883):(4*883)]) == self.get_subject_info()['subject'][(1*883):(2*883)])
        return np.concatenate([hcp.get_rehos() for hcp in self.hcps])

class HCP1200KindaLikeTRT(HCPKindaLikeTRT):
    version = "1200"

class HCP1200GeoKindaLikeTRT(HCPKindaLikeTRT):
    version = "1200Geo"
    def _get_positions(self):
        raise NotImplementedError
    def get_dists(self):
        return self.hcps[0].get_dists()

def HCP1200Unrelated(*args, **kwargs):
    base = HCP1200(*args, **kwargs)
    demo = pandas.read_excel(HCP_DEMOGRAPHICS).reset_index()
    subjs = demo.query("N334 == True")['Subject']
    inds = np.isin(base.get_subject_info().reset_index()['Subject'], subjs)
    return Subset("hcpunrelated", base, inds)

def HCP1200UnrelatedKindaLikeTRT(*args, **kwargs):
    base = HCP1200KindaLikeTRT(*args, **kwargs)
    demo = pandas.read_excel(HCP_DEMOGRAPHICS).reset_index()
    subjs = demo.query("N334 == True")['Subject']
    inds = np.isin(base.get_subject_info()['subject'], subjs)
    return Subset("hcpunrelatedkindaliketrt", base, inds)

class MakeTiny(Dataset):
    """Subset the Dataset `base` to have only `size` subjects."""
    def _setup(self, base, size):
        self.base = base
        self.size = size
        self.name = f"tiny_{self.size}_{self.base.name}"
        self.TR = base.TR
        self.highpass = base.highpass
    def get_timeseries(self):
        return self.base.get_timeseries()[0:self.size]
    def N_subjects(self):
        return self.size
    def get_positions(self):
        return self.base.get_positions()
    def _in_cache(self, key):
        return self.base._in_cache(key)
    def _set_cache(self, key, value):
        raise NotImplementedError("Please cache in original dataset")
    def _from_cache(self, key):
        if key in self._cache_keys_special:
            return self.base._from_cache(key)
        if "nodal" in key:
            return {k : v[0:self.size] for k,v in self.base._from_cache(key).items()}
        else:
            return self.base._from_cache(key)[0:self.size]
    def _init_cache(self):
        pass


#################### Non-generative surrogates ####################

class Surrogate(Dataset):
    def _setup(self, surrogate_of, seed=0):
        self.name = f"{self.surrogate_name}_{surrogate_of.name}_seed{seed}"
        self.surrogate_of = surrogate_of
        self.seed = seed
    def N_regions(self):
        return self.surrogate_of.N_regions()
    def N_subjects(self):
        return self.surrogate_of.N_subjects()
    def N_timepoints(self):
        return self.surrogate_of.N_timepoints()
    def _get_positions(self):
        return self.surrogate_of.get_positions()
    def get_dists(self):
        return self.surrogate_of.get_dists()
    def get_matrices(self):
        if self.cache_matrices:
            if not self._in_cache('matrices'):
                self._set_cache('matrices', self._get_matrices())
            return self._from_cache('matrices')
        else:
            return super().get_matrices()


class Eigensurrogate(Surrogate):
    surrogate_name = "eigensurrogate"
    cache_matrices = True
    def _get_timeseries(self):
        matrices = self._get_matrices()
        N_regions = self.surrogate_of.N_regions()
        N_timepoints = self.surrogate_of.N_timepoints()
        tss = []
        rng = np.random.RandomState(self.seed+1000)
        for m in matrices:
            msqrt = scipy.linalg.sqrtm(m)
            ts = msqrt @ rng.randn(N_regions, N_timepoints)
            tss.append(ts)
        return np.asarray(tss)
    def _get_matrices(self):
        orig_eigs = self.surrogate_of.get_eigenvalues()
        rng = np.random.RandomState(self.seed)
        surrogates = []
        for i in range(0, self.surrogate_of.N_subjects()):
            desired_evs = orig_eigs[i]
            if min(desired_evs) < 0:
                desired_evs = np.maximum(0, desired_evs)
                newsum = np.sum(desired_evs)
                desired_evs[0] -= (newsum - len(desired_evs))
                print(f"Warning: eigenvalues were less than zero in source matrix by {newsum-len(desired_evs)}")
            m = scipy.stats.random_correlation.rvs(eigs=desired_evs, tol=1e-12, random_state=rng)
            np.fill_diagonal(m, 1)
            surrogates.append(util.make_perfectly_symmetric(m))
        return surrogates

class PhaseRandomize(Surrogate):
    surrogate_name = "phase"
    cache_matrices = False
    def _get_timeseries(self):
        tss = self.surrogate_of.get_timeseries()
        rng = np.random.RandomState(self.seed)
        surrogates = [util.phase_randomize(ts, seed=rng.randint(0, 1e7)) for ts in tss]
        return surrogates

class DegreeRandomize(Surrogate):
    surrogate_name = "degreerand"
    cache_matrices = False
    def get_thresh(self, density=10):
        if not self._in_cache('thresh'+str(density)):
            thresh = self.surrogate_of.get_thresh(density)
            rng = np.random.RandomState(self.seed)
            surrogates = [util.degree_preserving_randomization(adj, seed=rng.randint(0, 1e7)) for adj in thresh]
            self._set_cache('thresh'+str(density), surrogates)
        return self._from_cache('thresh'+str(density))

class Zalesky2012(Surrogate):
    surrogate_name = "zalesky"
    cache_matrices = True
    def _get_matrices(self):
        N_regions = self.surrogate_of.N_regions()
        desired_means = self.surrogate_of.get_cmstats()['meancor']
        desired_vars = self.surrogate_of.get_cmstats()['varcor']
        rng = np.random.RandomState(self.seed)
        surrogates = []
        for desired_mean,desired_var in zip(desired_means,desired_vars):
            nmax = 1000
            nmin = 2
            while nmax - nmin > 1:
                n = int(np.floor(nmin + (nmax-nmin)/2))
                rho = self._fitmean(desired_mean, n, N_regions, rng)
                muhat = np.mean(rho[np.triu_indices(N_regions, 1)])
                sigma2hat = np.var(rho[np.triu_indices(N_regions, 1)])
                if sigma2hat > desired_var:
                    nmin = n
                else:
                    nmax = n
            surrogates.append(util.make_perfectly_symmetric(rho))
        return surrogates
    @staticmethod
    def _fitmean(mu, n, N, rng): # N = number of regions, n = number of timepoints, rng=random number generator, mu = desired mean
        x = rng.randn(N, n) # Each ROW is a different region.  This is inconsistent with the paper.
        y = rng.randn(n, 1)
        amax = 10
        amin = 0
        while np.abs(amax-amin) > .001:
            a = amin + (amax-amin)/2
            rho = np.corrcoef(x+a*(y@np.ones((1, N))).T)
            assert rho.shape[0] == N, "Bad shape"
            muhat = np.mean(rho[np.triu_indices(N, 1)])
            if muhat > mu:
                amax = a
            else:
                amin = a
        return rho


#################### Generative models ####################

class _WrapModel(Dataset):
    _model_to_wrap = None
    def _setup(self, fit_of, params, seed=0):
        assert len(params) == fit_of.N_subjects(), "Invalid parameters"
        self.fit_of = fit_of
        self.seed = seed
        self.params = params
        params_hash = hashlib.md5(str(tuple(map(lambda x : tuple(sorted(x.items())), params))).encode()).hexdigest()
        self.name = f"{self._model_to_wrap.name}_{fit_of.name}_{params_hash}_seed{seed}"
    def _get_positions(self):
        return self.fit_of.get_positions()
    def _get_timeseries(self):
        ar1s = self.fit_of.get_ar1s()
        dist = self.fit_of.get_dists()
        tss = []
        for i,p in enumerate(self.params):
            print(i,p)
            ts = self._model_to_wrap.generate_timeseries(distance_matrix=dist, params=p, ar1vals=ar1s[i], num_timepoints=self.fit_of.N_timepoints(), TR=self.fit_of.TR/1000, highpass_freq=self.fit_of.highpass, seed=self.seed)
            tss.append(ts)
        return tss
    def get_dists(self):
        return self.fit_of.get_dists()

class ModelColorless(_WrapModel):
    _model_to_wrap = models.Model_Colorless

class ModelColorlessCor(_WrapModel):
    _model_to_wrap = models.Model_ColorlessCor

class ModelColorfull(_WrapModel):
    _model_to_wrap = models.Model_Colorfull

class ModelColorlessTimeonly(_WrapModel):
    _model_to_wrap = models.Model_Colorless_Timeonly
    def _setup(self, fit_of, seed=0):
        super()._setup(fit_of=fit_of, params=[{}]*fit_of.N_subjects(), seed=seed)

class ModelColorfullTimeonly(_WrapModel):
    _model_to_wrap = models.Model_Colorfull_Timeonly
    def _setup(self, fit_of, seed=0):
        super()._setup(fit_of=fit_of, params=[{}]*fit_of.N_subjects(), seed=seed)

class ModelSpaceonly(_WrapModel):
    _model_to_wrap = models.Model_Spaceonly

# Summary of models:
#
# - ColorSurrogateCorrelated: Includes noise in power spectrum, but the noise
#   has the same correlation as the exponent. Parameter free.
#
# - ColorfullSurrogate: Fit AR1 using exponent.  No noise added.  Parameter
#   free.
#
# - ColorlessSurrogateTheoretical: Model doesn't work.
#
# - ColorlessHet: Main model, needs parameters


# Need: ColorlessHet (fit), ColorfullHet (fit), ColorfullHet (no-param),
# Time-only, Space only, Zalesky, autoregressive (no-param)


#################### Generative surrogates ####################

class ColorSurrogateCorrelated(Dataset):
    """A variant of the Colorless model which has correlated white noise added.

    This model is useful because we can use it as a surrogate, i.e. we can
    eliminate all parameters.  Note that this doesn't work for all sets of
    power spectra, since the method for ensuring this works can often give
    non-positive-semidefinite matrices.  When this happens, they are replaced
    by a version which sets the floor (i.e. GC) to zero.  Empirically this seem
    to work pretty well.  Use the method "get_valid" for a list of indices
    which did not have their floor set to zero.
    """
    def _setup(self, surrogate_of, seed=0):
        self.name = f"colorsurrogatecorrelated_{surrogate_of.name}_seed{seed}"
        self.surrogate_of = surrogate_of
        self.seed = seed
    def _get_timeseries(self):
        rng = np.random.RandomState(self.seed)
        ar1s = self.surrogate_of.get_ar1s()
        lmbdas = self.surrogate_of.get_lmbda()
        floors = self.surrogate_of.get_floor()
        dist = self.surrogate_of.get_dists()
        N_subjs = self.surrogate_of.N_subjects()
        N_timepoints = self.surrogate_of.N_timepoints()
        TR = self.surrogate_of.TR/1000
        highpass = self.surrogate_of.highpass
        tss = []
        valid = []
        for i in range(0, N_subjs):
            try:
                cm = util.spatial_exponential_floor(dist, lmbda=lmbdas[i], floor=floors[i])
                spectra = np.asarray([util.make_noisy_spectrum(N_timepoints, TR, 2, highpass, max(.01,ar1)) for ar1 in ar1s[i]])
                ts = util.spatial_temporal_timeseries(cm, spectra, seed=rng.randint(0, 1e8))
                tss.append(ts)
                valid.append(i)
            except util.PositiveSemidefiniteError:
                cm = util.spatial_exponential_floor(dist, lmbda=lmbdas[i], floor=0)
                ts = util.spatial_temporal_timeseries(cm, spectra, seed=rng.randint(0, 1e8))
                tss.append(ts)
        self._set_cache("valid", valid)
        return tss
    def _get_positions(self):
        return self.surrogate_of.get_positions()
    def get_valid(self):
        assert self._in_cache('valid'), "Please run get_timeseries() before this function"
        valids = np.zeros(self.N_subjects())
        valids[self._from_cache('valid')] = 1
        return valids.astype('bool')

class ColorSurrogate(Dataset):
    """The Colorfull model, i.e. the no-noise model.

    Generate filtered 1/f^alpha noise with spatial structure and AR1 matched to
    the data.  This model is useful because we can use it as a surrogate,
    i.e. we can eliminate all parameters.  Note that this doesn't work for all
    sets of power spectra, since the method for ensuring this works can often
    give non-positive-semidefinite matrices.  When this happens, they are
    replaced by a version which sets the floor (i.e. GC) to zero.  Empirically
    this seem to work pretty well.  Use the method "get_valid" for a list of
    indices which did not have their floor set to zero.
    """
    def _setup(self, surrogate_of, seed=0):
        self.name = f"colorsurrogatenonoise_{surrogate_of.name}_seed{seed}"
        self.surrogate_of = surrogate_of
        self.seed = seed
    def _get_timeseries(self):
        rng = np.random.RandomState(self.seed)
        ar1s = self.surrogate_of.get_ar1s()
        lmbdas = self.surrogate_of.get_lmbda()
        floors = np.maximum(0, self.surrogate_of.get_floor())
        dist = self.surrogate_of.get_dists()
        N_subjs = self.surrogate_of.N_subjects()
        N_timepoints = self.surrogate_of.N_timepoints()
        TR = self.surrogate_of.TR/1000
        highpass = self.surrogate_of.highpass
        tss = []
        valid = []
        for i in range(0, N_subjs):
            try:
                cm = util.spatial_exponential_floor(dist, lmbda=lmbdas[i], floor=floors[i])
                alphas = [util.ar1_to_alpha_fast(N_timepoints, TR, highpass, max(0, ar1)) for ar1 in ar1s[i]]
                spectra = np.asarray([util.make_spectrum(N_timepoints, TR, alpha, highpass) for alpha in alphas])
                ts = util.spatial_temporal_timeseries(cm, spectra, seed=rng.randint(0, 1e8))
                tss.append(ts)
                valid.append(i)
            except util.PositiveSemidefiniteError:
                cm = util.spatial_exponential_floor(dist, lmbda=lmbdas[i], floor=0)
                ts = util.spatial_temporal_timeseries(cm, spectra, seed=rng.randint(0, 1e8))
                tss.append(ts)
        self._set_cache("valid", valid)
        return tss
    def _get_positions(self):
        return self.surrogate_of.get_positions()
    def get_dists(self):
        return self.surrogate_of.get_dists()
    def get_valid(self):
        assert self._in_cache('valid'), "Please run get_timeseries() before this function"
        valids = np.zeros(self.N_subjects())
        valids[self._from_cache('valid')] = 1
        return valids.astype('bool')

class ColorSurrogateTheoretical(Dataset):
    """The version of the main generative model which does not require fitting parameters.

    In theory, this finds a correlation matrix C' such that, for a desired
    correlation matrix C, you can generate correlated Brownian motion and then
    add uncorrelated noise to bring down the AR1, such that all timeseries have
    the desired correlation matrix C and specified AR1s.  This in theory allows
    matching the spatial and temporal autocorrelation.

    However, this model does not work.  It is implemented here just for the
    sake of completeness.  In practice, this generates covariance matrices
    which are nowhere near positive semidefinite.

    """
    def _setup(self, surrogate_of, seed=0):
        self.name = f"colorsurrogatetheoretical_{surrogate_of.name}_seed{seed}"
        self.surrogate_of = surrogate_of
        self.seed = seed
    def _get_timeseries(self):
        rng = np.random.RandomState(self.seed)
        ar1s = self.surrogate_of.get_ar1s()
        lmbdas = self.surrogate_of.get_lmbda()
        floors = self.surrogate_of.get_floor()
        dist = self.surrogate_of.get_dists()
        N_subjs = self.surrogate_of.N_subjects()
        N_timepoints = self.surrogate_of.N_timepoints()
        N_regions = self.surrogate_of.N_regions()
        TR = self.surrogate_of.TR/1000
        highpass = self.surrogate_of.highpass
        tss = []
        valid = []
        for i in range(0, N_subjs):
            try:
                spectrum = util.make_spectrum(N_timepoints, TR, 2, highpass)
                variance = util.get_spectrum_variance(spectrum)
                noise_amounts = np.asarray([[util.how_much_noise(spectrum, max(0.01,ar1)) for ar1 in ar1s[i]]])**2
                cm = util.spatial_exponential_floor(dist, lmbda=lmbdas[i], floor=floors[i])
                gen_corr = cm * np.sqrt((1 + noise_amounts.T/variance)*(1 + noise_amounts/variance))
                print(variance, list(noise_amounts))
                np.fill_diagonal(gen_corr, 1)
                print(np.linalg.eigvalsh(gen_corr))
                #gen_corr = np.diag(np.diag(gen_corr)**-1) @ gen_corr @ np.diag(np.diag(gen_corr)**-1)
                ts = util.spatial_temporal_timeseries(gen_corr, np.asarray([spectrum]*N_regions), seed=rng.randint(0, 1e8))
                ts += rng.multivariate_normal([0]*N_regions, np.diag(noise_amounts[0]), N_timepoints).T
                tss.append(ts)
                valid.append(i)
            except ValueError:
                print("Didn't work")
                cm = util.spatial_exponential_floor(dist, lmbda=lmbdas[i], floor=0)
                gen_corr = cm * np.sqrt((1 + noise_amounts.T**2/variance)*(1 + noise_amounts**2/variance))
                np.fill_diagonal(gen_corr, 1)
                ts = util.spatial_temporal_timeseries(gen_corr, np.asarray([spectrum]*N_regions), seed=rng.randint(0, 1e8))
                ts += rng.multivariate_normal([0]*N_regions, np.diag(noise_amounts[0]), N_timepoints).T
                tss.append(ts)
        self._set_cache("valid", valid)
        return tss
    def _get_positions(self):
        return self.surrogate_of.get_positions()
    def get_valid(self):
        assert self._in_cache('valid'), "Please run get_timeseries() before this function"
        valids = np.zeros(self.N_subjects())
        valids[self._from_cache('valid')] = 1
        return valids.astype('bool')


import h5py
import scipy.io
import numpy as np
from config import HCP_FILENAME, HCP_DEMOGRAPHICS, HCPGSR_FILENAME, HCP1200_FILENAME, HCP1200GSR_FILENAME, DRAFT, memoize, HCP_MOVEMENT_FILENAME, CAMCAN_PATH, TRT_PATH, SHEN_PARC_PATH, TRT_MOVEMENT, HCP1200_DEMOGRAPHICS, AAL_PATH, HCP1200_DEMOGRAPHICS2, TRT_DEMOGRAPHICS, LSD_FILENAME, PSILOCYBIN_FILENAME, PSI_DEMO, LSD_DEMO, HCPREHO
from util import distance_matrix_euclidean, correlation_matrix_pearson, make_perfectly_symmetric
import paranoid as pns
from paranoidtypes import DistanceMatrix, CorrelationMatrix
import glob
import pandas
import nilearn
import nilearn.plotting
import re
import nibabel
import os

#################### HCP ####################

@pns.returns(pns.List(pns.NDArray(d=2, t=pns.Number)))
@memoize
def get_hcp_timeseries(scan=0):
    #get timeseries from an h5py file
    assert scan in [0, 1, 2, 3], "Invalid scan"
    f = h5py.File(HCP_FILENAME, 'r')
    subj_keys = []
    timeseries = []
    for k in f['timeseries']:
        subj_keys.append(str(k))
        data = list(f['timeseries/' + str(k)+'/scan_'+str(scan)])
        timeseries.append(np.array(data).T)
    assert len(subj_keys) == len(timeseries)
    return timeseries

@pns.returns(pns.NDArray(d=2, t=pns.Number))
def get_hcp_positions():
    """Return distances as a Pandas DataFrame"""
    f = h5py.File(HCP_FILENAME, 'r')
    return np.array(f['parcel_centroid'])

@pns.returns(DistanceMatrix)
def get_hcp_distances():
    """Return distances as a Pandas DataFrame"""
    f = h5py.File(HCP_FILENAME, 'r')
    return distance_matrix_euclidean(np.array(f['parcel_centroid']))

@pns.returns(pns.List(CorrelationMatrix))
def get_hcp_matrices(scan=0):
    return [correlation_matrix_pearson(ts) for ts in get_hcp_timeseries(scan=scan)]

@pns.returns(pns.Dict(k=pns.Set(["nvertices", "areas", "bordersize"]), v=pns.NDArray(d=1, t=pns.Positive0)))
def get_hcp_parcelstats():
    f = h5py.File(HCP1200_FILENAME, 'r')
    return {"nvertices": np.asarray(f['parcel_nvertices']),
            "areas": np.asarray(f['parcel_areas']),
            "bordersize": np.asarray(f['parcel_bordersize'])}

def get_hcp_movement(scan=0):
    f = h5py.File(HCP_FILENAME, 'r')
    df = pandas.read_csv(HCP_MOVEMENT_FILENAME, sep='\t').query('stat == "frame_dspl"')
    # Labels are incorrect for frame_displ, so "dx" here doesn't
    # actually mean dx.  I'm almost positive it means the mean
    # framewise displacement.
    return [df.query(f'session == {subj} and run == "bold {scan+1}"').iloc[0]['dx'] for subj in f['timeseries'].keys()]

@pns.returns(pns.List(pns.NDArray(d=2, t=pns.Number)))
def get_hcpgsr_timeseries(scan=0):
    #get timeseries from an h5py file
    assert scan in [0, 1, 2, 3], "Invalid scan"
    f = h5py.File(HCPGSR_FILENAME, 'r')
    subj_keys = []
    timeseries = []
    for k in f['timeseries']:
        subj_keys.append(str(k))
        data = list(f['timeseries/' + str(k)+'/scan_'+str(scan)])
        timeseries.append(np.array(data).T)
    assert len(subj_keys) == len(timeseries)
    return timeseries
    
@pns.returns(pns.List(CorrelationMatrix))
def get_hcpgsr_matrices(scan=0):
    return [correlation_matrix_pearson(ts) for ts in get_hcpgsr_timeseries(scan=scan)]

def get_hcp_demographics():
    dem = pandas.read_excel(HCP_DEMOGRAPHICS)
    f = h5py.File(HCPGSR_FILENAME, 'r')
    keys = list(map(int, f['timeseries'].keys())) # The same order as above
    dem.set_index("Subject", inplace=True)
    return dem.loc[keys]

def get_hcp1200_rehos(scan=0):
    rehoinfo = np.load(HCPREHO)
    rehos = rehoinfo['rehos'][np.asarray(list(map(int, rehoinfo['scan_nums'])))==int(scan+1)]
    subjs = list(rehoinfo['subjs'][np.asarray(list(map(int, rehoinfo['scan_nums'])))==int(scan+1)])
    f = h5py.File(HCP1200_FILENAME, 'r')
    keys = list(f['timeseries'].keys())
    return np.asarray([rehos[keys.index(subj)] for subj in keys])
    
        


#################### Cam-Can ####################

def _camcan_excluded_regions(parcellation):
    if parcellation == "AAL":
        exclude = [108]
    if parcellation == "Craddock":
        # These are nan in all subjects, except 713 which is nan in only one
        # subject.
        exclude = [ 40, 53, 63, 70, 73, 88, 90, 94, 100, 102, 109, 121, 151,
                    161, 167, 179, 209, 304, 343, 366, 408, 465, 538, 550, 635,
                    646, 713, 721, 740, 750, 780, 800, 828]
    if parcellation == "HOA116":
        exclude = [111, 112, 113, 114]
    if parcellation == "OSL":
        exclude = [] # OSL seems to be for MEG comparison, probably not relevant here...
    return exclude


def get_camcan_subjects():
    with h5py.File(CAMCAN_PATH.format(parcellation="AAL"), 'r') as f:
        subjects = list(sorted(f['timeseries'].keys()))
    # Exclude these subjects because they don't have all three fMRI scans.
    exclude = ['CC220519', 'CC410129', 'CC610050', 'CC610462', 'CC710214', 'CC710518']
    subjects = [s for s in subjects if s not in exclude]
    return subjects

def get_camcan_timeseries(dataset="Rest", parcellation="AAL"):
    subjects = get_camcan_subjects()
    with h5py.File(CAMCAN_PATH.format(parcellation=parcellation), 'r') as f:
        tss = []
        for subj in subjects:
            tss.append(f['timeseries'][subj][dataset])
        tss = np.asarray(tss)
    # Element 108 has a lot of nans, so we remove it.  This is Vermis_1_2 in AAL.
    exclude = _camcan_excluded_regions(parcellation)
    tss = np.delete(tss, exclude, axis=1)
    assert not np.any(np.isnan(tss))
    return tss

def get_camcan_positions(parcellation="AAL"):
    with h5py.File(CAMCAN_PATH.format(parcellation=parcellation), 'r') as f:
        poss = np.asarray(f['parcel_centroids'])
    exclude = _camcan_excluded_regions(parcellation)
    poss = np.delete(poss, exclude, axis=0)
    return poss

def get_camcan_distances(parcellation="AAL"):
    return distance_matrix_euclidean(get_camcan_positions(parcellation=parcellation))

def get_camcan_matrices(dataset="Rest", parcellation="AAL"):
    return [correlation_matrix_pearson(ts) for ts in get_camcan_timeseries(dataset, parcellation=parcellation)]

def get_camcan_behavior():
    behpath = os.path.split(CAMCAN_PATH)[0] + "/dataman/useraccess/processed/Maxwell_Shinn_494/"
    df_std = pandas.read_csv(behpath+"standard_data.csv").set_index("CCID")
    df_app = pandas.read_csv(behpath+"approved_data.tsv", sep="\t").set_index("CCID")
    ids = get_camcan_subjects()
    sorted_std = df_std.loc[ids]
    ages = np.asarray(sorted_std['Age'])
    sexes = np.asarray((sorted_std['Sex']=="FEMALE").astype(int))
    sorted_app = df_app.loc[ids]
    mq = np.asarray(sorted_app['additional_ten_mq_total'])
    memory = np.asarray(sorted_app['additional_memory'])
    fluencies = np.asarray(sorted_app['additional_fluencies'])
    language = np.asarray(sorted_app['additional_language'])
    visuospatial = np.asarray(sorted_app['additional_visuospatial'])
    acer = np.asarray(sorted_app['additional_acer'])
    attention = np.asarray(sorted_app['additional_attention_orientation'])
    cattell_path = os.path.split(CAMCAN_PATH)[0] + "/cc700-scored/Cattell/release001/data/"
    cattell_scores = []
    for subj in ids:
        try:
            cattell_scores.append(float(pandas.read_csv(cattell_path+f"Cattell_{subj}_scored.txt", sep="\t")['TotalScore'].iloc[0]))
        except FileNotFoundError:
            cattell_scores.append(np.nan)
    cattell_scores = np.asarray(cattell_scores)
    cardio_path = os.path.split(CAMCAN_PATH)[0] + "/cc700-scored/CardioMeasures/release001/exported/cardio_measures_exported.csv"
    df_cardio = pandas.read_csv(cardio_path).set_index("Observations")
    weight = np.asarray(df_cardio.loc[ids]['weight'])
    height = np.asarray(df_cardio.loc[ids]['height'])
    systolic = np.asarray(df_cardio.loc[ids]['bp_sys_mean'])
    diastolic = np.asarray(df_cardio.loc[ids]['bp_dia_mean'])
    pulse = np.asarray(df_cardio.loc[ids]['pulse_mean'])
    return {"age": ages, "sex": sexes, "10mq": mq, "memory": memory, "fluencies": fluencies,
            "language": language, "visuospatial": visuospatial, "ace-r": acer, "cattell": cattell_scores,
            "systolic": systolic, "diastolic": diastolic, "pulse": pulse, "weight": weight, "height": height}

def get_camcan_movement(dataset="Rest", parcellation="AAL"):
    subjects = get_camcan_subjects()
    with h5py.File(CAMCAN_PATH.format(parcellation=parcellation), 'r') as f:
        ms = []
        for subj in subjects:
            ms.append(f['timeseries'][subj][dataset+"_motion"])
        ms = np.asarray(ms)
    # Element 108 has a lot of nans, so we remove it.  This is Vermis_1_2 in AAL.
    exclude = _camcan_excluded_regions(parcellation)
    ms = np.delete(ms, exclude, axis=1)
    print(ms.shape)
    # Compute framewise displacement
    ms[:,:,3:6] = 50*np.sin(ms[:,:,3:6]) # Circle with radius 50mm
    msd = np.diff(ms, axis=1)
    fd = np.sum(np.abs(msd), axis=2)
    return np.mean(fd, axis=1)

def get_camcan_voxels(region):
    subjects = get_camcan_subjects()
    with h5py.File(CAMCAN_PATH.format(parcellation="midbrain_voxels_aal"), 'r') as f:
        N_subjects = len(subjects)
        regionname = f"region{region}"
        voxel_xyz = np.asarray(f[regionname]['voxel_xyz'])
        N_voxels = voxel_xyz.shape[1]
        N_timepoints = f[regionname]['timeseries'][subjects[0]].shape[0]
        voxels = np.zeros((N_subjects, N_voxels, N_timepoints))*np.nan
        for i,s in enumerate(subjects):
            voxels[i,:,:] = np.asarray(f[regionname]['timeseries'][s]).T
    return voxel_xyz, voxels

def get_camcan_parcelareas(parcellation="AAL"):
    fn = AAL_PATH+"/ROI_MNI_V4.nii"
    modimg = nibabel.load(fn)
    unique,counts = np.unique(modimg.dataobj, return_counts=True)
    assert np.all(sorted(unique) == unique)
    sizes = counts[1:]
    sizes = np.delete(counts[1:], _camcan_excluded_regions(parcellation))
    return sizes

#################### HCP1200 ####################

@pns.returns(pns.List(pns.NDArray(d=2, t=pns.Number)))
def get_hcp1200_timeseries(scan=0, gsr=False):
    #get timeseries from an h5py file
    assert scan in [0, 1, 2, 3], "Invalid scan"
    f = h5py.File(HCP1200_FILENAME if not gsr else HCP1200GSR_FILENAME, 'r')
    subj_keys = []
    timeseries = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = list(f['timeseries/' + str(k)+'/scan'+str(scan+1)])
        timeseries.append(np.array(data).T)
    assert len(subj_keys) == len(timeseries)
    return timeseries

@pns.returns(pns.NDArray(d=2, t=pns.Number))
def get_hcp1200_positions():
    """Return distances as a Pandas DataFrame"""
    f = h5py.File(HCP1200_FILENAME, 'r')
    return np.array(f['parcel_centroid'])

@pns.returns(DistanceMatrix)
def get_hcp1200_distances():
    """Return distances as a Pandas DataFrame"""
    f = h5py.File(HCP1200_FILENAME, 'r')
    return distance_matrix_euclidean(np.array(f['parcel_centroid']))

@pns.returns(pns.List(CorrelationMatrix))
def get_hcp1200_matrices(scan=0, gsr=False):
    return [correlation_matrix_pearson(ts) for ts in get_hcp1200_timeseries(scan=scan, gsr=gsr)]

@pns.returns(pns.Dict(k=pns.Set(["nvertices", "bordersize"]), v=pns.NDArray(d=1, t=pns.Positive0)))
def get_hcp1200_parcelstats():
    f = h5py.File(HCP1200_FILENAME, 'r')
    return {"nvertices": np.asarray(f['parcel_nvertices']),
            "areas": np.asarray(f['parcel_areas']),
            "bordersize": np.asarray(f['parcel_bordersize'])}

def get_hcp1200_movement(scan=0, gsr=False):
    if not gsr:
        f = h5py.File(HCP1200_FILENAME, 'r')
    else:
        f = h5py.File(HCP1200GSR_FILENAME, 'r')
    df = pandas.read_csv(HCP_MOVEMENT_FILENAME, sep='\t').query('stat == "frame_dspl"')
    # Labels are incorrect for frame_displ, so "dx" here doesn't
    # actually mean dx.  I'm almost positive it means the mean
    # framewise displacement.
    return [df.query(f'session == {subj} and run == "bold {scan+1}"').iloc[0]['dx'] for subj in sorted(f['timeseries'].keys())]

def get_hcp1200_demographics(gsr=False):
    dem = pandas.read_csv(HCP1200_DEMOGRAPHICS)
    dem2 = pandas.read_csv(HCP1200_DEMOGRAPHICS2)
    if not gsr:
        f = h5py.File(HCP1200_FILENAME, 'r')
    else:
        f = h5py.File(HCP1200GSR_FILENAME, 'r')
    keys = list(map(int, sorted(f['timeseries'].keys()))) # The same order as above
    dem.set_index("Subject", inplace=True)
    dem2.set_index("Subject", inplace=True)
    dem_merge = dem.join(dem2)
    return dem_merge.loc[keys]

def _get_trt_scaninfo():
    """Returns one tuple for each scan: (subject, session number, scanner id, scan number, subject_sex, subject_age)"""
    demographics = pandas.read_csv(TRT_DEMOGRAPHICS)
    demographics['id'] = demographics['Subject_ID'].map(lambda x : x[8:10])
    demographics['sexname'] = demographics['Gender (1=F)'].map(lambda x : "M" if x == 0 else "F")
    demographics.set_index("id", inplace=True)
    files = sorted(glob.glob(TRT_PATH+"/*.txt"))
    matches = [re.match(".*TRT0([0-9]+)_([0-9])_T([AB])_S00([0-9])_bis_matrix_roimean.txt", f) for f in files]
    info = [(int(m.group(1)),
             int(m.group(2)),
             m.group(3),
             int(m.group(4)),
             demographics["sexname"].loc[m.group(1)],
             int(demographics["Age"].loc[m.group(1)]),
    ) for m in matches]
    info = [i for i in info if i[3] <= 6]
    return info


def get_trt_timeseries():
    info = _get_trt_scaninfo()
    tss = []
    for inf in info:
        fn = f"/TRT{inf[0]:03}_{inf[1]}_T{inf[2]}_S00{inf[3]}_bis_matrix_roimean.txt"
        ts = np.loadtxt(TRT_PATH+fn, skiprows=1)[:,1:].T
        tss.append(ts)
    return tss

def get_trt_positions():
    shennii = nibabel.load(SHEN_PARC_PATH)
    shen_centroids = nilearn.plotting.find_parcellation_cut_coords(shennii)
    return shen_centroids

def get_trt_movement():
    info = _get_trt_scaninfo()
    df = pandas.read_csv(TRT_MOVEMENT, sep="\t", names=["name", "?", "ffd"]).set_index("name")
    movs = []
    for inf in info:
        print(inf)
        index_name = f"TRT{inf[0]:03}_{inf[1]}_T{inf[2]}_stack4d_S00{inf[3]}_frametoframe_tc.mat"
        movs.append(df.loc[index_name]['ffd'])
    return movs

def get_trt_parcelareas(parcellation="AAL"):
    modimg = nibabel.load(SHEN_PARC_PATH)
    unique,counts = np.unique(modimg.dataobj, return_counts=True)
    assert np.all(sorted(unique) == unique)
    sizes = counts[1:]
    return sizes

def get_trt_rehos():
    reho_data = scipy.io.loadmat("kendalls_trt.mat")['w']
    #(subject, session number, scanner id, scan number, subject_sex, subject_age)
    info = _get_trt_scaninfo()
    rehos = []
    for (subj, session, _, scan, _, _) in info:
        rehos.append(reho_data[subj-1,session-1,scan-1])
    return np.asarray(rehos)

@pns.accepts(exp=pns.Set(["Control", "LSD", "LSD+Ket"]),
             timepoint=pns.Set(["early", "late"]),
             gsr=pns.Boolean)
@pns.returns(pns.List(pns.NDArray(d=2, t=pns.Number)))
def get_lsd_timeseries(exp="Control", timepoint="early", gsr=True):
    #get timeseries from an h5py file
    f = h5py.File(LSD_FILENAME, 'r')
    subj_keys = []
    timeseries = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = list(f[f'timeseries/{k}/{exp}/{timepoint}{"" if gsr else "_nogsr"}'])
        timeseries.append(np.array(data).T[0:360])
    assert len(subj_keys) == len(timeseries)
    return timeseries


@pns.accepts(exp=pns.Set(["Control", "LSD", "LSD+Ket"]),
             timepoint=pns.Set(["early", "late"]))
@pns.returns(pns.List(pns.List(pns.Natural0)))
def get_lsd_dropped_frames(exp="Control", timepoint="early"):
    #get dropped frames from an h5py file
    f = h5py.File(LSD_FILENAME, 'r')
    subj_keys = []
    dropped_frames = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = list(f[f'timeseries/{k}/{exp}/{timepoint}_excluded_frames'])
        dropped_frames.append(data)
    assert len(subj_keys) == len(dropped_frames)
    return dropped_frames

@pns.accepts(exp=pns.Set(["Control", "LSD", "LSD+Ket"]),
             timepoint=pns.Set(["early", "late"]))
@pns.returns(pns.List(pns.Positive0))
def get_lsd_movement(exp="Control", timepoint="early"):
    #get mean fd from an h5py file
    f = h5py.File(LSD_FILENAME, 'r')
    subj_keys = []
    movement = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = f[f'timeseries/{k}/{exp}/{timepoint}_movement'][()]
        movement.append(data)
    assert len(subj_keys) == len(movement)
    return movement

@pns.returns(pandas.DataFrame)
def get_lsd_subjectinfo():
    #get mean fd from an h5py file
    f = h5py.File(LSD_FILENAME, 'r')
    dems = []
    dem = pandas.read_excel(LSD_DEMO)
    keys = list(map(int, sorted(f['timeseries'])))
    dem = dem.set_index('ID').loc[keys].reset_index()
    return dem


@pns.accepts(exp=pns.Set(["Control", "Psilocybin"]),
             timepoint=pns.Set(["early", "middle", "late"]),
             gsr=pns.Boolean())
@pns.returns(pns.List(pns.NDArray(d=2, t=pns.Number)))
def get_psilocybin_timeseries(exp="Control", timepoint="early", gsr=False):
    #get timeseries from an h5py file
    f = h5py.File(PSILOCYBIN_FILENAME, 'r')
    subj_keys = []
    timeseries = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = list(f[f'timeseries/{k}/{exp}/{"gsr" if gsr else "nogsr"}/{timepoint}'])
        timeseries.append(np.array(data).T[0:360])
    assert len(subj_keys) == len(timeseries)
    return timeseries


@pns.accepts(exp=pns.Set(["Control", "Psilocybin"]),
             timepoint=pns.Set(["early", "middle", "late"]))
@pns.returns(pns.List(pns.List(pns.Natural0)))
def get_psilocybin_dropped_frames(exp="Control", timepoint="early"):
    #get dropped frames from an h5py file
    f = h5py.File(PSILOCYBIN_FILENAME, 'r')
    subj_keys = []
    dropped_frames = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = list(f[f'timeseries/{k}/{exp}/{timepoint}_excluded_frames'])
        if k == '5220' and exp == "Control" and timepoint == "early":
            data += list(range(250, 265))
        dropped_frames.append(data)
    assert len(subj_keys) == len(dropped_frames)
    return dropped_frames

@pns.accepts(exp=pns.Set(["Control", "Psilocybin"]),
             timepoint=pns.Set(["early", "middle", "late"]))
@pns.returns(pns.List(pns.Positive0))
def get_psilocybin_movement(exp="Control", timepoint="early"):
    #get mean fd from an h5py file
    f = h5py.File(PSILOCYBIN_FILENAME, 'r')
    subj_keys = []
    movement = []
    for k in sorted(f['timeseries']):
        subj_keys.append(str(k))
        data = f[f'timeseries/{k}/{exp}/{timepoint}_movement'][()]
        movement.append(data)
    assert len(subj_keys) == len(movement)
    return movement

@pns.returns(pandas.DataFrame)
def get_psilocybin_subjectinfo():
    #get mean fd from an h5py file
    f = h5py.File(PSILOCYBIN_FILENAME, 'r')
    dems = []
    dem = pandas.read_excel(PSI_DEMO)
    keys = list(map(int, sorted(f['timeseries'])))
    dem = dem.set_index('ID').loc[keys].reset_index()
    return dem


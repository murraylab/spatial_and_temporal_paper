from glob import glob
import numpy as np
import scipy.io
import re
import h5py

parcellation = "AAL" # AAL, HOA116, Craddock, OSL
path_to_aamod = f"cc700/mri/pipeline/release004/data_fMRI_Unsmooth_{parcellation}/aamod_roi_extract_epi_0000*"
OUTPUT = f"camcan_{parcellation}.hdf5"

with h5py.File(OUTPUT, "w") as hf:
    hf.create_group('timeseries')
    for task in ["Rest", "Movie", "SMT"]:
        for f in glob(f'{path_to_aamod}/*/{task}/ROI_epi.mat'):
            subj = re.search('(CC[0-9]+)', f).group(1)
            print(task, subj)
            if task == "Rest":
                hf.create_group("timeseries/"+subj)
            try:
                mat = scipy.io.loadmat(f, squeeze_me=True, chars_as_strings=True)
            except:
                print("ERROR with subject", subj)
                continue
            ts = np.asarray([mat['ROI']['mean'][i] for i in range(0, len(mat['ROI']['mean']))]).astype(float)
            hf.create_dataset(f"timeseries/{subj}/{task}", data=ts)
            motionfile = glob(f"cc700/mri/pipeline/release004/data_fMRI/aamod_realignunwarp_00001/{subj}/{task}/rp_*.txt")[0]
            motionparams = np.loadtxt(motionfile)
            hf.create_dataset(f"timeseries/{subj}/{task}_motion", data=motionparams)
    xyz = np.asarray([mat['ROI']['XYZcentre'][i] for i in range(0, len(mat['ROI']['mean']))]).astype(float)
    hf.create_dataset(f"parcel_centroids", data=xyz)


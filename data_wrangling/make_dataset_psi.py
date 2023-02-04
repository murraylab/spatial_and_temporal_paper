import glob
import h5py
import nibabel
import pandas
import numpy as np

OUTPUT = "psilocybin_dataset.h5"
# PATH = "/project/fas/n3/Studies/MurrayLab/josh/Studies/LSD/subjects/"
PATH = "./"

# subjects_plus = [v.split("/")[-1] for v in glob.glob(PATH+"*_*")]
# subjects = list(sorted(set([v.split("_")[0] for v in subjects_plus])))
# Use _3 here because it automatically excludes the invalid subject with no session 3
subjects = list(sorted(set([v.split("_")[1] for v in glob.glob(PATH+"bold3_*_*.ptseries.nii")])))

# 1 = control, 2 = lsd, 3 = lsd+ketanserin

#fn_template = "{}_{}/images/functional/bold{}_Atlas_g7_hpss_res-mVWMWB1d_CAB-NP_ParcelOrder_r_GBC.pscalar.nii"
#movement_template = "{}_{}/images/functional/movement/bold{}_Atlas_g7_hpss_res-mVWMWB1d_CAB-NP_ParcelOrder_r_GBC.pscalar.nii"
#fn_template = "{}_{}/images/functional/{}_1_bold{}_Atlas_g7_hpss_res-mVWMWB1d_gbc_mFz_.ptseries.nii"
fn_template = "bold{}_{}_{}.ptseries.nii"
fn_template_gsr = "bold{}_{}_{}_gsr.ptseries.nii"
fn_template_scrub = "bold{}_{}_{}_scrub.txt" # TODO
fn_template_movement = "bold{}_{}_{}_movement.txt"

exp_dfn = {"Pla": "Control", "Psi": "Psilocybin"}
timepoint_dfn = {1: "early", 2: "middle", 3: "late"}

with h5py.File(OUTPUT, "w") as hf:
    hf.create_group('timeseries')
    for subject in reversed(subjects):
        hf.create_group(f'timeseries/{subject}')
        for exp in ["Pla", "Psi"]:
            hf.create_group(f'timeseries/{subject}/{exp_dfn[exp]}')
            hf.create_group(f'timeseries/{subject}/{exp_dfn[exp]}/gsr')
            hf.create_group(f'timeseries/{subject}/{exp_dfn[exp]}/nogsr')
            for timepoint in [1, 2, 3]:
                print(subject, exp, timepoint)
                scrub = pandas.read_csv(PATH+fn_template_scrub.format(timepoint, subject, exp), delim_whitespace=True)
                frames_to_use = np.asarray(scrub.iloc[0:-2].iloc[0:265]['use']).astype(bool)
                hf.create_dataset(f'timeseries/{subject}/{exp_dfn[exp]}/{timepoint_dfn[timepoint]}_excluded_frames', data=np.nonzero(~frames_to_use)[0])
                im = nibabel.load(PATH+fn_template.format(timepoint, subject, exp))
                # Only use 0:265 of the dataset since most subjects have this
                # many frames but subject 5208 has a few more for bold1 and
                # bold2 in placebo condition.
                hf.create_dataset(f'timeseries/{subject}/{exp_dfn[exp]}/nogsr/{timepoint_dfn[timepoint]}', data=im.get_data()[0:265,:][frames_to_use,:])
                if im.get_data().shape[0] != 265:
                       print("Shape is", im.get_data().shape)
                im = nibabel.load(PATH+fn_template_gsr.format(timepoint, subject, exp))
                hf.create_dataset(f'timeseries/{subject}/{exp_dfn[exp]}/gsr/{timepoint_dfn[timepoint]}', data=im.get_data()[0:265,:][frames_to_use,:])
                movement = pandas.read_csv(PATH+fn_template_movement.format(timepoint, subject, exp), delim_whitespace=True)
                hf.create_dataset(f'timeseries/{subject}/{exp_dfn[exp]}/{timepoint_dfn[timepoint]}_movement', data=np.mean(np.asarray(movement.iloc[:-3].iloc[0:265]['fd'])[frames_to_use]))

import nibabel
import glob
import numpy as np
import h5py

OUTPUT = 'timeseries_gsr.hdf5' # HDF5 file output
TSDIR = 'GSR' # Directory containing subject timeseries
GLASSERDIR = 'GlasserParcellation' # Directory containing glasser parcellation
SURFACEDIR = "surfaces"

# UPDATES:
# This file fixes the overlayed parcellation error we had before.
# 6/2020: Adds parcel sizes by counting # vertices, created OUTPUT and TSDIR constants
# 11/2020: This is a brand new version of this file for the HCP1200 version of the qnex processed data.
# 12/2020: Added feature to calculate surface area of each parcel

parcs = nibabel.cifti2.load(GLASSERDIR+'/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii')
surfR = nibabel.load(SURFACEDIR+'/Q1-Q6_RelatedValidation210.R.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii')
surfL = nibabel.load(SURFACEDIR+'/Q1-Q6_RelatedValidation210.L.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii')
# The following load requires you to run this command in the "glasser" directory:
#      wb_command -cifti-label-adjacency Glasser*/*.LR.*fs_LR.dlabel.nii -left-surface surfaces/*.L.*.surf.gii -right-surface surfaces/*.R.*.surf.gii parcellation-label-adjacency-matrix-generated.pconn.nii
adjacency = nibabel.load('parcellation-label-adjacency-matrix-generated.pconn.nii')
# The following load requires you to run these commands in the "surfaces" directory:
#     wb_command -surface-vertex-areas Q1-Q6_RelatedValidation210.L.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii areaL.func.gii
#     wb_command -surface-vertex-areas Q1-Q6_RelatedValidation210.R.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii areaR.func.gii
areaR = nibabel.load(SURFACEDIR+'/areaR.func.gii')
areaL = nibabel.load(SURFACEDIR+'/areaL.func.gii')


im = parcs.header.matrix.get_index_map(1)
adj_parcels = [p.name for p in adjacency.header.matrix.get_index_map(1).parcels]

parcs_data = np.asarray(parcs.get_data()).squeeze()

centroids = {}
parcelsizes = {}
parcelareas = {}

# Brain models are essentially brain regions (e.g. L or R cortex or
# else a subcortex structure) inside the parcellation.  Each one
# either has an xyz coordinate (which is only for subcortex so we
# don't care about it) or else a list of vertex indices, which will
# match up to vertices in a surface file.
for bm in im.brain_models:
    # Use the appropriate surface file
    if bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
        surf = surfR
        area = areaR
        hemi = 'R'
    elif bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
        surf = surfL
        area = areaL
        hemi = 'L'
    else:
        continue
    # Make sure the surface is appropriate for the brain model's list
    # of vertex indices
    assert surf.darrays[0].data.shape[0] == bm.surface_number_of_vertices, "Invalid surface vertices"
    # Get the list which maps vertices to parcels
    parcels = parcs_data[bm.index_offset:bm.index_count+bm.index_offset]
    # The parcels list and the bm.vertex_indices list correspond to
    # each other: together they can map vertices to parcels
    assert len(parcels) == len(bm.vertex_indices), "Parcels not vertex indices"
    # Operate on each parcel separately
    for parcel in set(parcels):
        # Find all indices of the verticies which match the parcel
        inds = np.asarray(bm.vertex_indices)[parcels == parcel]
        # surf.darrays[0].data is a list of positions of vertices.
        # Using the indices found in the previous step, we take the
        # indices which are a part of this parcel
        points = surf.darrays[0].data[inds]
        # Make sure we found the correct number of points
        assert points.shape == (len(inds), 3), "Invalid shape"
        # Each parcel should only occur once
        assert parcel not in centroids.keys(), "Parcel not in key"
        # The areaL and areaR images contain the total area for each vertex.
        # To find the area of the parcel, we look up the area of each vertex in
        # the parcel nd add them together.
        sumarea = np.sum(area.darrays[0].data[inds])
        # Save our results.  Save the parcel index, the name of the
        # parcel, and the mean of the indices to generate the
        # centroid.
        # Extract the name of the parcel
        name = list(parcs.header.matrix[0].named_maps)[0].label_table[parcel].label
        # Save all parcel information
        centroids[parcel] = (name, hemi, tuple(np.mean(points, axis=0)))
        parcelsizes[parcel] = (name, hemi, len(points))
        parcelareas[parcel] = (name, hemi, sumarea)


centroids_items = sorted([(parcel, *c) for parcel,c in centroids.items()], key=lambda x : x[0])
parcelsizes_items = sorted([(parcel, *c) for parcel,c in parcelsizes.items()], key=lambda x : x[0])
parcelareas_items = sorted([(parcel, *c) for parcel,c in parcelareas.items()], key=lambda x : x[0])
assert np.all([ci[1] for ci in centroids_items] == adj_parcels), "Adjacency matrix parcellation not in the correct order"


# First save all data matrices into the all_scans dictionary indexed
# by subject ID.
all_scans = {}

# Loop through and load nifti timeseries files
from tqdm import tqdm
for f in tqdm(glob.glob(TSDIR+'/*.ptseries.nii')):
    try:
        ts = nibabel.load(f, keep_file_open=False, mmap=False)
    except ImportError:
        continue
    # Check to make sure that the parcellation in the timeseries is in the
    # same order as the parcellation we processed previously.
    assert [p.name for p in ts.header.get_index_map(1).parcels] == [c[1] for c in centroids_items]
    # Save the data
    subj_id = f.split("/")[-1].split("_")[0]
    scan_id = f.split("/")[-1].split("_")[1].split(".")[0]
    if subj_id not in all_scans.keys():
        all_scans[subj_id] = {}
    all_scans[subj_id][scan_id] = np.asarray(ts.get_data())

# Include only subjects who had all four scans
all_scans = {k:v for k,v in all_scans.items() if len(v) == 4}

assert len(all_scans) != 0, "Invalid directory, no timeseries found"
# Generate the list of times, in seconds.  Doing this for the last
# subject only (i.e. the "ts" remaining at the end of the loop) should
# be fine since they should all be the same.
times = ts.header.get_index_map(0).series_start + np.arange(0, len(ts.get_data()[:,0]))*ts.header.get_index_map(0).series_step

# Save as an HDF5 file
string_dtype = h5py.special_dtype(vlen=str)
with h5py.File(OUTPUT, "w") as hf:
    hf.create_dataset('times', data=times)
    hf.create_dataset('parcel_name', data=np.asarray([c[1] for c in centroids_items], dtype=object), dtype=string_dtype)
    hf.create_dataset('parcel_centroid', data=np.asarray([c[3] for c in centroids_items]))
    hf.create_dataset('parcel_hemisphere', data=np.asarray([1 if c[2]=="R" else 2 for c in centroids_items]))
    hf.create_dataset('parcel_nvertices', data=np.asarray([c[3] for c in parcelsizes_items]))
    hf.create_dataset('parcel_areas', data=np.asarray([c[3] for c in parcelareas_items]))
    hf.create_dataset('parcel_bordersize', data=np.sum(adjacency.get_data(), axis=0))
    hf.create_group('timeseries')
    for subjid,scans in all_scans.items():
        hf.create_group("timeseries/"+str(subjid))
        for scanid,data in scans.items():
            hf.create_dataset("timeseries/"+str(subjid)+"/"+str(scanid), data=data)

DRAFT = True

HCP_FILENAME = 'timeseries.hdf5'
HCP_MOVEMENT_FILENAME = '/home/max/Research_data/murray/reprocessed_hcp/bold_movement_report.txt'
HCP_DEMOGRAPHICS = '/home/max/Research_data/murray/reprocessed_hcp/hcp_demographics.xlsx'
HCPGSR_FILENAME = '/home/max/Research_data/murray/reprocessed_hcp/timeseries_gsr.hdf5'
HCP1200_FILENAME = '/home/max/Research_data/murray/reprocessed_hcp/Atlas_MSMAll_hp2000_clean_demean-100f_Glasser_S1200_RelatedValidation210/hcp_max/timeseries.hdf5'
HCPREHO = 'computed_rehos.npz'
HCP1200GSR_FILENAME = '/home/max/Research_data/murray/reprocessed_hcp/Atlas_MSMAll_hp2000_clean_demean-100f_Glasser_S1200_RelatedValidation210/hcp_max/timeseries_gsr.hdf5'
HCP1200_DEMOGRAPHICS = '/home/max/Research_data/murray/reprocessed_hcp/RESTRICTED_amatkovic_11_26_2020_18_19_54.csv'
HCP1200_DEMOGRAPHICS2 = '/home/max/Research_data/murray/reprocessed_hcp/unrestricted_amatkovic_7_6_2020_10_57_38.csv'
HCP_GEODESIC_R_FILENAME = 'RightParcelGeodesicDistmat.txt' # Taken from the brainsmash documentation
HCP_GEODESIC_L_FILENAME = 'LeftParcelGeodesicDistmat.txt'
CAMCAN_PATH = "/home/max/Research_data/murray/camcan/camcan_{parcellation}.hdf5"
TRT_PATH = "/home/max/Research_data/murray/trt/timeseries"
TRT_DEMOGRAPHICS = "/home/max/Research_data/murray/trt/demographics_test_retest.csv"
SHEN_PARC_PATH = "/home/max/Research_data/murray/trt/shen_1mm_268_parcellation.nii.gz"
TRT_MOVEMENT = "/home/max/Research_data/murray/trt/mean_FFD_in_mm.txt"
AAL_PATH = "/home/max/Scripts/spatialgraphs/aal8"
SHEN_PATH = "/home/max/Research_data/murray/trt/shen_1mm_268_parcellation.nii.gz"
LSD_FILENAME = "/home/max/Research_data/murray/lsd/lsd_dataset.h5"
LSD_DEMO = "/home/max/Research_data/murray/lsd/LSD_Age.xlsx"
PSILOCYBIN_FILENAME = "/home/max/Research_data/murray/psilocybin/psilocybin_dataset.h5"
PSI_DEMO = "/home/max/Research_data/murray/psilocybin/Psi_Age.xlsx"
CACHE_DIR = "/home/max/Tmp/cdatasets_cache" # No trailing slash


USE_CACHE = False

#import klepto
#memoize = klepto.lfu_cache(maxsize=100, cache=klepto.archives.file_archive('cache.klepto'))

if USE_CACHE:
    import joblib
    memoize = joblib.Memory("./cachedir", verbose=0).cache
else:
    memoize = lambda x : x

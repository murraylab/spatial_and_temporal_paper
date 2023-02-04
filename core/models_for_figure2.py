import cdatasets
import util


fit_params = util.pload("fit_parameters.pkl")
fit_params_geo = util.pload("fit_parameters_geo.pkl")

#################### Dataset: TRT ####################

trtinfo = cdatasets.TRT().get_subject_info()
datafull = cdatasets.TRT()
data = [cdatasets.Subset(f"subset{i}", datafull, trtinfo['run']==(i+1)) for i in range(0, 6)]
dataretest = data[2:6]+data[0:2]


DATASET = "trt"

colorless_params = [v for _,v in sorted(fit_params[(DATASET, "Colorless", 'eigsL2')].items())]
colorfull_params = [v for _,v in sorted(fit_params[(DATASET, "Colorfull", 'eigsL2')].items())]
spaceonly_params = [v for _,v in sorted(fit_params[(DATASET, "Spaceonly", 'eigsL2')].items())]
_generative = cdatasets.ModelColorless(fit_of=datafull, params=colorless_params, seed=25)
generative = [cdatasets.Subset(f"subset{i}", _generative, trtinfo['run']==(i+1)) for i in range(0, 6)]
_colorfull = cdatasets.ModelColorfull(fit_of=datafull, params=colorfull_params, seed=30)
colorfull = [cdatasets.Subset(f"subset{i}", _colorfull, trtinfo['run']==(i+1)) for i in range(0, 6)]
_spaceonly = cdatasets.ModelSpaceonly(fit_of=datafull, params=spaceonly_params, seed=35)
spaceonly = [cdatasets.Subset(f"subset{i}", _spaceonly, trtinfo['run']==(i+1)) for i in range(0, 6)]
_color_surrogate = cdatasets.ColorSurrogate(surrogate_of=datafull, seed=40)
color_surrogate = [cdatasets.Subset(f"subset{i}", _color_surrogate, trtinfo['run']==(i+1)) for i in range(0, 6)]
_cftimeonly = cdatasets.ModelColorfullTimeonly(fit_of=datafull, seed=20)
cftimeonly = [cdatasets.Subset(f"subset{i}", _cftimeonly, trtinfo['run']==(i+1)) for i in range(0, 6)]
_timeonly = cdatasets.ModelColorlessTimeonly(fit_of=datafull, seed=15)
timeonly = [cdatasets.Subset(f"subset{i}", _timeonly, trtinfo['run']==(i+1)) for i in range(0, 6)]
_phase = cdatasets.PhaseRandomize(surrogate_of=datafull, seed=1)
phase = [cdatasets.Subset(f"subset{i}", _phase, trtinfo['run']==(i+1)) for i in range(0, 6)]
_eigen = cdatasets.Eigensurrogate(surrogate_of=datafull, seed=10)
eigen = [cdatasets.Subset(f"subset{i}", _eigen, trtinfo['run']==(i+1)) for i in range(0, 6)]
_zalesky = cdatasets.Zalesky2012(surrogate_of=datafull, seed=10)
zalesky = [cdatasets.Subset(f"subset{i}", _zalesky, trtinfo['run']==(i+1)) for i in range(0, 6)]
_degreerand = cdatasets.DegreeRandomize(surrogate_of=datafull, seed=5)
degreerand = [cdatasets.Subset(f"subset{i}", _degreerand, trtinfo['run']==(i+1)) for i in range(0, 6)]

# all_models = [_generative, _color_surrogate, _spaceonly, _timeonly, _phase, _zalesky, _degreerand]
# for m in all_models:
#     m.cache_all()

cmodels = [("retest", dataretest),
           ("ColorlessHet", generative),
           ("Colorsurrogate", color_surrogate),
           ("Spaceonly", spaceonly),
           #("ColorfullTimeonly", cftimeonly),
           ("ColorlessTimeonly", timeonly),
           #("Colorfull", colorfull),
           ("phase", phase),
           ("eigen", eigen),
           ("zalesky2012", zalesky),
           ("degreerand", degreerand)]

cmodels_trt = cmodels
data_trt = data



#################### Dataset: Cam-CAN ####################

datafull = cdatasets.CamCanFiltered()
data = [datafull]
DATASET = "camcanfilteredRestAAL"

colorless_params = [v for _,v in sorted(fit_params[(DATASET, "Colorless", 'eigsL2')].items())]
colorfull_params = [v for _,v in sorted(fit_params[(DATASET, "Colorfull", 'eigsL2')].items())]
spaceonly_params = [v for _,v in sorted(fit_params[(DATASET, "Spaceonly", 'eigsL2')].items())]
generative = [cdatasets.ModelColorless(fit_of=datafull, params=colorless_params, seed=25)]
colorfull = [cdatasets.ModelColorfull(fit_of=datafull, params=colorfull_params, seed=30)]
spaceonly = [cdatasets.ModelSpaceonly(fit_of=datafull, params=spaceonly_params, seed=35)]
color_surrogate = [cdatasets.ColorSurrogate(surrogate_of=datafull, seed=40)]
cftimeonly = [cdatasets.ModelColorfullTimeonly(fit_of=datafull, seed=20)]
timeonly = [cdatasets.ModelColorlessTimeonly(fit_of=datafull, seed=15)]
phase = [cdatasets.PhaseRandomize(surrogate_of=datafull, seed=1)]
eigen = [cdatasets.Eigensurrogate(surrogate_of=datafull, seed=10)]
zalesky = [cdatasets.Zalesky2012(surrogate_of=datafull, seed=10)]
degreerand = [cdatasets.DegreeRandomize(surrogate_of=datafull, seed=5)]

# all_models = [generative[0], colorfull[0], spaceonly[0], timeonly[0], phase[0], zalesky[0], degreerand[0]]
# for m in all_models:
#     m.cache_all()

cmodels = [("ColorlessHet", generative),
           ("Colorsurrogate", colorfull),
           ("Spaceonly", spaceonly),
           #("ColorfullTimeonly", cftimeonly),
           ("ColorlessTimeonly", timeonly),
           #("Colorfull", colorfull),
           ("phase", phase),
           ("eigen", eigen),
           ("zalesky2012", zalesky),
           ("degreerand", degreerand)]


cmodels_camcan = cmodels
data_camcan = data




#################### Dataset: HCP ####################

DATASET = 'hcp1200'
data = [cdatasets.HCP1200(i) for i in range(0, 4)]
dataretest = data[1:4]+[data[0]]

colorless_params = [[v for _,v in sorted(fit_params[(DATASET+str(i), "Colorless", 'eigsL2')].items())] for i in range(0, 4)]
colorfull_params = [[v for _,v in sorted(fit_params[(DATASET+str(i), "Colorfull", 'eigsL2')].items())] for i in range(0, 4)]
spaceonly_params = [[v for _,v in sorted(fit_params[(DATASET+str(i), "Spaceonly", 'eigsL2')].items())] for i in range(0, 4)]
generative = [cdatasets.ModelColorless(fit_of=data[i], params=colorless_params[i], seed=25+i) for i in range(0, 4)]
colorfull = [cdatasets.ModelColorfull(fit_of=data[i], params=colorfull_params[i], seed=30+i) for i in range(0, 4)]
spaceonly = [cdatasets.ModelSpaceonly(fit_of=data[i], params=spaceonly_params[i], seed=35+i) for i in range(0, 4)]
color_surrogate = [cdatasets.ColorSurrogate(surrogate_of=data[i], seed=40+i) for i in range(0, 4)]
cftimeonly = [cdatasets.ModelColorfullTimeonly(fit_of=data[i], seed=i+20) for i in range(0, 4)]
timeonly = [cdatasets.ModelColorlessTimeonly(fit_of=data[i], seed=i+15) for i in range(0, 4)]
phase = [cdatasets.PhaseRandomize(surrogate_of=data[i], seed=i) for i in range(0, 4)]
#color_surrogate_correlated = [cdatasets.ColorSurrogateCorrelated(surrogate_of=data[i], seed=45+i) for i in range(0, 4)]
eigen = [cdatasets.Eigensurrogate(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
zalesky = [cdatasets.Zalesky2012(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
degreerand = [cdatasets.DegreeRandomize(surrogate_of=data[i], seed=i+5) for i in range(0, 4)]

cmodels = [("retest", dataretest),
           ("ColorlessHet", generative),
           ("Colorsurrogate", color_surrogate),
           ("Spaceonly", spaceonly),
           #("ColorfullTimeonly", cftimeonly),
           ("ColorlessTimeonly", timeonly),
           #("Colorfull", colorfull),
           ("phase", phase),
           ("eigen", eigen),
           ("zalesky2012", zalesky),
           ("degreerand", degreerand)]

cmodels_hcp = cmodels
data_hcp = data





#################### Dataset: HCP-GSR ####################

data = [cdatasets.HCP1200(i, gsr=True) for i in range(0, 4)]
dataretest = data[1:4]+[data[0]]

colorless_params = [[v for _,v in sorted(fit_params[("hcp1200"+str(i)+"gsr", "Colorless", 'eigsL2')].items())] for i in range(0, 4)]
colorfull_params = [[v for _,v in sorted(fit_params[("hcp1200"+str(i)+"gsr", "Colorfull", 'eigsL2')].items())] for i in range(0, 4)]
spaceonly_params = [[v for _,v in sorted(fit_params[("hcp1200"+str(i)+"gsr", "Spaceonly", 'eigsL2')].items())] for i in range(0, 4)]
generative = [cdatasets.ModelColorless(fit_of=data[i], params=colorless_params[i], seed=25+i) for i in range(0, 4)]
colorfull = [cdatasets.ModelColorfull(fit_of=data[i], params=colorfull_params[i], seed=30+i) for i in range(0, 4)]
spaceonly = [cdatasets.ModelSpaceonly(fit_of=data[i], params=spaceonly_params[i], seed=35+i) for i in range(0, 4)]
color_surrogate = [cdatasets.ColorSurrogate(surrogate_of=data[i], seed=40+i) for i in range(0, 4)]
cftimeonly = [cdatasets.ModelColorfullTimeonly(fit_of=data[i], seed=i+20) for i in range(0, 4)]
timeonly = [cdatasets.ModelColorlessTimeonly(fit_of=data[i], seed=i+15) for i in range(0, 4)]
phase = [cdatasets.PhaseRandomize(surrogate_of=data[i], seed=i) for i in range(0, 4)]
#color_surrogate_correlated = [cdatasets.ColorSurrogateCorrelated(surrogate_of=data[i], seed=45+i) for i in range(0, 4)]
eigen = [cdatasets.Eigensurrogate(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
zalesky = [cdatasets.Zalesky2012(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
degreerand = [cdatasets.DegreeRandomize(surrogate_of=data[i], seed=i+5) for i in range(0, 4)]

cmodels = [("retest", dataretest),
           ("ColorlessHet", generative),
           ("Colorsurrogate", color_surrogate),
           ("Spaceonly", spaceonly),
           #("ColorfullTimeonly", cftimeonly),
           ("ColorlessTimeonly", timeonly),
           #("Colorfull", colorfull),
           ("phase", phase),
           ("eigen", eigen),
           ("zalesky2012", zalesky),
           ("degreerand", degreerand)]
cmodels_hcpgsr = cmodels
data_hcpgsr = data


#################### Dataset: HCP-Geo ####################

data = [cdatasets.HCP1200Geo(i) for i in range(0, 4)]
dataretest = data[1:4]+[data[0]]

colorless_params = [[v for _,v in sorted(fit_params_geo[("hcp1200"+str(i)+"geoR", "Colorless", 'eigsL2')].items())] for i in range(0, 4)]
colorfull_params = [[v for _,v in sorted(fit_params_geo[("hcp1200"+str(i)+"geoR", "Colorfull", 'eigsL2')].items())] for i in range(0, 4)]
spaceonly_params = [[v for _,v in sorted(fit_params_geo[("hcp1200"+str(i)+"geoR", "Spaceonly", 'eigsL2')].items())] for i in range(0, 4)]
generative = [cdatasets.ModelColorless(fit_of=data[i], params=colorless_params[i], seed=25+i) for i in range(0, 4)]
colorfull = [cdatasets.ModelColorfull(fit_of=data[i], params=colorfull_params[i], seed=30+i) for i in range(0, 4)]
spaceonly = [cdatasets.ModelSpaceonly(fit_of=data[i], params=spaceonly_params[i], seed=35+i) for i in range(0, 4)]
color_surrogate = [cdatasets.ColorSurrogate(surrogate_of=data[i], seed=40+i) for i in range(0, 4)]
timeonly = [cdatasets.ModelColorlessTimeonly(fit_of=data[i], seed=i+15) for i in range(0, 4)]
phase = [cdatasets.PhaseRandomize(surrogate_of=data[i], seed=i) for i in range(0, 4)]
#color_surrogate_correlated = [cdatasets.ColorSurrogateCorrelated(surrogate_of=data[i], seed=45+i) for i in range(0, 4)]
eigen = [cdatasets.Eigensurrogate(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
zalesky = [cdatasets.Zalesky2012(surrogate_of=data[i], seed=i+10) for i in range(0, 4)]
degreerand = [cdatasets.DegreeRandomize(surrogate_of=data[i], seed=i+5) for i in range(0, 4)]

cmodels = [("retest", dataretest),
           ("ColorlessHet", generative),
           ("Colorsurrogate", color_surrogate),
           ("Spaceonly", spaceonly),
           #("ColorfullTimeonly", cftimeonly),
           ("ColorlessTimeonly", timeonly),
           #("Colorfull", colorfull),
           ("phase", phase),
           ("eigen", eigen),
           ("zalesky2012", zalesky),
           ("degreerand", degreerand)]
cmodels_hcpgeo = cmodels
data_hcpgeo = data

# This can be useful when changing models: [m.cache_all() for mods in cmodels for m in mods[1]]

#################### Compiling everything together ####################

all_cmodels = {"hcp": (data_hcp, cmodels_hcp),
               "hcpgeo": (data_hcpgeo, cmodels_hcpgeo),
               "hcpgsr": (data_hcpgsr, cmodels_hcpgsr),
               "trt": (data_trt, cmodels_trt),
               "camcan": (data_camcan, cmodels_camcan),
               }

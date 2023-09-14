"""
This script plots metrics extracted from neural data to analyse whether visual
function is modulated by oestrus cycle in female mice.
"""
#NOTE: decoding accuracy chance = 0.033, both v1 & hpf
#NOTE: grating decoding accuracy chance = 1/6

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from pixtools import utils

from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM

import statsmodels.formula.api as smf
import pingouin as pg

csv_dir = '/Users/student/Desktop/estrus_analysis/nobaselinesubtraction.csv' # csv file for no baseline subtraction
#csv_dir = '/Users/student/Desktop/estrus_analysis/baselinesubtracted.csv' 

fig_dir = '/Users/student/Desktop/oestrus_plots' # saved figures for estrous project

# load data
data = pd.read_csv(csv_dir, sep=',')
print('> Loaded organised csv file.\n')

data = data.set_index("Session")
#drop first 3 days
#exclude = ["20220310_az_ESCN00", "20220322_az_ESCN00", "20220323_az_ESCN00", "20220406_az_ESCN02", "20220407_az_ESCN02", "20220408_az_ESCN02", "20220407_az_ESCN03", "20220408_az_ESCN03", "20220409_az_ESCN03"]

'''
#data baseline subtracted
data = data.rename(columns={'nTotalV1':'nV1', 'nHPCUnits':'nHPF', 'nOrientationResponsive':'nOR',
'nMovieResponsive':'nMR', 'hpf_nMovieResponsive': 'hpf_nMR', 'PercentMovieResp':'pMR',
'hpf_propMovieResp': 'hpf_pMR', 'PercentOrientationResp':'pOR', 'V1_PV_OSI':'fs_OSI', 'V1_PV_OSI_superficial':'fs_OSIs', 'V1_PV_OSI_deep':'fs_OSId', 'TuningWidth':'TW', 'V1_PV_Tuning':'fs_TW', 'V1_PV_Tuning_superficial': 'fs_TWs', 'V1_PV_Tuning_deep':'fs_TWd', 'V1_Decoding_Accuracy':'DA', 'hpf_Decoding_Accuracy':'hpf_DA', 'TrialCorr':'TC', 'hpf_TrialCorr':'hpf_TC', 'DarkFiringRate':'DFR', 'MovieFiringRate':'MFR','GratingFiringRate':'GFR','DarkFiringRate_PV':'fs_DFR','MovieFiringRate_PV':'fs_MFR','GratingFiringRate_PV':'fs_GFR','hpf_DarkFiringRate':'hpf_DFR','hpf_MovieFiringRate':'hpf_MFR','hpf_GratingFiringRate':'hpf_GFR','Peak_GammaFreq':'PGF','hpf_Peak_GammaFreq':'hpf_PGF','Normalized_GammaPower':'NGP', 'hpf_Normalized_GammaPower':'hpf_NGP','hpf_PeakThetaFreq':'hpf_PTF','hpf_Norm_ThetaPower':'hpf_NTP'})
'''

#data = data.drop(exclude)
#data no baseline subtraction
data = data.rename(columns={'nTotalV1':'nV1', 'nHPCUnits':'nHPF', 'nOrientationResponsive':'nOR',
'nMovieResponsive':'nMR', 'hpf_nMovieResponsive': 'hpf_nMR', 'PercentMovieResp':'pMR',
'hpf_propMovieResp': 'hpf_pMR', 'PercentOrientationResp':'pOR', 'V1_PV_OSI':'fs_OSI', 'V1_PV_OSI_superficial':'fs_OSIs', 'V1_PV_OSI_deep':'fs_OSId', 'TuningWidth':'TW', 'V1_PV_Tuning':'fs_TW', 'V1_PV_Tuning_superficial': 'fs_TWs', 'V1_PV_Tuning_deep':'fs_TWd', 'decoding_for_gratings':'DAg', 'V1_Decoding_Accuracy':'DA', 'hpf_Decoding_Accuracy':'hpf_DA', 'TrialCorr':'TC', 'Trial-trial_correlation_for_gratings':'TCg', 'hpf_TrialCorr':'hpf_TC', 'DarkFiringRate':'DFR', 'MovieFiringRate':'MFR','GratingFiringRate':'GFR','DarkFiringRate_PV':'fs_DFR','MovieFiringRate_PV':'fs_MFR','GratingFiringRate_PV':'fs_GFR','hpf_DarkFiringRate':'hpf_DFR','hpf_MovieFiringRate':'hpf_MFR','hpf_GratingFiringRate':'hpf_GFR','Peak_GammaFreq':'PGF','hpf_Peak_GammaFreq':'hpf_PGF','Normalized_GammaPower':'NGP', 'hpf_Normalized_GammaPower':'hpf_NGP','hpf_PeakThetaFreq':'hpf_PTF','hpf_Norm_ThetaPower':'hpf_NTP'})

'''
# v1 & hpc count & proportion
both_count = ['nV1', 'nHPF', 'nMR', 'hpf_nMR', 'nOR']
both_prop = ['pMR', 'hpf_pMR', 'pOR']

# v1 fs,  v1, & hpc selectivity
fs_v1_hpf_OSI = ['OSI', 'fs_OSI', 'fs_OSIs', 'fs_OSId']
fs_v1_hpf_TW = ['TW', 'fs_TW', 'fs_TWs', 'fs_TWd']

# v1 grating, v1, & hpc DA & TC
g_v1_hpf_da = ['DA', 'hpf_DA']
g_v1_hpf_tc = ['TC', 'hpf_TC']

# v1 fs, v1, hpc firing rates
fs_v1_hpf_fr = ['DFR', 'MFR', 'GFR', 'hpf_DFR', 'hpf_MFR', 'hpf_GFR', 'fs_DFR', 'fs_MFR', 'fs_GFR']

# v1 & hpc lfp analysis
both_lfp = ['PGF', 'hpf_PGF', 'NGP', 'hpf_NGP', 'hpf_PTF', 'hpf_NTP']
'''
#simplified, without Tom's data
#sdata = data.iloc[:-2, :]

# v1 & hpc count & proportion
both_count = ['nV1', 'nHPF', 'nMR', 'hpf_nMR', 'nOR']
both_prop = ['pMR', 'hpf_pMR', 'pOR']

# v1 fs,  v1, & hpc selectivity
fs_v1_hpf_OSI = ['OSI', 'fs_OSI', 'fs_OSIs', 'fs_OSId']
fs_v1_hpf_TW = ['TW', 'fs_TW', 'fs_TWs', 'fs_TWd']

# v1 grating, v1, & hpc DA & TC
g_v1_hpf_da = ['DAg', 'DA', 'hpf_DA']
g_v1_hpf_tc = ['TCg','TC', 'hpf_TC']

# v1 fs, v1, hpc firing rates
fs_v1_hpf_fr = ['DFR', 'MFR', 'GFR', 'hpf_DFR', 'hpf_MFR', 'hpf_GFR', 'fs_DFR', 'fs_MFR', 'fs_GFR']

# v1 & hpc lfp analysis
both_lfp = ['PGF', 'hpf_PGF', 'NGP', 'hpf_NGP', 'hpf_PTF', 'hpf_NTP']

MOVIE_CHANCE_LEVEL = 0.033 #decoding accuracy chance for both v1 & hpf
#grating decoding accuracy change = 1/6
GRATING_CHANCE_LEVEL = 1/6
########################################################################

from oestrusplots import plot_metrics
'''
all_metrics = [both_count, both_prop, fs_v1_hpf_OSI, fs_v1_hpf_TW, g_v1_hpf_da, g_v1_hpf_tc, fs_v1_hpf_fr, both_lfp]
for m in all_metrics:
    plot_metrics(
            data=data, 
            sdata=data, 
            metrics=m, 
            xd='session', 
            xs='Stage', 
            fig_dir=fig_dir, 
            r=3, 
            c=3, 
            fsize=(25, 15),
            movie=MOVIE_CHANCE_LEVEL,
            grating=GRATING_CHANCE_LEVEL,
        )

'''

datas = data.rename(columns={'DAg':'Grating Responsive V1 Neurons', 'DA':'Responsive V1 Neurons', 'hpf_DA':'Responsive HPF Neurons'})
datac = datas[['Grating Responsive V1 Neurons', 'Responsive V1 Neurons', 'Responsive HPF Neurons']]
from scipy.stats import f_oneway
dec = f_oneway(datac[['Grating Responsive V1 Neurons']], datac[['Responsive V1 Neurons']], datac[['Responsive HPF Neurons']])
print(dec)

other = stats.ttest_ind(datac[['Grating Responsive V1 Neurons']], datac[['Responsive V1 Neurons']])
hpfv1 = stats.ttest_ind(datac[['Responsive HPF Neurons']], datac[['Responsive V1 Neurons']])
hpfgr = stats.ttest_ind(datac[['Responsive HPF Neurons']], datac[['Grating Responsive V1 Neurons']])
print(other, hpfv1, hpfgr)

assert 0

datac = datac.iloc[:-2,:]#datac = datac.pivot(columns = 'Neuron', values = 'DA')
datac = datac.reset_index()
del datac[datac.columns[0]]
#datac["metric"] = 'DA'
datac = datac.mean()
datac.columns = ['Neuron','DA']
print(datac)

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('DA ~ Neuron', data=datac).fit()
aov_table = sm.stats.anova_lm(model, typ=1)
print(aov_table)

#Decoding Accuracy Barplot
sns.set_style('darkgrid')
fig_decode = sns.barplot(
    data=datas[['Grating Responsive V1 Neurons', 'Responsive V1 Neurons', 'Responsive HPF Neurons', 'mouse_id']], 
    palette = "blend:#7AB,#EDA",
    )
plt.hlines(
    MOVIE_CHANCE_LEVEL,
    xmin=-1,
    xmax=2,
    linestyles='--',
    color='k',
    )

fig_decode.set(title = 'Average Decoding Accuracy', ylabel = 'Decoding Accuracy')

utils.save(
    path=fig_dir + 'decodea.pdf',
    fig=fig_decode.get_figure(),
    nosize=True,
    )


#all plots
#all_metrics = [both_count, both_prop, fs_v1_hpf_OSI, fs_v1_hpf_TW, g_v1_hpf_da, g_v1_hpf_tc, both_lfp]
#for m in all_metrics:

    #plot_metrics(
            #data=data, 
            #sdata=sdata, 
            #metrics=m, 
            #xd='session', 
            #xs='Stage', 
            #fig_dir=fig_dir, 
            #r=3, 
            #c=3, 
            #fsize=(25, 15),
            #movie=MOVIE_CHANCE_LEVEL,
            #grating=GRATING_CHANCE_LEVEL,
    #)
 ################################################################



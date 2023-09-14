
import pandas as pd
import numpy as np
from numpy import array

import seaborn as sns
import matplotlib.pyplot as plt
from pixtools import utils

from statannotations.Annotator import Annotator
from scipy import stats
import scipy.stats as stat
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as multi
from statistics import stdev

import statsmodels.formula.api as smf
import pingouin as pg


csv_dir = '/Users/student/Desktop/estrus_analysis/nobaselinesubtraction.csv' # csv file

fig_dir = '/Users/student/Desktop/estrusplots' # saved figures for estrous project

# load data
data = pd.read_csv(csv_dir, sep=',')
print('> Loaded organised csv file.\n')
print(data.head())

#drop first three days of each mouse

data = data.set_index("Session")
exclude = ["20220310_az_ESCN00", "20220322_az_ESCN00", "20220323_az_ESCN00",
"20220406_az_ESCN02", "20220407_az_ESCN02", "20220408_az_ESCN02", "20220407_az_ESCN03",
"20220408_az_ESCN03", "20220409_az_ESCN03"]
#data = data.drop(exclude)

data = data.rename(columns={'Trial-trial_correlation_for_gratings': 'TC_for_gratings'})

MOVIE_CHANCE_LEVEL = 0.033
GRATING_CHANCE_LEVEL = 1/6

all_metrics = data.columns.values[3:-1]

for_ttest = ['decoding_for_gratings', 'V1_Decoding_Accuracy', 'hpf_Decoding_Accuracy']

# cycles statistics

for m, metric in enumerate(all_metrics):

    df = data.iloc[:-2, :][[metric, 'mouse_id', 'session', 'Stage']]
    df1 = data.iloc[:-2, :].pivot(index = ['Stage', 'session'], columns = 'mouse_id', values = metric)
   
# assumptions check

    # repeated measures ANOVA, mixed effects models, friedman, compare variation across days
    # normality, Shapiro-Wilk (p>0.05?)
    cat = df1.columns.values
    normality_check = []
    values = []
    for i, mouse in enumerate(df1.columns.values):
        series = df1[df1.columns[i]].dropna()
        values.append(series.values)
        W,W_p = stats.shapiro(series)
        normality_check.append((W, W_p))
    
    normality = pd.DataFrame(normality_check, columns=['W', 'p'], index=cat)
    print(f'{metric} normality check:\n', normality)

    if (normality.p > 0.05).all():
        if df[metric].isnull().values.any():
            df = df.dropna(axis=0)
            md = smf.mixedlm(f'{metric} ~ session', df, groups = df['mouse_id'])
            mdf = md.fit()
            print(f"mixedlm test {mdf.summary()}")
        else:
            print(f"anovaRM {AnovaRM(data=df, depvar = metric, subject = 'mouse_id', within=['session']).fit()}")
    else:
        dfs = df1.T
        nan_id = np.where(dfs.loc[mouse, :].isnull())[0]
        columns = np.delete(dfs.columns.values, nan_id)
        print(f"from pingouin {pg.friedman(data=df, dv = metric, within = 'session', subject = 'mouse_id')}")

print(f"calculating difference from chance level for decoding accuracy")
for metric in for_ttest:
    if metric[0] == 'd':
        t_statistic, p_value = stat.ttest_1samp(a=data[metric], popmean=1/6)
        print(f"one-sample t-test for (grating) {metric}: {t_statistic} p = {p_value}")
    else:
        t_statistic, p_value = stat.ttest_1samp(a=data[metric], popmean=0.033)
        print(f"one-sample t-test for (movie) {metric}: {t_statistic} p = {p_value}")


ess = []
for_power = ['PercentMovieResp', 'hpf_propMovieResp', 'MovieFiringRate_PV', 'MovieFiringRate', 'hpf_MovieFiringRate', 'OSI', 'V1_PV_OSI', 'Normalized_GammaPower', 'hpf_Normalized_GammaPower', 'decoding_for_gratings', 'V1_Decoding_Accuracy', 'hpf_Decoding_Accuracy']
power_metrics = []


print(f":::by stage statistics:::")
#by stage statistics
for m, metric in enumerate(all_metrics):

    df = data.iloc[:-2, :][[metric, 'mouse_id', 'session', 'Stage']]
    dfp = data.pivot(columns = 'Stage', values = metric)

# assumptions check
    # normality, Shapiro-Wilk (p>0.05?)
    cat = dfp.columns.values
    normality_check = []
    values = []
    for i in range(len(dfp.columns)):
        series = dfp[dfp.columns[i]].dropna()
        if series.shape[0] > 2:
            values.append(series.values)
            W,W_p = stats.shapiro(series)
            normality_check.append((W, W_p))
        else:
            print(f'{series.name} has less than 3 datapoints.')
            cat = cat.tolist()
            cat.remove(series.name)

    normality = pd.DataFrame(normality_check, columns=['W', 'p'], index=cat)
    print(f'{metric} normality check:\n', normality)

    #array = np.array(values)

    #H, H_p = stats.kruskal(*array)
    #print('Kruskal-Wallis test statistic =', H, 'p =', H_p)

    #if (H_p < 0.05).all():
        #print('\nKruskal-Wallis is significant, doing post-hoc tests:')
        
        # post hoc Conover
        #post_hoc = sp.posthoc_conover(array, val_col=metric, group_col=cat, p_adjust = 'holm')
        #print('\nPost hoc tests results: \n', post_hoc)

    dfi = df.groupby(['mouse_id', 'Stage']).mean()
    dfis = dfi.reset_index(level = ['mouse_id'])
    dfis = dfis.reset_index(level = ['Stage'])

    #to see means for each stage for each animal
    print(dfi)

    #overall mean for each animal
    smean = df.groupby(['Stage'])[metric].mean()
    print(f"overall mean for each stage{smean}")
    import statistics

    stddata = df.groupby(['Stage'])[metric].describe()
    print(f"stats descriptive mean for each stage {stddata}")

    bmean = df.groupby(['mouse_id'])[metric].mean()
    print(f"overall mean for each mouse {bmean}")
    
    '''
    #ols with different variables to see change in R-square
    both = ols(f'{metric} ~ C(Stage) + C(mouse_id)', data = dfis).fit()
    #stage_only = ols(f'{metric} ~ C(Stage)', data = dfis).fit()
    #mouse_only = ols(f'{metric} ~ C(mouse_id)', data = dfis).fit()
    #interaction = ols(f'{metric} ~ C(Stage) + C(mouse_id) +\
            #C(Stage) * C(mouse_id)', data = dfis).fit()
    result = sm.stats.anova_lm(both, typ = 2)
    #print(f"\nols summary with mouse only:\n",
         #mouse_only.summary())
    #print(f"\nols summary with stage only:\n",
         #stage_only.summary())
    print(f"\nols summary with both:\n",
         both.summary())
    #print(f"\nols summary with interaction:\n",
         #interaction.summary())
    print(f"\ntwo-way anova on stage & mouse id:\n",
         result)

    '''
    #interaction analysis
    tw = ols(f'{metric} ~ Stage + mouse_id', data = dfis).fit()
    result = sm.stats.anova_lm(tw, typ = 2)
    print(f'anova_lm', result)
    

    #calculate effect sizes for specific metrics
    if metric in for_power:
        power_metrics.append(metric)
        ssbetween = result.iloc[0,0] 
        ssresidual = result.iloc[-1,0]
        sstotal = ssbetween + ssresidual
        es = ssbetween/sstotal
        print(f"effect size for {metric} is {es}")
        ess.append(es)

    '''
    if (result['PR(>F)'] < 0.05).any():
        print('\none-way ANOVA is significant, doing post-hoc tests:')
        sigs = result.loc[result['PR(>F)'] < 0.05].index.values

        # post hoc tukey HSD
        for sig in sigs:
            post_hoc = multi.MultiComparison(dfis[metric], dfis[sig]).tukeyhsd()
            print('\nPost hoc tests results: \n', post_hoc.summary())
    '''
effect_sizes = pd.DataFrame([power_metrics, ess]).T
print(effect_sizes)

#from statsmodels.stats.power import ftest_anova_power
#effect = 0.8
#alpha = 0.05
#power = 0.8
#k = 3
#nobs = 30

#panalysis = ftest_anova_power(effect_size = effect, nobs = nobs, alpha = alpha, k_groups = k, df=None)
#print(f"expected power for current sample size {panalysis}")




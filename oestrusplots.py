import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pixtools import utils

def plot_metrics(data, sdata, metrics, xd, xs, fig_dir, r, c, fsize, movie, grating):

    # lineplot, by day
    fig_d, axes_d = plt.subplots(r, c, figsize = fsize, sharey = True)
    faxes_d = axes_d.flatten()
    # lineplot, by stage
    fig_s, axes_s = plt.subplots(r, c, figsize = fsize, sharey = True)
    faxes_s = axes_s.flatten()
    # boxplot
    fig_b, axes_b = plt.subplots(r, c, figsize = fsize, sharey = True)
    faxes_b = axes_b.flatten()
    titles = faxes_d.copy()
    

    for m, metric in enumerate(metrics):

        if 'hpf' in metric.casefold():
            region = 'HPF'
        else:
            region = 'V1'

        if 'fs' in metric:
            unit_type = 'Fast-Spiking'
    
        if 'dfr' in metric.casefold():
            condition = 'in Darkness'
        elif 'mfr' in metric.casefold():
            condition = 'Movie'
        elif 'gfr' in metric.casefold():
            condition = 'Grating'

        if metrics[0][0] == 'n':
            ylabel = 'Neuron Count'
            if metric[-2] == 'O':
                titles[m] = f'Number of {region} Grating Responsive Neurons'
            elif metrics == 'nMR':
                titles[m] = f'Number of {region} Movie Responsive Neurons'
            else:
                titles[m] = f'Total Number of Neurons Recorded in {region}'

        elif metrics[0][0] == 'p':
            ylabel = 'Proportion (% of Total Neurons)'
            if metric[-2] == 'O':
                titles[m] = f'Proportion of {region} Grating Responsive Neurons'
            else:
                titles[m] = f'Proportion of {region} Movie Responsive Neurons'

        elif metrics[0][0] == 'O':
            ylabel = 'OSI'
            if 'fs' in metric.casefold():
                titles[m] = f'Orientation Selectivity Index of {region} Fast-Spiking Neurons'
                if metric[-1] == 's':
                    titles[m] = f'Orientation Selectivity Index of Superficial {region} Fast-Spiking Neurons'
                else:
                    titles[m] = f'Orientation Selectivity Index of Deep {region} Fast-Spiking Neurons'
            else:            
                titles[m] = f'Orientation Selectivity Index of V1 Neurons'

        elif metrics[0][:2] == 'TW':
            ylabel = 'Degrees (*)'
            if 'fs' in metric.casefold():
                titles[m] = f'Tuning Width of V1 Fast-Spiking Neurons'
                if metric[-1] == 's':
                    titles[m] = f'Tuning Width of Superficial V1 Fast-Spiking Neurons'
                else:
                    titles[m] = f'Tuning Width of Deep V1 Fast-Spiking Neurons'
            else:            
                titles[m] = f'Tuning Width of V1 Neurons'

        elif metrics[0][:2] == 'DA':
            ylabel = 'Decoding Accuracy'
            if metric[-1] != 'g':
                titles[m] = f'Decoding Accuracy of {region} Movie Responsive Neurons'
                chance = movie
            else:
                titles[m] = f'Decoding Accuracy of {region} Grating Responsive Neurons'
                chance = grating
            faxes_d[m].hlines(
                chance,
                xmin=1,
                xmax=10,
                linestyles='--',
                color='k',
            )
            faxes_b[m].hlines(
                chance,
                xmin=-7,
                xmax=10,
                linestyles='--',
                color='k'
            )
                
        elif metrics[0][:2] == 'TC':
            ylabel = 'Trial-to-trial Correlation'
            if metric[-1] == 'g':
                titles[m] = f'Trial-to-trial Correlation of {region} Grating Responsive Neurons'
            else:
                titles[m] = f'Trial-to-trial Correlation of {region} Movie Responsive Neurons'

        elif metrics[0][:2] == 'DF':
            ylabel = 'Firing Rate (Hz)'
            if metrics[0][1] == 'f':
                if 'dfr' in metric.casefold():
                    titles[m] = f'Firing Rate of {region} Fast-Spiking Neurons in Darkness'
                else:
                    titles[m] = f'Firing Rate of {region} Fast-Spiking {condition} Responsive Neurons'
            else:
                if 'dfr' in metric.casefold():
                    titles[m] = f'Firing Rate of {region} Neurons in Darkness'
                else:
                    titles[m] = f'Firing Rate of {region} {condition} Responsive Neurons'
                 
        elif metrics[0][0] == 'P':
            ylabel = 'Normalised Power'
            if 'pgf' in metric.casefold():
                titles[m] = f'{region} Peak Gamma Frequency'
            elif 'ngp' in metric.casefold():
                titles[m] = f'{region} Normalised Gamma Power'
            elif 'ptf' in metric.casefold():
                titles[m] = f'{region} Peak Theta Frequency'
            elif 'ntp' in metric.casefold():
                titles[m] = f'{region} Normalised Theta Power'
       
        fig_d = sns.set_style('darkgrid')
        fig_d = sns.lineplot(
            data = sdata,
            x = xd,
            y = metric,
            hue = 'mouse_id',
            ax = faxes_d[m],
            palette = 'Set2',
            marker = 'o',
        )
        faxes_d[m].set_title(titles[m])
        faxes_d[m].set_ylabel(ylabel)
         
        fig_s = sns.set_style('darkgrid')
        fig_s = sns.lineplot(
            data = sdata,
            x = xs,
            y = metric,
            hue = 'mouse_id',
            ax = faxes_s[m],
            #errorbar = ('sd', 2),
            err_style = 'band',
            palette = 'Set2',
        )

        faxes_s[m].set_title(titles[m])
        faxes_s[m].set_ylabel(ylabel)

        fig_b = sns.boxplot(
            data = data,
            x = xs,
            y = metric,
            width = 0.8,
            color = 'lightsteelblue',
            ax = faxes_b[m],
            order = ['p', 'e', 'm', 'd']
       )

        sns.stripplot(
            data = data, 
            x = xs, 
            y = metric, 
            dodge = True,
            linewidth = 1,
            color = 'darkkhaki',
            edgecolor= 'black',
            alpha = 0.8,
            ax = faxes_b[m],
            order = ['p', 'e', 'm', 'd']

        )
        faxes_b[m].set_title(titles[m])
        faxes_b[m].set_ylabel(ylabel)
    
   
    if m > 0:
        faxes_d[m].get_legend().remove()
        faxes_s[m].get_legend().remove()
    
    fig_d = fig_d.get_figure()
    fig_s = fig_s.get_figure()
    fig_b = fig_b.get_figure()
    
    utils.save(
        path=fig_dir + f'/{ylabel}_day.pdf',
        fig=fig_d.get_figure(),
        nosize=True,
    )
    utils.save(
        path=fig_dir + f'/{ylabel}_s.pdf',
        fig=fig_s.get_figure(),
        nosize=True,
    )
    utils.save(
        path=fig_dir + f'/{ylabel}_box.pdf',
        fig=fig_b.get_figure(),
        nosize=True,
    )

    return None

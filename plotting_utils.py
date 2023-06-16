import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kaplanmeier as km

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering

import plotly.graph_objects as go
import plotly.figure_factory as ff
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from IPython.display import IFrame

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage


def cdist(m):
    """
    Distance function with some rounding errors that runs in O(n^2).
    Necessary for creating dendrograms with 10000+ elements
    """
    xy = np.dot(m, m.T)  # O(k^2)
    x2 = y2 = (m * m).sum(1)  # O(k^2)
    d2 = np.add.outer(x2, y2) - 2 * xy  # O(k^2)
    d2.flat[::len(m) + 1] = 0  # Rounding issues
    d2[d2 < 0] = 0  # More rounding issues with identical samples
    return squareform(np.sqrt(d2))  # O (k^2)


def color_to_rgblist(color, n):
    """
    Gets the first n items of a categorical colormap,
    or splits a continuous colormap into n uniformly spaced colors
    """
    if n < 2:
        n = 2
    raw_cmap = cm.get_cmap(color)
    if type(raw_cmap) == LinearSegmentedColormap or len(raw_cmap.colors) > 20:
        raw_cmap = raw_cmap.resampled(n)
        clist = [raw_cmap(i)[:3] for i in range(n)]
        raw_cmap = ListedColormap(clist)
    cmap_colors = raw_cmap.colors
    colorscale = cmap_colors[:n]
    new_cmap = ListedColormap(colorscale)
    rgblist = [f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})' for c in colorscale]
    return rgblist


def categorical_heatmap_annotation(spectrum, dendro_leaves_x, legendgroup, color, xaxis, yaxis, rank=1000):
    """
    For placement above a main plot, spectrum must be 1D and the title will be on the left.
    Returns a figure with a heatmap on xaxis and yaxis, as well as a legend
        created using invisible scatter plots on xaxis1000 and yaxis100
    Motivated by heatmaps that only support continuous variables.
    """
    spectrum_filtered = spectrum[~pd.isnull(spectrum)]
    num_categories = int(len(np.unique(spectrum_filtered)))
    if type(color) == list:
        rgblist = color
    else:
        rgblist = color_to_rgblist(color, num_categories)
    if len(rgblist) < 2:  # catch bad colors
        rgblist = ['rgb(0,0,0)', 'rgb(0,0,0)']
    legend = go.Figure()
    for rgb, specval in zip(rgblist, pd.unique(spectrum_filtered)):
        legend.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                xaxis='x1000',
                yaxis='y1000',
                legendgroup=legendgroup,
                legendgrouptitle_text=legendgroup,
                legendrank=rank,
                name=str(specval),
                mode='markers',
                marker=dict(
                    size=100,
                    color=rgb,
                    symbol='square',
                    line=dict(
                        width=0,
                    )
                )
            )
        )
    spectrum_vals = np.zeros(len(spectrum))
    for i, specval in enumerate(pd.unique(spectrum)):
        spectrum_vals[spectrum == specval] = i
    spectrum_vals[pd.isnull(spectrum)] = np.nan
    heatmap = go.Figure(
        go.Heatmap(
            x=dendro_leaves_x,
            y=[legendgroup] * len(dendro_leaves_x),
            z=spectrum_vals[dendro_leaves_x],
            xaxis=xaxis,
            yaxis=yaxis,
            colorscale=rgblist,
            showscale=False,
        )
    )
    return heatmap, legend


def plot_dendrogram(networks_flat, title='', method='ward', spectrums=[], spectrum_labels=[], spectrum_types=[],
                    colors=[], show_legends=[], savepath=None, dendro_height=100):
    heatmap_height = 300  # pixels
    spectrum_height = 20  # pixels
    spectrums_height = len(spectrums) * spectrum_height
    # dendro_height = 100  # pixels
    total_height = heatmap_height + spectrums_height + dendro_height
    heatmap_frac = heatmap_height / total_height
    spectrums_frac = spectrums_height / total_height
    spectrum_frac = spectrums_frac / len(spectrums)
    heatmap_x_frac = 0.7
    cbar_legend_x = 0.72
    cbar_width = 80 / total_height
    legend_x = cbar_legend_x + cbar_width + 0.02
    divider = 0.003

    # row normalize
    row_mins = np.min(networks_flat, axis=0)
    row_maxs = np.max(networks_flat, axis=0)
    row_scale = row_maxs - row_mins
    row_scale[row_scale == 0] = 1
    scaled_data = ((networks_flat - row_mins) / row_scale * 2 - 1)
    data_array = scaled_data
    labels = np.arange(len(data_array)).astype(str)

    def linkagefun(x):
        return linkage(x, method)

    # Initialize figure by creating upper dendrogram (patient similarities)
    fig = ff.create_dendrogram(networks_flat, orientation='bottom', color_threshold=np.Inf, distfun=cdist,
                               linkagefun=linkagefun)  # , labels=labels)
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'
        fig['data'][i]['showlegend'] = False  # needs to be manually turned off when showing other legends

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(networks_flat.T, orientation='right', color_threshold=np.Inf, distfun=cdist,
                                       linkagefun=linkagefun)
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'
        dendro_side['data'][i]['showlegend'] = False  # also needs to be manually turned off

    # Add Side Dendrogram Data to Figure
    #     for data in dendro_side['data']:
    #         fig.add_trace(data)

    # Get dendrogram indices
    dendro_leaves_x = fig['layout']['xaxis']['ticktext']
    dendro_leaves_x = list(map(int, dendro_leaves_x))
    dendro_leaves_y = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves_y = list(map(int, dendro_leaves_y))

    # Add covariate spectrum bars
    num_colorbars = 0  # do a count because we build legends backwards
    for spectrum_type, show_legend in zip(spectrum_types, show_legends):
        if spectrum_type.lower() == 'continuous' and show_legend:
            num_colorbars += 1
    colorbar_i = 0
    for fig_i, (spectrum, spectrum_label, spectrum_type, color, show_legend) in enumerate(
            zip(spectrums[::-1], spectrum_labels[::-1], spectrum_types[::-1], colors[::-1], show_legends[::-1])):
        spectrum_filtered = spectrum[~pd.isnull(spectrum)]
        #         if '3q' in spectrum_label or 'TP53' in spectrum_label:
        #             plt.figure(figsize=(4, 0.5))
        #             plt.pcolormesh([spectrum, spectrum[dendro_leaves_x]])
        #             plt.show()
        if spectrum_type.lower() == 'continuous':
            colorscale = color
            legend_kwargs = {'showscale': False}
            if show_legend:
                legend_kwargs.update({
                    'showscale': True,
                    'colorbar_orientation': 'h',
                    'colorbar_xanchor': 'left',
                    'colorbar_x': cbar_legend_x,
                    'colorbar_yanchor': 'top',
                    'colorbar_y': 1.095 - (num_colorbars * (80 / total_height)) + (colorbar_i) * (80 / total_height),
                    'colorbar_len': 80 / total_height,
                    'colorbar_thickness': 15,
                    'colorbar_tickvals': [np.min(spectrum_filtered), np.round(np.mean(spectrum_filtered)),
                                         np.max(spectrum_filtered)],
                    'colorbar_title': dict(text=spectrum_label, side='top'),
                })
                colorbar_i += 1
            oncoplot_fig = go.Figure(data=go.Heatmap(
                x=dendro_leaves_x,
                y=[spectrum_label] * len(dendro_leaves_x),
                z=spectrum[dendro_leaves_x],
                yaxis=f'y{fig_i + 3}',
                xaxis='x',
                colorscale=colorscale,
                **legend_kwargs,
            ))
            # oncoplot_legend = go.Figure(data=go.Heatmap(
            #     z=np.linspace(np.min(spectrum_filtered), np.max(spectrum_filtered), 100)[np.newaxis, :],
            #     colorscale=colorscale,
            #     yaxis=f'y{fig_i + 3}',
            #     xaxis=f'x{fig_i + 3}',
            #     showscale=False,
            # ))
        else:
            oncoplot_fig, oncoplot_legend = categorical_heatmap_annotation(
                spectrum,
                dendro_leaves_x,
                spectrum_label,
                color,
                'x',
                f'y{fig_i + 3}',
                rank=len(spectrums) - fig_i
            )
        for i in range(len(oncoplot_fig['data'])):
            oncoplot_fig['data'][i]['x'] = fig['layout']['xaxis']['tickvals']
        for data in oncoplot_fig['data']:
            fig.add_trace(data)
        if show_legend and spectrum_type != 'continuous':
            # Not showing continuous legends, since plotly groups colorbars and other legends differently
            for data in oncoplot_legend['data']:
                fig.add_trace(data)

    # Generate Heatmap
    heat_data = data_array.T
    heat_data = heat_data[dendro_leaves_y, :]
    heat_data = heat_data[:, dendro_leaves_x]
    heatmap = [
        go.Heatmap(
            x=dendro_leaves_x,
            y=dendro_leaves_y,
            z=heat_data,
            colorscale='RdBu',
            zmid=0,
            #             legendgroup = 'Networks',
            colorbar_orientation='v',
            colorbar_xanchor='left',
            colorbar_x=0.7,
            colorbar_yanchor='bottom',
            colorbar_y=0.0,
            colorbar_len=heatmap_frac,
            # colorbar_thickness=5,
            #             name = 'Networks',
            #             showlegend = True,
        ),
    ]
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout({
        'width': 1200,
        'height': total_height,
        'showlegend': True,
        'hovermode': 'closest',
    })
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.1, heatmap_x_frac],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': False,
                             'ticks': ""})
    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .1],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, heatmap_frac],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': False,
                             'ticks': ""
                             })

    # Edit yaxis2
    fig.update_layout(yaxis2={'domain': [heatmap_frac + spectrums_frac + divider * 2, 1.],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

    fig.update_layout(title=title)
    legend_kwargs = {
        'yaxis1000': {
            'domain': [0, 0.001],
            'showgrid': False,
            'showticklabels': False,
            'visible': True,
        },
        'xaxis1000': {
            'domain': [0, 0.001],
            'showgrid': False,
            'showticklabels': False,
            'visible': True,
        },
        'legend': {
            'yanchor': 'top',
            'y': 1.0,
            'xanchor': 'left',
            'x': legend_x,
            'orientation': 'v',
            #         'itemwidth': 30,
            #         'thickness': 30,
        },
    }
    fig.update_layout(**legend_kwargs)

    for i, spectrum_type in enumerate(spectrum_types[::-1]):
        kwargs = {
            f'yaxis{i + 3}': {
                'domain': [heatmap_frac + divider + spectrum_frac * i,
                           heatmap_frac + divider + spectrum_frac * (i + 1)],
                'showgrid': True,
                'showticklabels': True,
            },
        }
        if spectrum_type == 'continuous':
            kwargs.update({
                f'xaxis{i + 3}': {
                    'domain': [0.9, 1.],
                    'showgrid': False,
                    'showticklabels': False,
                },
            })
        fig.update_layout(**kwargs)

    # Plot!
    if savepath is None:
        savepath = 'tempplot.pdf'
    fig.write_image(savepath, scale=2.)
    return IFrame(savepath, width=1000, height=total_height)


if __name__ == '__main__':
    n_spectrums = 20
    n_features = 10
    n_patients = 100
    networks_dummy = np.random.normal(0, 1, (n_patients, n_features))
    spectrums = [np.random.randint(0, 5, n_patients) for i in range(n_spectrums)]
    spectrum_labels = [f'spectrum {i}' for i in range(n_spectrums)]
    spectrum_types = np.random.choice(['categorical', 'continuous'], n_spectrums, replace=True)
    colors = np.random.choice(['Blues', 'viridis', 'plasma'], n_spectrums, replace=True)
    show_legends = [True] * 5 + [False] * (n_spectrums - 5)

    # spectrums = [np.random.uniform(-1, 1, 100), np.random.normal(0, 1, 100), np.random.randint(0, 10, 100)]
    # spectrum_labels = ['unif', 'normal', 'randint']
    # spectrum_types = ['continuous', 'continuous', 'categorical']
    # colors = ['Blues', 'Viridis', 'tab20']
    # show_legends = [True, False, True]
    plot_dendrogram(networks_dummy, title='hello dendrogram', method='ward', spectrums=spectrums,
                    spectrum_labels=spectrum_labels, spectrum_types=spectrum_types, colors=colors,
                    show_legends=show_legends)
    plot_dendrogram(networks_dummy, title='hello dendrogram', method='ward', spectrums=spectrums,
                    spectrum_labels=spectrum_labels, spectrum_types=spectrum_types, colors=colors,
                    show_legends=show_legends, dendro_height=1000)  # easy way to make legends visible

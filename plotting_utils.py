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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


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
    if '-log' in legendgroup:
        legendgroup_title = legendgroup.split(' (-log')[0]
    else: 
        legendgroup_title = legendgroup
    spectrum_filtered = spectrum[~pd.isnull(spectrum)]
    num_categories = int(len(np.unique(spectrum_filtered)))
    if type(color) == list:
        rgblist = color
    else:
        rgblist = color_to_rgblist(color, num_categories)
    if len(rgblist) < 2:  # catch bad colors
        rgblist = ['rgb(0,0,0)', 'rgb(0,0,0)']
    legend = go.Figure()
    # spectrum_filtered = spectrum_filtered.astype(str)
    for rgb, specval in zip(rgblist, np.unique(spectrum_filtered)):
        legend.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                xaxis='x1000',
                yaxis='y1000',
                legendgroup=legendgroup,
                legendgrouptitle_text=legendgroup_title,
                legendrank=rank,
                name=str(specval),
                mode='markers',
                # visible='legendonly',  # makes the legends semitransparent, maybe fix if this is an issue?
                showlegend=True,
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
    for i, specval in enumerate(np.unique(spectrum_filtered)):  # pd retains ordering, np sorts
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
                    colors=[], show_legends=[], savepath=None, dendro_height=100, n_clusters=None):
    heatmap_height = 300  # pixels
    spectrum_height = 20  # pixels
    spectrums_height = len(spectrums) * spectrum_height
    # dendro_height = 100  # pixels
    total_height = heatmap_height + spectrums_height + dendro_height
    total_width = 1400
    heatmap_frac = heatmap_height / total_height
    spectrums_frac = spectrums_height / total_height
    spectrum_frac = spectrums_frac / len(spectrums)
    heatmap_x_frac = 0.6
    cbar_legend_x = 0.72
    cbar_width = 100 / total_width
    cbar_height = 80 / total_height
    legend_x = cbar_legend_x + cbar_width + 0.05
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
                colorbar_title = spectrum_label
                if "-log" in spectrum_label:
                    colorbar_title = colorbar_title.split(" (-log")[0] 
                # if "percent" in colorbar_title:
                #     max_val = 100
                #     min_val = 0
                # else:
                max_val = np.max(spectrum_filtered)
                min_val = np.min(spectrum_filtered)
                tickvals = [min_val, (max_val - min_val) / 2 + min_val, max_val]
                # Compress to 2 sig figs. If the values are >1, display as int.
                if 'Purity' in spectrum_label:
                    ticktext = [str(float(f"{min_val:.2f}")), str(float(f"{(max_val - min_val) / 2 + min_val:.2f}")), str(float(f"{max_val:.2f}"))]
                elif max_val < 10:
                    ticktext = [str(float(f"{min_val:.1E}")), str(float(f"{(max_val - min_val) / 2 + min_val:.1E}")), str(float(f"{max_val:.1E}"))]
                else:
                    ticktext = [str(int(float(f"{min_val:.1E}"))), str(int(float(f"{(max_val - min_val) / 2 + min_val:.1E}"))), str(int(float(f"{max_val:.1E}")))]
                # ticktext = [str(int(float(f"{min_val:.1E}"))), str(int(float(f"{(max_val - min_val) / 2 + min_val:.1E}"))), str(int(float(f"{max_val:.1E}")))] 
                legend_kwargs.update({
                    'showscale': True,
                    'colorbar_orientation': 'h',
                    'colorbar_xanchor': 'left',
                    'colorbar_x': cbar_legend_x,
                    'colorbar_yanchor': 'top',
                    'colorbar_y': 1.095 - (num_colorbars * cbar_height) + (colorbar_i) * cbar_height,
                    'colorbar_len': cbar_width,
                    'colorbar_thickness': 15,
                    'colorbar_tickvals': tickvals,
                    'colorbar_ticktext': ticktext,
                    'colorbar_tickfont_size': 10,
                    'colorbar_title': dict(text=colorbar_title, side='top'),
                    'colorbar_title_font_size': 13,
                    'zmin': min_val,
                    'zmax': max_val,
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
            colorbar_x=heatmap_x_frac,
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
        'width': total_width,
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
    
    
    # Infer clusters if requested and add vertical line separators
    if n_clusters is not None or n_clusters > 1:
        method = 'ward'
        criterion = 'maxclust'
        dist_array = cdist(networks_flat)
        Z = linkage(dist_array, method=method)
        cluster_labels = fcluster(Z, n_clusters, criterion=criterion)
        dendrogram_idx = dendrogram(Z, no_plot=True)['leaves']

        # Get relative positions across heatmap for vertical lines
        cluster_positions = []
        curr = None
        for i, cluster_label in enumerate(cluster_labels[dendrogram_idx]):
            if curr is None or cluster_label == curr:
                curr = cluster_label
                continue
            else:
                cluster_positions.append(i / len(cluster_labels))
                curr = cluster_label

        # Draw the vertical lines
        for i, cluster_position in enumerate(cluster_positions): 
            axis_num = f'{500 + i:03d}'
            line = go.Scatter(
                        x=[1, 1],
                        y=[-1, 2],
                        xaxis='x' + axis_num,
                        yaxis='y' + axis_num,
                        legendgroup=None,
                        showlegend=False,
                        mode='lines',
                        line=dict(color='black', width=100), 
                    )
            fig.add_trace(line)
            xpos = heatmap_x_frac * cluster_position
            fig.update_layout(**{
                # format to 2 digits with leading zero
                'xaxis' + axis_num: {
                    'domain': [xpos, xpos + 0.002], 
                    'showgrid': False, 
                    'showline': False, 
                    'zeroline': False, 
                    'showticklabels': False,
                    'visible': False,
                },
                'yaxis' + axis_num: {
                    'domain': [0., heatmap_frac + spectrums_frac + divider], 
                    'showgrid': False, 
                    'showline': False, 
                    'zeroline': False, 
                    'showticklabels': False,
                    'visible': False,
                    'range': [0, 1],
                },

                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
            })

    # Plot!
    if savepath is None:
        savepath = 'tempplot.pdf'
    fig.write_image(savepath, scale=1.)
    return IFrame(savepath, width=1000, height=total_height)


if __name__ == '__main__':
    n_spectrums = 20
    n_features = 10
    n_patients = 100
    networks_dummy = np.random.normal(0, 1, (n_patients, n_features))
    spectrums = [(10 * np.random.randint(0, 10, n_patients)).astype(float) for i in range(n_spectrums)]
    spectrums[1][3] = np.nan
    spectrum_labels = [f'spectrum percent {i}' for i in range(n_spectrums)]
    spectrum_types = np.random.choice(['categorical', 'continuous'], n_spectrums, replace=True)
    colors = np.random.choice(['Blues', 'viridis', 'plasma'], n_spectrums, replace=True)
    show_legends = [True] * 5 + [False] * (n_spectrums - 5)

    # spectrums = [np.random.uniform(-1, 1, 100), np.random.normal(0, 1, 100), np.random.randint(0, 10, 100)]
    # spectrum_labels = ['unif', 'normal', 'randint']
    # spectrum_types = ['continuous', 'continuous', 'categorical']
    # colors = ['Blues', 'Viridis', 'tab20']
    # show_legends = [True, False, True]
    # plot_dendrogram(networks_dummy, title='hello dendrogram', method='ward', spectrums=spectrums,
    #                 spectrum_labels=spectrum_labels, spectrum_types=spectrum_types, colors=colors,
    #                 show_legends=show_legends)
    plot_dendrogram(networks_dummy, title='hello dendrogram', method='ward', spectrums=spectrums,
                    spectrum_labels=spectrum_labels, spectrum_types=spectrum_types, colors=colors,
                    show_legends=show_legends, dendro_height=1000)  # easy way to make legends visible

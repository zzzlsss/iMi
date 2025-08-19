# from turtle import width
import numpy as np
import pandas as pd
import re
import astropy.io.fits as fits
import astropy.visualization as apvis
from astropy.wcs import WCS
# import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='astropy')

import panel as pn
pn.extension("mathjax")

import holoviews as hv
from holoviews.operation.datashader import rasterize
from holoviews import opts
from holoviews.streams import Selection1D
from bokeh.models import HoverTool, NumeralTickFormatter
hv.extension('bokeh') # 'matplotlib') # 

# zakapo = True

# if zakapo:
#     FITS_FILE_PATH = '/Volumes/ZLS HD/PhD_Documents/Astro_Projects/Ice_Proposals/IceAge_ERS/Spectral_Extraction_Code/Real_Data_Code/FW_Files/IceAge_CHAMMS1-C2-FIELD_lw_F410M_visitall_modall_i2d.fits'
#     PICKLE_FILE_PATH = "/Users/zaklukasmith/Documents/IceMapping1/Ice_N_values_DFs/G95_All_Ice_Map.pkl"
# else:
#     FITS_FILE_PATH = "/Users/hjd229/Documents/Data/Zak/IceAge_CHAMMS1-C2-FIELD_lw_F410M_visitall_modall_i2d.fits"
#     PICKLE_FILE_PATH = "/Users/hjd229/Documents/Data/Zak/G95_All_Ice_Map.pkl"

img_data_path = './IceAge_Original_Data/IA_F410M_img_data.npy'
img_data = np.load(img_data_path)

wcs_path = './IceAge_Original_Data/IA_F410M_wcs.pkl'
wcs = pd.read_pickle(wcs_path)

cat_path = './IceAge_Original_Data/IA_F410M_cat.pkl'
cat = pd.read_pickle(cat_path)

pixels = wcs.world_to_pixel_values(cat['H2O_RA'].values, cat['H2O_Dec'].values)
cat['x_pix'], cat['y_pix'] = pixels[0], pixels[1]   
cat['ID'] = cat.index

## ENSURES THAT SEARCH BY ID WORKS AND NEEDED AS SELECETED OPTION IN POINTS IS DONE BY ILOC!!!
cat.reset_index(drop=True, inplace=True)

# cat = cat[['ID', 'x_pix', 'y_pix', 'H2O_RA', 'H2O_Dec', 'H2O_N', 'H2O_N_err_lower', 'H2O_N_err_upper', 'H2O_WLs',]]# 'H2O_Fluxes', 'H2O_FluxErrs','H2O_Baseline', 'H2O_Baseline_err', 'H2O_OD_spec', 'H2O_OD_spec_err']]

## Cannot find way to label hover points with scientific notation therefore...
cat['sci_H2O_N'] = cat['H2O_N'].apply(lambda x: f"{x:.3e}")
cat['sci_H2O_N_err_lower'] = cat['H2O_N_err_lower'].apply(lambda x: f"{x:.3e}")
cat['sci_H2O_N_err_upper'] = cat['H2O_N_err_upper'].apply(lambda x: f"{x:.3e}")
cat['sci_CO2_N'] = cat['CO2_N'].apply(lambda x: f"{x:.3e}")
cat['sci_CO2_N_err_lower'] = cat['CO2_N_err_lower'].apply(lambda x: f"{x:.3e}")
cat['sci_CO2_N_err_upper'] = cat['CO2_N_err_upper'].apply(lambda x: f"{x:.3e}")
cat['sci_CO_N'] = cat['CO_N'].apply(lambda x: f"{x:.3e}")
cat['sci_CO_N_err_lower'] = cat['CO_N_err_lower'].apply(lambda x: f"{x:.3e}")
cat['sci_CO_N_err_upper'] = cat['CO_N_err_upper'].apply(lambda x: f"{x:.3e}")
cat['sci_H2_N'] = cat['H2_N'].apply(lambda x: f"{x:.3e}")



""" 
Non-linked materials
- Rasterised FITS Image
- Overlaid catalog labels 
"""

img_data = np.flipud(data[1].data)
norm = apvis.ImageNormalize(img_data, stretch=apvis.HistEqStretch(img_data), clip=True)

img = rasterize(
    hv.Image(img_data.astype(np.float32),bounds=(0, 0, data[1].header['NAXIS1'], data[1].header['NAXIS2'])).opts(cnorm='eq_hist',),#clim=(norm.vmin, norm.vmax)),
    precompute=True,
).opts(colorbar=True, cmap='gist_heat', width=600, height=600)


"""Utility data class to convert arbitrary LaTeX-like strings into 
appropriate format for plotting as axis labels with Bokeh"""

class RenderLatexLabels(hv.streams.Stream):

    do_render_latex_labels = hv.param.Boolean(default=False, doc="Should labels be rendered as latex?")

    def make_latex_label_string(self, latex_like_string):
        if not self.do_render_latex_labels:
            return latex_like_string
        latex_label_string= re.sub(r"([^\^_0-9]+)", r"\\text{{\1}}", latex_like_string.replace("$", ""))
        return f"$${latex_label_string}$$"

rll = RenderLatexLabels(do_render_latex_labels = False, transient=False)

""" 
Linked materials
- Catalog points overlayed on the image - index_map
- H2 vs Ice CDs (H2O, CO2, CO) scatter plots - index_H2O, index_CO2, index_CO
- Ternary plot (future work) - index_ternary
- Flux and OD Spectrum plots - points linked to plot these spectra 

A selection1D stream for each plot is created to capture selections from the points plot.
Ensure tap and lasso_select tools are enabled in the points plot.

This will allow us to select points and update the spectrum plot dynamically
"""

class SelectedIndices(hv.streams.Stream):
    selected_indices = hv.param.List(default=[], doc="Selected object indices")


selected_indices = SelectedIndices(selected_indices = [], transient=False)

def update_selected_indices(index=[], index_CO2=[], index_CO = [], index_H2O=[], index_H2O_CO2=[], index_H2O_CO=[], index_CO2_CO=[],index_table=[]):
    combined_indices = list(set(index + index_CO2 + index_CO + index_H2O + index_H2O_CO2 + index_H2O_CO + index_CO2_CO + index_table))
    selected_indices.event(selected_indices=combined_indices)

def plot_source_locations(selected_indices, *args, **kwargs):
    points = hv.Points(cat, kdims=['x_pix', 'y_pix'],vdims=['ID','H2O_RA','H2O_Dec',
                                                            'sci_H2O_N','H2O_N_err_lower','H2O_N_err_upper',
                                                            'sci_CO2_N','CO2_N_err_lower','CO2_N_err_upper',
                                                            'sci_CO_N','CO_N_err_lower','CO_N_err_upper',
                                                            'sci_H2_N']).opts(
        marker='square', size=6, color='blue', fill_color=None,
        tools=['tap','lasso_select', 'hover'],
        hover_tooltips=[('ID', '@ID'),
                        ('RA', '@H2O_RA'), 
                        ('Dec', '@H2O_Dec'), 
                        ('N H2O', '@sci_H2O_N'),  # scientific notation
                        # ('H2O_N_err_lower', '@H2O_N_err_lower'), 
                        # ('H2O_N_err_upper', '@H2O_N_err_upper'), 
                        ('N CO2', '@sci_CO2_N'), 
                        # ('CO2_N_err_lower', '@CO2_N_err_lower'), 
                        # ('CO2_N_err_upper', '@CO2_N_err_upper')
                        ('N CO', '@sci_CO_N'), 
                        # ('CO_N_err_lower', '@CO_N_err_lower'), 
                        # ('CO_N_err_upper', '@CO_N_err_upper')
                        ('N H2', '@sci_H2_N'),
                        ], 
        selected=selected_indices,
        selection_color='green', 
        selection_alpha=1,
        nonselection_alpha=0.4,
        )
    return points

points = hv.DynamicMap(plot_source_locations, streams=[selected_indices])
## Create a stream to capture selections from the points plot
# This will allow us to update the spectrum plot based on selected points
points_stream = hv.streams.Selection1D(source=points) #.rename(index='index_map')
points_stream.add_subscriber(update_selected_indices)


def plot_labels(selected_indices=[], show_labels=True, *args, **kwargs):
    """
    Plot labels for selected points in the catalog.
    """
    # valid_indices = [i for i in selected_indices if 0 <= i < len(cat)]
    if selected_indices:
        # If there are valid selected indices, plot labels for those points
        labels = hv.Labels(cat.iloc[selected_indices], kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(
            text_color='blue', text_font_size='11pt', yoffset=15,
        )
    else:
        if show_labels:
            # If no points are selected but labels should be shown, plot all labels
            labels = hv.Labels(cat, kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(
                text_color='blue', text_font_size='11pt', yoffset=15,
            )
        else:
            # If no points are selected and labels should not be shown, return an empty Labels object
            labels = hv.Labels([], kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(
                text_color='blue', text_font_size='11pt', yoffset=15,
            )
    return labels

# Panel toggle button to show/hide labels
label_toggle = pn.widgets.Toggle(name='Show/Hide ID Labels', value=True, button_type='primary')

# Use a Param stream to link the toggle button to the DynamicMap
class ShowLabelsStream(hv.streams.Stream):
    show_labels = hv.param.Boolean(default=True)

show_labels_stream = ShowLabelsStream(show_labels=label_toggle.value)

def update_show_labels(event):
    show_labels_stream.event(show_labels=event.new)

label_toggle.param.watch(update_show_labels, 'value')

# DynamicMap for labels, using the toggle button value
labels = hv.DynamicMap(
    lambda selected_indices, show_labels, *args, **kwargs: plot_labels(selected_indices, show_labels),
    streams=[selected_indices, show_labels_stream]
)

## Create a stream to capture selections from the points plot
# This will allow us to update the spectrum plot based on selected points
points_stream = hv.streams.Selection1D(source=points) #.rename(index='index_map')
points_stream.add_subscriber(update_selected_indices)

def plot_h2_vs_h2o(selected_indices, *args, **kwargs):
    indices = selected_indices if selected_indices and len(selected_indices) > 0 else []

    # # Prepare data for errorbars
    # x = cat['H2_N'].values
    # y = cat['H2O_N'].values
    # yerr_lower = cat['H2O_N_err_lower'].values
    # yerr_upper = cat['H2O_N_err_upper'].values

    # # Holoviews ErrorBars expects (x, y, yerr_neg, yerr_pos)
    # errorbars = hv.ErrorBars(
    #     (x, y, y - yerr_lower, y + yerr_upper)
    # ).opts(color='gray', alpha=0.4, line_width=1)

    if indices:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'H2O_N'], vdims=['ID', 'sci_H2_N', 'sci_H2O_N']
        ).opts(width=400, height=400,
            color='blue', size=6, marker='circle', alpha=0.7,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N H$_2$O'), title=rll.make_latex_label_string('N H$_2$ vs. N H$_2$O'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N H$_2$', '@sci_H2_N'),
            ], 
            selected=indices,
            nonselection_alpha=0.1,
        )

    else:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'H2O_N'], vdims=['ID','sci_H2_N', 'sci_H2O_N']
        ).opts(width=400, height=400,
            color='blue', size=6, marker='circle', alpha=0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N H$_2$O'), title=rll.make_latex_label_string('N H$_2$ vs. N H$_2$O'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N H$_2$', '@sci_H2_N'),
            ]
        )

    return scatter # * errorbars 

scatter_H2O = hv.DynamicMap(plot_h2_vs_h2o, streams=[selected_indices, rll])
scatter_H2O_stream = hv.streams.Selection1D(source=scatter_H2O).rename(index='index_H2O')
scatter_H2O_stream.add_subscriber(update_selected_indices)

def plot_h2_vs_co2(selected_indices, *args, **kwargs):
    indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
    if indices:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'CO2_N'], vdims=['ID','sci_H2_N', 'sci_CO2_N']
        ).opts(width=400, height=400,
            color='purple', size=6, marker='circle', alpha=0.7,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H_2'), ylabel=rll.make_latex_label_string('N CO_2'), title=rll.make_latex_label_string('N H_2 vs. N CO_2'),
            hover_tooltips=[('ID', '@ID'), 
            ('N H$_2$', '@sci_H2_N'), 
            ('N CO$_2$', '@sci_CO2_N')],
            selected=indices,
            nonselection_alpha=0.1,
        )
    else:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'CO2_N'], vdims=['ID','sci_H2_N', 'sci_CO2_N']
        ).opts(width=400, height=400,
            color='purple', size=6, marker='circle', alpha=0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N H$_2$ vs. N CO$_2$'),
            hover_tooltips=[('ID', '@ID'), 
            ('N H$_2$', '@sci_H2_N'), 
            ('N CO$_2$', '@sci_CO2_N')],
        )
    return scatter

scatter_CO2 = hv.DynamicMap(plot_h2_vs_co2, streams=[selected_indices, rll])
scatter_CO2_stream = hv.streams.Selection1D(source=scatter_CO2).rename(index='index_CO2')
scatter_CO2_stream.add_subscriber(update_selected_indices)


def plot_h2_vs_co(selected_indices, *args, **kwargs):
    indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
    if indices:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'CO_N'], vdims=['ID','sci_H2_N', 'sci_CO_N']
        ).opts(
            width=400, height=400,
            color='green', size=6, marker='circle', alpha=0.7,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N CO'), title=rll.make_latex_label_string('N H$_2$ vs. N CO'),
            hover_tooltips=[('ID', '@ID'), ('N H$_2$', '@sci_H2_N'), ('N CO', '@sci_CO_N')],
            selected=indices,
            nonselection_alpha=0.1,
        )
    else:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'CO_N'], vdims=['ID','sci_H2_N', 'sci_CO_N']
        ).opts(
            width=400, height=400,
            color='green', size=6, marker='circle', alpha=0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N CO'), title=rll.make_latex_label_string('N H$_2$ vs. N CO'),
            hover_tooltips=[('ID', '@ID'), ('N H$_2$', '@sci_H2_N'), ('N CO', '@sci_CO_N')],
        )
    return scatter

scatter_CO = hv.DynamicMap(plot_h2_vs_co, streams=[selected_indices, rll])
scatter_CO_stream = hv.streams.Selection1D(source=scatter_CO).rename(index='index_CO')
scatter_CO_stream.add_subscriber(update_selected_indices)

""" Ice Ratio Plots """

def plot_h2o_vs_co2(selected_indices, *args, **kwargs):
    indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
    if indices:
        scatter = hv.Points(
            cat,
            kdims=['H2O_N', 'CO2_N'], vdims=['ID', 'sci_H2O_N', 'sci_CO2_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.7,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$O'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N H$_2$O vs. N CO$_2$'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N CO$_2$', '@sci_CO2_N'),
            ],
            selected=indices,
            nonselection_alpha=0.1,
        )
    else:
        scatter = hv.Points(
            cat,
            kdims=['H2O_N', 'CO2_N'], vdims=['ID', 'sci_H2O_N', 'sci_CO2_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$O'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N H$_2$O vs. N CO$_2$'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N CO$_2$', '@sci_CO2_N'),
            ],
        )
    return scatter

scatter_H2O_CO2 = hv.DynamicMap(plot_h2o_vs_co2, streams=[selected_indices, rll])
scatter_H2O_CO2_stream = hv.streams.Selection1D(source=scatter_H2O_CO2).rename(index='index_H2O_CO2')
scatter_H2O_CO2_stream.add_subscriber(update_selected_indices)

def plot_h2o_vs_co(selected_indices, *args, **kwargs):
    indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
    if indices:
        scatter = hv.Points(
            cat,
            kdims=['H2O_N', 'CO_N'], vdims=['ID', 'sci_H2O_N', 'sci_CO_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.7,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$O'), ylabel=rll.make_latex_label_string('N CO'), title=rll.make_latex_label_string('N H$_2$O vs. N CO'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N CO', '@sci_CO_N'),
            ],
            selected=indices,
            nonselection_alpha=0.1,
        )
    else:
        scatter = hv.Points(
            cat,
            kdims=['H2O_N', 'CO_N'], vdims=['ID', 'sci_H2O_N', 'sci_CO_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$O'), ylabel=rll.make_latex_label_string('N CO'), title=rll.make_latex_label_string('N H$_2$O vs. N CO'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N CO', '@sci_CO_N'),
            ],
        )
    return scatter

scatter_H2O_CO = hv.DynamicMap(plot_h2o_vs_co, streams=[selected_indices, rll])
scatter_H2O_CO_stream = hv.streams.Selection1D(source=scatter_H2O_CO).rename(index='index_H2O_CO')
scatter_H2O_CO_stream.add_subscriber(update_selected_indices)

def plot_co_vs_co2(selected_indices, *args, **kwargs):
    indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
    if indices:
        scatter = hv.Points(
            cat,
            kdims=['CO_N', 'CO2_N'], vdims=['ID', 'sci_CO_N', 'sci_CO2_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.7,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N CO'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N CO vs. N CO$_2$'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N CO$_2$', '@sci_CO2_N'),
                ('N CO', '@sci_CO_N'),
            ],
            selected=indices,
            nonselection_alpha=0.1,
        )
    else:
        scatter = hv.Points(
            cat,
            kdims=['CO_N', 'CO2_N'], vdims=['ID', 'sci_CO_N', 'sci_CO2_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N CO'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N CO vs. N CO$_2$'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N CO$_2$', '@sci_CO2_N'),
                ('N CO', '@sci_CO_N'),
            ],
        )
    return scatter 

scatter_CO2_CO = hv.DynamicMap(plot_co_vs_co2, streams=[selected_indices, rll])
scatter_CO2_CO_stream = hv.streams.Selection1D(source=scatter_CO2_CO).rename(index='index_CO2_CO')
scatter_CO2_CO_stream.add_subscriber(update_selected_indices)

""" Search bar for ID search - updates selected_indices stream with the index of the searched ID"""
search_bar = pn.widgets.TextInput(name='Search by ID', placeholder='Enter Spectrum ID / List of IDs (comma-separated)')

def search_selected_indices(search_id):
    if search_id:
        try:
            # Support comma-separated list of IDs
            search_ids = [int(s.strip()) for s in search_id.split(',') if s.strip().isdigit()]
            indices = cat[cat['ID'].isin(search_ids)].index.tolist()
            # indices = cat[cat['ID'] == search_id_int].index.tolist()
            selected_indices.event(selected_indices=indices)
            # Update all scatter plot selection streams to reflect the search
            points_stream.event(index=indices)
            scatter_H2O_stream.event(index=indices)
            scatter_CO2_stream.event(index=indices)
            scatter_CO_stream.event(index=indices)
            scatter_H2O_CO2_stream.event(index=indices)
            scatter_H2O_CO_stream.event(index=indices)
            scatter_CO2_CO_stream.event(index=indices)
        except ValueError:
            selected_indices.event(selected_indices=[])
            points_stream.event(index=[])
            scatter_H2O_stream.event(index=[])
            scatter_CO2_stream.event(index=[])
            scatter_CO_stream.event(index=[])
            scatter_H2O_CO2_stream.event(index=[])
            scatter_H2O_CO_stream.event(index=[])
            scatter_CO2_CO_stream.event(index=[])

search_bar.param.watch(lambda event: search_selected_indices(event.new.strip()) if event is not None and event.new is not None and event.new.strip() != "" else selected_indices.event(selected_indices=[]), 'value')



""" Spectrum Plots """
# def plot_spectrum(index, index_H2O, index_CO2, index_CO):
def plot_spectrum(selected_indices, *args, **kwargs):
    indices = (selected_indices if selected_indices and len(selected_indices) > 0 else [])
    if indices:
        overlays = []
        color_cycle = ['black', 'red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']
        for num, i in enumerate(indices):
            row = cat.iloc[i]

            title = f"Spectrum ID: {row['ID']}" if len(indices) == 1 else f"Spectrum IDs {', '.join(str(cat.iloc[j]['ID']) for j in indices)}"
            color = 'black' if num == 0 else color_cycle[num % len(color_cycle)]

            h2o_wls = np.array(row['H2O_WLs'])
            h2o_fluxes = np.array(row['H2O_Fluxes'])
            h2o_baseline = np.array(row['H2O_Baseline'])
            
            
            curve = hv.Curve((h2o_wls, h2o_fluxes), 'Wavelength (μm)', 'Flux (mJy)').opts(
                xlim=(2.4, 5.1), ylim=(1e-3, 0.7), logy=True, line_width=0.75, color=color, title=title
            )
            baseline_curve = hv.Curve((h2o_wls, h2o_baseline)).opts(color='blue', line_dash='dashed', alpha=0.7)

            # Add CO2 spectrum if available
            # if 'CO2_WLs' in row and 'CO2_Fluxes' in row and row['CO2_WLs'] is not None and row['CO2_Fluxes'] is not None:
            co2_wls = np.array(row['CO2_WLs'])
            co2_fluxes = np.array(row['CO2_Fluxes'])
            co2_baseline = np.array(row['CO2_Baseline'])
            co2_curve = hv.Curve((co2_wls, co2_fluxes), 'Wavelength (μm)', 'CO2 Flux (mJy)').opts(
                color=color, line_width=0.75
            )
            baseline_co2_curve = hv.Curve((co2_wls, co2_baseline)).opts(color='purple', line_dash='dashed', alpha=0.7)

            # Add CO spectrum if available
            # if 'CO_WLs' in row and 'CO_Fluxes' in row and row['CO_WLs'] is not None and row['CO_Fluxes'] is not None:
            co_wls = np.array(row['CO_WLs'])
            co_fluxes = np.array(row['CO_Fluxes'])
            co_baseline = np.array(row['CO_Baseline'])
            co_curve = hv.Curve((co_wls, co_fluxes), 'Wavelength (μm)', 'CO Flux (mJy)').opts(
                color=color, line_width=0.75
            )
            baseline_co_curve = hv.Curve((co_wls, co_baseline)).opts(color='green', line_dash='dashed', alpha=0.7)

            overlays.append(curve * baseline_curve * co2_curve * baseline_co2_curve * co_curve * baseline_co_curve)
            # else:
            #     overlays.append(curve * baseline_curve)
    else:
        overlays = [hv.Curve([], 'Wavelength (μm)', 'Flux').opts(title="No selection") * hv.Curve([], 'Wavelength (μm)', 'Flux (mJy)')]

    return hv.Overlay(overlays).opts(width=600, height=200, xlim=(2.4, 5.1), ylim=(1e-3, 0.7), logy=True)

# def plot_od_spectrum(index, index_H2O, index_CO2, index_CO):
def plot_od_spectrum(selected_indices, *args, **kwargs):
    indices = (selected_indices if selected_indices and len(selected_indices) > 0 else [])

    overlays = []
    color_cycle = ['black', 'red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']
    
    if indices:
        for num, i in enumerate(indices):
            row = cat.iloc[i]

            title = f"OD Spectrum ID {row['ID']}" if len(indices) == 1 else f"OD Spectrum IDs {', '.join(str(cat.iloc[j]['ID']) for j in indices)}"
            color = color_cycle[num % len(color_cycle)] if len(indices) > 1 else 'black'
            if num == 0:
                color = 'black'

            h2o_wls = np.array(row['H2O_WLs'])
            h2o_od = np.array(row['H2O_OD_spec'])

            h2o_od_curve = hv.Curve((h2o_wls, h2o_od), 'Wavelength (μm)', 'Optical Depth').opts(color=color, title=title, alpha=0.75, line_width=0.75)

            # Add fill_between region for H2O OD between 2.715 and 3.35 μm
            h2o_mask = (h2o_wls >= 2.715) & (h2o_wls <= 3.35) & (h2o_od > 0)
            if np.any(h2o_mask):
                fill_between = hv.Area((h2o_wls[h2o_mask], h2o_od[h2o_mask])).opts(
                    color='lightblue', alpha=0.4, line_alpha=0
                )
                h2o_od_curve = h2o_od_curve * fill_between

            co2_wls = np.array(row['CO2_WLs'])
            co2_od = np.array(row['CO2_OD_spec'])

            co2_od_curve = hv.Curve((co2_wls, co2_od), 'Wavelength (μm)', 'Optical Depth').opts(color=color, alpha=0.75, line_width=0.75)

            # Add fill_between region for CO2 OD between 4.2 and 4.34 μm
            co2_mask = (co2_wls >= 4.2) & (co2_wls <= 4.34) & (co2_od > 0)
            if np.any(co2_mask):
                fill_between = hv.Area((co2_wls[co2_mask], co2_od[co2_mask])).opts(
                    color='purple', alpha=0.4, line_alpha=0
                )
                co2_od_curve = co2_od_curve * fill_between

            co_wls = np.array(row['CO_WLs'])
            co_od = np.array(row['CO_OD_spec'])

            co_od_curve = hv.Curve((co_wls, co_od), 'Wavelength (μm)', 'Optical Depth').opts(color=color, alpha=0.75, line_width=0.75)

            # Add fill_between region for CO OD between 4.65 and 4.705 μm
            co_mask = (co_wls >= 4.65) & (co_wls <= 4.705) & (co_od > 0)
            if np.any(co_mask):
                fill_between = hv.Area((co_wls[co_mask], co_od[co_mask])).opts(
                    color='green', alpha=0.4, line_alpha=0
                )
                co_od_curve = co_od_curve * fill_between

            overlays.append(h2o_od_curve * co2_od_curve * co_od_curve)
    else:
        overlays = [hv.Curve([], 'Wavelength (μm)', 'Optical Depth').opts(title="No selection")]

    return hv.Overlay(overlays).opts(
            width=600, height=200, xlim=(2.4, 5.1), ylim=(-0.2, 5), 
        )
    
# Shows the zero continuum line
# Plotted only for first source as if done within loop, 
# the od spectra are not plotted after 2 sources... 
# Seemingly cannot plot every continuum line for each source in flux
# if num == 0:
#     baseline_curve = hv.Curve((wls, np.zeros_like(wls)), 'Wavelength (μm)', 'Optical Depth').opts(
#         color='red', line_dash='dashed', alpha=0.7,
#     )
    
#     overlays.append(baseline_curve)

def source_info_table(selected_indices):
    valid_indices = [i for i in selected_indices if 0 <= i < len(cat)]
    columns = ['ID', 'H2O_RA', 'H2O_Dec', 
               'sci_H2O_N', 'sci_H2O_N_err_upper', 'sci_H2O_N_err_lower',
               'sci_CO2_N', 'sci_CO2_N_err_upper', 'sci_CO2_N_err_lower', 
               'sci_CO_N', 'sci_CO_N_err_upper', 'sci_CO_N_err_lower', 
               'sci_H2_N']
    rename_dict = {
        'H2O_RA': 'RA',
        'H2O_Dec': 'Dec',
        'sci_H2O_N': 'N H2O',
        'sci_H2O_N_err_upper': 'N H2O err upp',
        'sci_H2O_N_err_lower': 'N H2O err low',
        'sci_CO2_N': 'N CO2',
        'sci_CO2_N_err_upper': 'N CO2 err upp',
        'sci_CO2_N_err_lower': 'N CO2 err low',
        'sci_CO_N': 'N CO',
        'sci_CO_N_err_upper': 'N CO err upp',
        'sci_CO_N_err_lower': 'N CO err low',
        'sci_H2_N': 'N H2'
    }
    # If selection is from index_table, show full DataFrame
    if selected_indices is not None and set(table_stream.index) == set(selected_indices):
        df = cat[columns].reset_index(drop=True)
        df = df.rename(columns=rename_dict)
    elif valid_indices is None or len(valid_indices) == 0:
        # Show full DataFrame if no selection
        df = cat[columns].reset_index(drop=True)
        df = df.rename(columns=rename_dict)

    else:
        df = cat.iloc[valid_indices][columns].reset_index(drop=True)
        df = df.rename(columns=rename_dict)
        if df.empty:
            df = pd.DataFrame([{col: "" for col in df.columns}])

    return hv.Table(df)

table = hv.DynamicMap(source_info_table, streams=[selected_indices]).opts(
    width=600, height=100, 
    # tools=['hover'], 
    # show_index=False, 
    title='Source Information Table',
    selectable=True, # can be selectable but pointless as I only build table from selected indices
    # selection_policy='union'
)

table_stream = hv.streams.Selection1D(source=table).rename(index='index_table')
table_stream.add_subscriber(update_selected_indices)

""" All plots for app layout here """
# Pair the plots so that selections in one update the other and axes stay synced
layout = (img * points * labels) # * labels_but)

spectrum_map = hv.DynamicMap(plot_spectrum,streams=[selected_indices]) # streams=[points_stream, scatter_H2O_stream, scatter_CO2_stream, scatter_CO_stream])
od_spectrum_map = hv.DynamicMap(plot_od_spectrum,streams=[selected_indices]) # streams=[points_stream, scatter_H2O_stream, scatter_CO2_stream, scatter_CO_stream])

app_bar = pn.Row(
    pn.pane.Markdown('## <span style="color:white">ice Mapping interface (iMi)</span>', width=1000, sizing_mode="fixed", margin=(10,5,10,15)), 
    pn.Spacer(),
    pn.pane.PNG("http://holoviews.org/_static/logo.png", height=50, width=50, sizing_mode="fixed", align="center"),
    pn.pane.PNG("https://panel.holoviz.org/_static/logo_horizontal.png", height=50, width=100, sizing_mode="fixed", align="center"), 
    styles={'background': 'black'},
)
# app_bar


accord=pn.Accordion(
            ("H2 vs Ice Column Density Correlation Plots", pn.Row(
                scatter_H2O,
                scatter_CO2,
                scatter_CO,
            )),
            ("Ice Column Density Ratio Plots", pn.Row(
                scatter_H2O_CO2,
                scatter_H2O_CO,
                scatter_CO2_CO,
            )),
            sizing_mode='fixed',
            width=1200,
            height=400,
        )

def toggle_latex_labels(*args, **kwargs):
    if not rll.do_render_latex_labels:
        rll.event(do_render_latex_labels = True)

accord.param.watch(toggle_latex_labels, 'active')

if __name__ == "__main__":
    app = pn.Column(
        app_bar,
        pn.Spacer(height=10),
        pn.Row(label_toggle, search_bar),
        pn.Row(
            layout, 
            pn.Column(
                spectrum_map,
                od_spectrum_map,
                table,
                ),
        ),
        accord,
        sizing_mode='stretch_both',
        # margin=(10, 10, 10, 10),
        css_classes=['imi-dashboard']
    )

    # Launch Panel app in browser
    pn.serve(app, show=True, title="ice Mapping interface (iMi)")

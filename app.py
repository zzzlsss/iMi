import numpy as np
import pandas as pd
import re
import astropy.io.fits as fits
import astropy.visualization as apvis
from astropy.wcs import WCS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='astropy')

import panel as pn
pn.extension("mathjax")

import holoviews as hv
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')

import os
import shutil

# # --- Cleanup code: delete downloaded data on session end ---
# def cleanup_files():
#     files_to_remove = [
#         "IceAge_Original_Data/IA_F410M_WCS.pkl",
#         "IceAge_Original_Data/IA_F410M_img_data.npy",
#         "IceAge_Original_Data/Smith2025_Data.pkl",
#     ]
#     for f in files_to_remove:
#         try:
#             os.remove(f)
#             print(f"Removed {f}")
#         except FileNotFoundError:
#             pass
#         except Exception as e:
#             print(f"Error removing {f}: {e}")

#     try:
#         shutil.rmtree("IceAge_Original_Data", ignore_errors=True)
#         print("Removed directory IceAge_Original_Data")
#     except Exception as e:
#         print(f"Error removing directory: {e}")

# pn.state.on_session_destroyed(lambda session_context: cleanup_files())
# # --- End cleanup code ---

from utils_gdrive import download_multiple_files
from functools import lru_cache

FILE_MAP = {
    "IceAge_Original_Data/IA_F410M_WCS.pkl":      "1BPXKvzOOiPMOBVW3-vPS7Tg0EOMa4Gj_",
    "IceAge_Original_Data/IA_F410M_img_data.npy": "1Rz3xQiPRgJcVxAAYkI2zTKznEwZ4kMQT", 
    "IceAge_Original_Data/Smith2025_Data.pkl":    "1Vrxb-ZtjZ03HGhu2EP9FfYzQlsOFPlxb",
}
download_multiple_files(FILE_MAP)

@lru_cache(maxsize=2)
def get_wcs():
    wcs = pd.read_pickle("IceAge_Original_Data/IA_F410M_WCS.pkl")
    img, orig_shape, factor = get_img_data()
    new_shape = img.shape
    wcs.naxis1 = new_shape[1]
    wcs.naxis2 = new_shape[0]
    if hasattr(wcs, 'wcs'):
        wcs.wcs.cdelt = wcs.wcs.cdelt * factor
        wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / factor + 0.5
    return wcs

@lru_cache(maxsize=2)
def get_img_data():
    img = np.load("IceAge_Original_Data/IA_F410M_img_data.npy")
    orig_shape = img.shape
    # Downsample to max 1500x1500 pixels for memory safety!
    max_dim = 1500
    factor = max(img.shape[0] // max_dim, img.shape[1] // max_dim, 1)
    if factor > 1:
        img = img[::factor, ::factor]
    return img, orig_shape, factor

@lru_cache(maxsize=2)
def get_cat():
    return pd.read_pickle("IceAge_Original_Data/Smith2025_Data.pkl")

def make_app():
    def get_pixels():
        cat = get_cat()
        wcs = get_wcs()
        pixels = wcs.world_to_pixel_values(cat['H2O_RA'].values, cat['H2O_Dec'].values)
        return pixels

    def get_cat_with_pixels():
        cat = get_cat().copy()
        pixels = get_pixels()
        cat['x_pix'], cat['y_pix'] = pixels[0], pixels[1]   
        cat['ID'] = cat.index
        cat.reset_index(drop=True, inplace=True)
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
        return cat

    def get_norm():
        img_data,_,_ = get_img_data()
        return apvis.ImageNormalize(img_data, stretch=apvis.HistEqStretch(img_data), clip=True)
    def get_img():
        img_data,_,_ = get_img_data()
        return rasterize(
            hv.Image(img_data.astype(np.float32),bounds=(0, 0, img_data.shape[1], img_data.shape[0])).opts(cnorm='eq_hist',),
            precompute=True,
        ).opts(colorbar=True, cmap='gist_heat', width=600, height=600)

    # --- STREAMS AND STATE ---
    class RenderLatexLabels(hv.streams.Stream):
        do_render_latex_labels = hv.param.Boolean(default=False)
        def make_latex_label_string(self, latex_like_string):
            if not self.do_render_latex_labels:
                return latex_like_string
            latex_label_string= re.sub(r"([^\^_0-9]+)", r"\\text{{\1}}", latex_like_string.replace("$", ""))
            return f"$${latex_label_string}$$"
    rll = RenderLatexLabels(do_render_latex_labels = False, transient=False)

    class SelectedIndices(hv.streams.Stream):
        selected_indices = hv.param.List(default=[], doc="Selected object indices")
    selected_indices = SelectedIndices(selected_indices = [], transient=False)

    def update_selected_indices(index=[], index_CO2=[], index_CO = [], index_H2O=[], index_H2O_CO2=[], index_H2O_CO=[], index_CO2_CO=[],index_table=[]):
        combined_indices = list(set(index + index_CO2 + index_CO + index_H2O + index_H2O_CO2 + index_H2O_CO + index_CO2_CO + index_table))
        selected_indices.event(selected_indices=combined_indices)

    # --- PLOTTING FUNCTIONS ---
    def plot_source_locations(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
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
                            ('N H2O', '@sci_H2O_N'), 
                            ('N CO2', '@sci_CO2_N'), 
                            ('N CO', '@sci_CO_N'), 
                            ('N H2', '@sci_H2_N'),
                            ], 
            selected=selected_indices,
            selection_color='green', 
            selection_alpha=1,
            nonselection_alpha=0.4,
            )
        return points

    points = hv.DynamicMap(plot_source_locations, streams=[selected_indices])
    points_stream = hv.streams.Selection1D(source=points)
    points_stream.add_subscriber(update_selected_indices)

    def plot_labels(selected_indices=[], show_labels=True, *args, **kwargs):
        cat = get_cat_with_pixels()
        if selected_indices:
            labels = hv.Labels(cat.iloc[selected_indices], kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(
                text_color='blue', text_font_size='11pt', yoffset=15,
            )
        else:
            if show_labels:
                labels = hv.Labels(cat, kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(
                    text_color='blue', text_font_size='11pt', yoffset=15,
                )
            else:
                labels = hv.Labels([], kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(
                    text_color='blue', text_font_size='11pt', yoffset=15,
                )
        return labels

    label_toggle = pn.widgets.Toggle(name='Show/Hide ID Labels', value=True, button_type='primary')


    # Search bar for ID search - updates selected_indices stream with the index of the searched ID
    search_bar = pn.widgets.TextInput(name='Search by ID', placeholder='Enter Spectrum ID / List of IDs (comma-separated)')

    def search_selected_indices(search_id):
        cat = get_cat_with_pixels()
        if search_id:
            try:
                # Support comma-separated list of IDs
                search_ids = [int(s.strip()) for s in search_id.split(',') if s.strip().isdigit()]
                indices = cat[cat['ID'].isin(search_ids)].index.tolist()
                selected_indices.event(selected_indices=indices)
            except Exception:
                selected_indices.event(selected_indices=[])
        else:
            selected_indices.event(selected_indices=[])

    search_bar.param.watch(
        lambda event: search_selected_indices(event.new.strip()) if event and event.new and event.new.strip() else selected_indices.event(selected_indices=[]),
        'value'
    )


    class ShowLabelsStream(hv.streams.Stream):
        show_labels = hv.param.Boolean(default=True)
    show_labels_stream = ShowLabelsStream(show_labels=label_toggle.value)

    def update_show_labels(event):
        show_labels_stream.event(show_labels=event.new)
    label_toggle.param.watch(update_show_labels, 'value')

    labels = hv.DynamicMap(
        lambda selected_indices, show_labels, *args, **kwargs: plot_labels(selected_indices, show_labels),
        streams=[selected_indices, show_labels_stream]
    )

    # --- All scatter plot types ---
    def plot_h2_vs_h2o(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'H2O_N'], vdims=['ID', 'sci_H2_N', 'sci_H2O_N']
        ).opts(width=400, height=400,
            color='blue', size=6, marker='circle', alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N H$_2$O'), title=rll.make_latex_label_string('N H$_2$ vs. N H$_2$O'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N H$_2$', '@sci_H2_N'),
            ], 
            selected=indices,
            nonselection_alpha=0.1 if indices else 0.4,
        )
        return scatter

    def plot_h2_vs_co2(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'CO2_N'], vdims=['ID','sci_H2_N', 'sci_CO2_N']
        ).opts(width=400, height=400,
            color='purple', size=6, marker='circle', alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N H$_2$ vs. N CO$_2$'),
            hover_tooltips=[('ID', '@ID'), 
            ('N H$_2$', '@sci_H2_N'), 
            ('N CO$_2$', '@sci_CO2_N')],
            selected=indices,
            nonselection_alpha=0.1 if indices else 0.4,
        )
        return scatter

    def plot_h2_vs_co(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'CO_N'], vdims=['ID','sci_H2_N', 'sci_CO_N']
        ).opts(
            width=400, height=400,
            color='green', size=6, marker='circle', alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$'), ylabel=rll.make_latex_label_string('N CO'), title=rll.make_latex_label_string('N H$_2$ vs. N CO'),
            hover_tooltips=[('ID', '@ID'), ('N H$_2$', '@sci_H2_N'), ('N CO', '@sci_CO_N')],
            selected=indices,
            nonselection_alpha=0.1 if indices else 0.4,
        )
        return scatter

    def plot_h2o_vs_co2(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        scatter = hv.Points(
            cat,
            kdims=['H2O_N', 'CO2_N'], vdims=['ID', 'sci_H2O_N', 'sci_CO2_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$O'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N H$_2$O vs. N CO$_2$'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N CO$_2$', '@sci_CO2_N'),
            ],
            selected=indices,
            nonselection_alpha=0.1 if indices else 0.4,
        )
        return scatter

    def plot_h2o_vs_co(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        scatter = hv.Points(
            cat,
            kdims=['H2O_N', 'CO_N'], vdims=['ID', 'sci_H2O_N', 'sci_CO_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N H$_2$O'), ylabel=rll.make_latex_label_string('N CO'), title=rll.make_latex_label_string('N H$_2$O vs. N CO'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N H$_2$O', '@sci_H2O_N'),
                ('N CO', '@sci_CO_N'),
            ],
            selected=indices,
            nonselection_alpha=0.1 if indices else 0.4,
        )
        return scatter

    def plot_co_vs_co2(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        scatter = hv.Points(
            cat,
            kdims=['CO_N', 'CO2_N'], vdims=['ID', 'sci_CO_N', 'sci_CO2_N']
        ).opts(
            width=400, height=400,
            color='k', size=6, marker='circle', alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=rll.make_latex_label_string('N CO'), ylabel=rll.make_latex_label_string('N CO$_2$'), title=rll.make_latex_label_string('N CO vs. N CO$_2$'),
            hover_tooltips=[
                ('ID', '@ID'),
                ('N CO$_2$', '@sci_CO2_N'),
                ('N CO', '@sci_CO_N'),
            ],
            selected=indices,
            nonselection_alpha=0.1 if indices else 0.4,
        )
        return scatter 

    # --- DynamicMap for all plots ---
    scatter_H2O       = hv.DynamicMap(plot_h2_vs_h2o, streams=[selected_indices, rll])
    scatter_CO2       = hv.DynamicMap(plot_h2_vs_co2, streams=[selected_indices, rll])
    scatter_CO        = hv.DynamicMap(plot_h2_vs_co, streams=[selected_indices, rll])
    scatter_H2O_CO2   = hv.DynamicMap(plot_h2o_vs_co2, streams=[selected_indices, rll])
    scatter_H2O_CO    = hv.DynamicMap(plot_h2o_vs_co, streams=[selected_indices, rll])
    scatter_CO2_CO    = hv.DynamicMap(plot_co_vs_co2, streams=[selected_indices, rll])

    # Attach all stream subscribers
    for scatter, name in zip(
        [scatter_H2O, scatter_CO2, scatter_CO, scatter_H2O_CO2, scatter_H2O_CO, scatter_CO2_CO],
        ['index_H2O', 'index_CO2', 'index_CO', 'index_H2O_CO2', 'index_H2O_CO', 'index_CO2_CO']
    ):
        stream = hv.streams.Selection1D(source=scatter).rename(index=name)
        stream.add_subscriber(update_selected_indices)

    # Table
    def source_info_table(selected_indices):
        cat = get_cat_with_pixels()
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
        if valid_indices:
            df = cat.iloc[valid_indices][columns].reset_index(drop=True)
        else:
            df = cat[columns].reset_index(drop=True)
        df = df.rename(columns=rename_dict)
        if df.empty:
            df = pd.DataFrame([{col: "" for col in df.columns}])
        return hv.Table(df)
    table = hv.DynamicMap(source_info_table, streams=[selected_indices]).opts(
        width=600, height=100, 
        title='Source Information Table',
        selectable=True, 
    )
    table_stream = hv.streams.Selection1D(source=table).rename(index='index_table')
    table_stream.add_subscriber(update_selected_indices)

    # --- Spectrum & OD spectrum ---
    def plot_spectrum(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
        indices = (selected_indices if selected_indices and len(selected_indices) > 0 else [])
        overlays = []
        color_cycle = ['black', 'red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']
        if indices:
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
                co2_wls = np.array(row['CO2_WLs'])
                co2_fluxes = np.array(row['CO2_Fluxes'])
                co2_baseline = np.array(row['CO2_Baseline'])
                co2_curve = hv.Curve((co2_wls, co2_fluxes), 'Wavelength (μm)', 'CO2 Flux (mJy)').opts(
                    color=color, line_width=0.75
                )
                baseline_co2_curve = hv.Curve((co2_wls, co2_baseline)).opts(color='purple', line_dash='dashed', alpha=0.7)
                co_wls = np.array(row['CO_WLs'])
                co_fluxes = np.array(row['CO_Fluxes'])
                co_baseline = np.array(row['CO_Baseline'])
                co_curve = hv.Curve((co_wls, co_fluxes), 'Wavelength (μm)', 'CO Flux (mJy)').opts(
                    color=color, line_width=0.75
                )
                baseline_co_curve = hv.Curve((co_wls, co_baseline)).opts(color='green', line_dash='dashed', alpha=0.7)
                overlays.append(curve * baseline_curve * co2_curve * baseline_co2_curve * co_curve * baseline_co_curve)
        else:
            overlays = [hv.Curve([], 'Wavelength (μm)', 'Flux').opts(title="No selection") * hv.Curve([], 'Wavelength (μm)', 'Flux (mJy)')]
        return hv.Overlay(overlays).opts(width=600, height=200, xlim=(2.4, 5.1), ylim=(1e-3, 0.7), logy=True)

    def plot_od_spectrum(selected_indices, *args, **kwargs):
        cat = get_cat_with_pixels()
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
                h2o_mask = (h2o_wls >= 2.715) & (h2o_wls <= 3.35) & (h2o_od > 0)
                if np.any(h2o_mask):
                    fill_between = hv.Area((h2o_wls[h2o_mask], h2o_od[h2o_mask])).opts(
                        color='lightblue', alpha=0.4, line_alpha=0
                    )
                    h2o_od_curve = h2o_od_curve * fill_between
                co2_wls = np.array(row['CO2_WLs'])
                co2_od = np.array(row['CO2_OD_spec'])
                co2_od_curve = hv.Curve((co2_wls, co2_od), 'Wavelength (μm)', 'Optical Depth').opts(color=color, alpha=0.75, line_width=0.75)
                co2_mask = (co2_wls >= 4.2) & (co2_wls <= 4.34) & (co2_od > 0)
                if np.any(co2_mask):
                    fill_between = hv.Area((co2_wls[co2_mask], co2_od[co2_mask])).opts(
                        color='purple', alpha=0.4, line_alpha=0
                    )
                    co2_od_curve = co2_od_curve * fill_between
                co_wls = np.array(row['CO_WLs'])
                co_od = np.array(row['CO_OD_spec'])
                co_od_curve = hv.Curve((co_wls, co_od), 'Wavelength (μm)', 'Optical Depth').opts(color=color, alpha=0.75, line_width=0.75)
                co_mask = (co_wls >= 4.65) & (co_wls <= 4.705) & (co_od > 0)
                if np.any(co_mask):
                    fill_between = hv.Area((co_wls[co_mask], co_od[co_mask])).opts(
                        color='green', alpha=0.4, line_alpha=0
                    )
                    co_od_curve = co_od_curve * fill_between
                overlays.append(h2o_od_curve * co2_od_curve * co_od_curve)
        else:
            overlays = [hv.Curve([], 'Wavelength (μm)', 'Optical Depth').opts(title="No selection")]
        return hv.Overlay(overlays).opts(width=600, height=200, xlim=(2.4, 5.1), ylim=(-0.2, 5))

    spectrum_map   = hv.DynamicMap(plot_spectrum, streams=[selected_indices])
    od_spectrum_map = hv.DynamicMap(plot_od_spectrum, streams=[selected_indices])

    # --- Layout ---
    app_bar = pn.Row(
        pn.pane.Markdown('## <span style="color:white">ice Mapping interface (iMi)</span>', width=1000, sizing_mode="fixed", margin=(10,5,10,15)), 
        pn.Spacer(),
        pn.pane.PNG("http://holoviews.org/_static/logo.png", height=50, width=50, sizing_mode="fixed", align="center"),
        pn.pane.PNG("https://panel.holoviz.org/_static/logo_horizontal.png", height=50, width=100, sizing_mode="fixed", align="center"), 
        styles={'background': 'black'},
    )

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
        # active=[0, 1],  # Show both panels open on app start -  Doesn't seem to work.
    )
    
    layout = (get_img() * points * labels)

    instruction_text = pn.pane.Markdown("""
    **Instructions:**  
    - Use the search bar to find a spectrum by ID.
    - Use the toggle to show/hide labels.
    - Click on a point to view its spectrum and data.
    - Use the plots and table on the right for detailed analysis.
    """, width=1200)  # Adjust width as needed

    app = pn.Column(
        app_bar,
        instruction_text,
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
        css_classes=['imi-dashboard']
    )
    return app

app = make_app()
app.servable()
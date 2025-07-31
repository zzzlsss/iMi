import numpy as np
import pandas as pd
import re
import astropy.io.fits as fits
import astropy.visualization as apvis
from astropy.wcs import WCS
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')

# --- Data Classes ---
class RenderLatexLabels(hv.streams.Stream):
    do_render_latex_labels = hv.param.Boolean(default=False)
    def make_latex_label_string(self, latex_like_string):
        if not self.do_render_latex_labels:
            return latex_like_string
        latex_label_string = re.sub(r"([^\^_0-9]+)", r"\\text{{\1}}", latex_like_string.replace("$", ""))
        return f"$${latex_label_string}$$"

class SelectedIndices(hv.streams.Stream):
    selected_indices = hv.param.List(default=[])

class ShowLabelsStream(hv.streams.Stream):
    show_labels = hv.param.Boolean(default=True)

# --- Main Plotting Class ---
class IceMappingDashboard:
    def __init__(self, cat, img_data, wcs):
        self.cat = cat
        self.img_data = img_data
        self.wcs = wcs

        # Streams
        self.rll = RenderLatexLabels(do_render_latex_labels=False)
        self.selected_indices = SelectedIndices(selected_indices=[])
        self.show_labels_stream = ShowLabelsStream(show_labels=True)

        # Widgets
        self.label_toggle = pn.widgets.Toggle(name='Show/Hide ID Labels', value=True)
        self.search_bar = pn.widgets.TextInput(name='Search by ID', placeholder='Enter Spectrum ID(s)')

        # Set up stream links
        self.label_toggle.param.watch(self.update_show_labels, 'value')
        self.search_bar.param.watch(self.search_selected_indices, 'value')

        # Plots
        self.img = self.make_image()
        self.points = hv.DynamicMap(self.plot_points, streams=[self.selected_indices])
        self.labels = hv.DynamicMap(self.plot_labels, streams=[self.selected_indices, self.show_labels_stream])

    # --- Utility Methods ---
    def update_show_labels(self, event):
        self.show_labels_stream.event(show_labels=event.new)

    def search_selected_indices(self, event):
        search_id = event.new.strip()
        if search_id:
            try:
                search_ids = [int(s.strip()) for s in search_id.split(',') if s.strip().isdigit()]
                indices = self.cat[self.cat['ID'].isin(search_ids)].index.tolist()
                self.selected_indices.event(selected_indices=indices)
            except ValueError:
                self.selected_indices.event(selected_indices=[])
        else:
            self.selected_indices.event(selected_indices=[])

    # --- Plotting Methods ---
    def make_image(self):
        norm = apvis.ImageNormalize(self.img_data, stretch=apvis.HistEqStretch(self.img_data), clip=True)
        return rasterize(
            hv.Image(self.img_data.astype(np.float32), bounds=(0, 0, self.img_data.shape[1], self.img_data.shape[0])).opts(cnorm='eq_hist'),
            precompute=True,
        ).opts(colorbar=True, cmap='gist_heat', width=600, height=600)

    def plot_points(self, selected_indices, *args, **kwargs):
        # Conditional logic for selected/nonselected
        sel = selected_indices if selected_indices else []
        opts_dict = dict(
            marker='square', size=6, color='blue', fill_color=None,
            tools=['tap', 'lasso_select', 'hover'],
            hover_tooltips=[('ID', '@ID'), ('RA', '@H2O_RA'), ('Dec', '@H2O_Dec')],
            selected=sel, selection_color='green', selection_alpha=1, nonselection_alpha=0.4,
        )
        return hv.Points(self.cat, kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(**opts_dict)

    def plot_labels(self, selected_indices, show_labels, *args, **kwargs):
        if selected_indices:
            df = self.cat.iloc[selected_indices]
        elif show_labels:
            df = self.cat
        else:
            df = pd.DataFrame(columns=['x_pix', 'y_pix', 'ID'])
        return hv.Labels(df, kdims=['x_pix', 'y_pix'], vdims=['ID']).opts(text_color='blue', text_font_size='11pt', yoffset=15)

    def plot_correlation(self, x, y, color, selected_indices, xlabel, ylabel, title, hover_fields):
        indices = selected_indices if selected_indices else []
        opts_dict = dict(
            width=400, height=400, color=color, size=6, marker='circle',
            alpha=0.7 if indices else 0.4,
            tools=['hover', 'tap', 'lasso_select'],
            xlabel=self.rll.make_latex_label_string(xlabel),
            ylabel=self.rll.make_latex_label_string(ylabel),
            title=self.rll.make_latex_label_string(title),
            hover_fields=[('ID', '@ID'), ('H$_2$O N', '@sci_H2O_N'), ('CO N', '@sci_CO_N')] if hover_fields is None else hover_fields,
            hover_tooltips=hover_fields,
            selected=indices, nonselection_alpha=0.1 if indices else 0.4,
        )
        return hv.Points(self.cat, kdims=[x, y], vdims=['ID']).opts(**opts_dict)

    # Add more plotting methods as needed...
    def plot_spectrum(self, selected_indices, *args, **kwargs):
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        overlays = []
        color_cycle = ['black', 'red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']
        if indices:
            for num, i in enumerate(indices):
                row = self.cat.iloc[i]
                title = f"Spectrum ID: {row['ID']}" if len(indices) == 1 else f"Spectrum IDs {', '.join(str(self.cat.iloc[j]['ID']) for j in indices)}"
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

        return hv.Overlay(overlays).opts(width=600, height=300, xlim=(2.4, 5.1), ylim=(1e-3, 0.7), logy=True)

    def plot_od_spectrum(self, selected_indices, *args, **kwargs):
        indices = selected_indices if selected_indices and len(selected_indices) > 0 else []
        overlays = []
        color_cycle = ['black', 'red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']
        if indices:
            for num, i in enumerate(indices):
                row = self.cat.iloc[i]
                title = f"OD Spectrum ID {row['ID']}" if len(indices) == 1 else f"OD Spectrum IDs {', '.join(str(self.cat.iloc[j]['ID']) for j in indices)}"
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

        return hv.Overlay(overlays).opts(
            width=600, height=300, xlim=(2.4, 5.1), ylim=(-0.2, 5),
        )

# --- Usage Example ---
if __name__ == "__main__":
    # Load your data as before...
    # Replace the following paths with your actual data file paths
    data = fits.open('/Volumes/ZLS HD/PhD_Documents/Astro_Projects/Ice_Proposals/IceAge_ERS/Spectral_Extraction_Code/Real_Data_Code/FW_Files/IceAge_CHAMMS1-C2-FIELD_lw_F410M_visitall_modall_i2d.fits')
    cat = pd.read_pickle('/Users/zaklukasmith/Documents/IceMapping1/Ice_N_values_DFs/G95_All_Ice_Map.pkl')
    img_data = np.flipud(data[1].data)
    wcs = WCS(data[1].header)

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

    app_bar = pn.Row(
    pn.pane.Markdown('## <span style="color:white">ice Mapping interface (iMi)</span>', width=1000, sizing_mode="fixed", margin=(10,5,10,15)), 
    pn.Spacer(),
    pn.pane.PNG("http://holoviews.org/_static/logo.png", height=50, width=50, sizing_mode="fixed", align="center"),
    pn.pane.PNG("https://panel.holoviz.org/_static/logo_horizontal.png", height=50, width=100, sizing_mode="fixed", align="center"), 
    styles={'background': 'black'},
    )

    accord=pn.Accordion(
            ("H2 vs Ice Column Density Correlation Plots", pn.Row(
                dashboard.plot_correlation('H2O_N', 'H2_N', 'blue', dashboard.selected_indices.selected_indices),
                dashboard.plot_correlation('CO2_N', 'H2_N', 'purple', dashboard.selected_indices.selected_indices),
                dashboard.plot_correlation('CO_N', 'H2_N', 'green', dashboard.selected_indices.selected_indices),
            )),
            ("Ice Column Density Ratio Plots", pn.Row(
                dashboard.plot_correlation('H2O_N', 'CO2_N', 'blue', dashboard.selected_indices.selected_indices), 
                dashboard.plot_correlation('H2O_N', 'CO_N', 'blue', dashboard.selected_indices.selected_indices), 
                dashboard.plot_correlation('CO2_N', 'CO_N', 'blue', dashboard.selected_indices.selected_indices), 
            )),
            sizing_mode='fixed',
            width=1200,
            height=400,
        )


    dashboard = IceMappingDashboard(cat, img_data, wcs)
    app = pn.Column(
        pn.Row(dashboard.label_toggle, dashboard.search_bar),
        (dashboard.img * dashboard.points * dashboard.labels),
        # Add more plots and widgets as needed...
    )

    app = pn.Column(
            app_bar,
            pn.Spacer(height=10),
            label_toggle, search_bar,
            pn.Row(
                layout, 
                pn.Column(
                    spectrum_map,
                    od_spectrum_map,
                    ),
            ),
            table,
            accord,
            sizing_mode='stretch_both',
            # margin=(10, 10, 10, 10),
            css_classes=['imi-dashboard']
        )

    pn.serve(app, show=True, title= 'iMi - ice Mapping interface')

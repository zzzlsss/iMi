import holoviews as hv
import astropy.io.fits as fits
import astropy.visualization as apvis
from astropy.wcs import WCS
import pandas as pd
import warnings
import numpy as np
import panel as pn

hv.extension('bokeh', 'matplotlib')
from holoviews.operation.datashader import rasterize
from holoviews import opts
from holoviews.streams import Selection1D
from bokeh.models import HoverTool, NumeralTickFormatter
from holoviews.selection import link_selections
# import bokeh.models.FuncTickFormatter

# from scipy.interpolate import griddata
data = fits.open('/Volumes/ZLS HD/PhD_Documents/Astro_Projects/Ice_Proposals/IceAge_ERS/Spectral_Extraction_Code/Real_Data_Code/FW_Files/IceAge_CHAMMS1-C2-FIELD_lw_F410M_visitall_modall_i2d.fits')
cat = pd.read_pickle("/Users/zaklukasmith/Documents/IceMapping1/Ice_N_values_DFs/G95_All_Ice_Map.pkl")



wcs = WCS(data[1].header)
pixels = wcs.world_to_pixel_values(cat['H2O_RA'].values, cat['H2O_Dec'].values)
cat['x_pix'], cat['y_pix'] = pixels[0], pixels[1]
cat['ID'] = cat.index

# cat = cat[['ID', 'x_pix', 'y_pix', 'H2O_RA', 'H2O_Dec', 'H2O_N', 'H2O_N_err_lower', 'H2O_N_err_upper', 'H2O_WLs',]]# 'H2O_Fluxes', 'H2O_FluxErrs','H2O_Baseline', 'H2O_Baseline_err', 'H2O_OD_spec', 'H2O_OD_spec_err']]

## Cannot find way to label hover points with scientific notation therefore...
cat['H2O_N_sci'] = cat['H2O_N'].apply(lambda x: f"{x:.3e}")
cat['CO2_N_sci'] = cat['CO2_N'].apply(lambda x: f"{x:.3e}")
cat['CO_N_sci'] = cat['CO_N'].apply(lambda x: f"{x:.3e}")
cat['H2_N_sci'] = cat['H2_N'].apply(lambda x: f"{x:.3e}")

img_data = np.flipud(data[1].data)
norm = apvis.ImageNormalize(img_data, stretch=apvis.HistEqStretch(img_data), clip=True)

img = rasterize(
    hv.Image(img_data.astype(np.float32),bounds=(0, 0, data[1].header['NAXIS1'], data[1].header['NAXIS2'])).opts(cnorm='eq_hist',),#clim=(norm.vmin, norm.vmax)),
    precompute=True,
).opts(colorbar=True, cmap='gist_heat', width=800, height=800)

points = hv.Points(cat, kdims=['x_pix', 'y_pix'],vdims=['ID','H2O_RA','H2O_Dec',
                                                        'H2O_N_sci','H2O_N_err_lower','H2O_N_err_upper',
                                                        'CO2_N_sci','CO2_N_err_lower','CO2_N_err_upper',
                                                        'CO_N_sci','CO_N_err_lower','CO_N_err_upper',
                                                        'H2_N_sci']).opts(
    marker='square', size=6, color='blue', fill_color=None,
    tools=['tap','lasso_select', 'hover'],
    selection_color='green', selection_alpha=1,
    nonselection_alpha=0.7,
    hover_tooltips=[('ID', '@ID'),
                    ('RA', '@H2O_RA'), 
                    ('Dec', '@H2O_Dec'), 
                    ('N H2O', '@H2O_N_sci'),  # scientific notation
                    # ('H2O_N_err_lower', '@H2O_N_err_lower'), 
                    # ('H2O_N_err_upper', '@H2O_N_err_upper'), 
                    ('N CO2', '@CO2_N_sci'), 
                    # ('CO2_N_err_lower', '@CO2_N_err_lower'), 
                    # ('CO2_N_err_upper', '@CO2_N_err_upper')
                    ('N CO', '@CO_N_sci'), 
                    # ('CO_N_err_lower', '@CO_N_err_lower'), 
                    # ('CO_N_err_upper', '@CO_N_err_upper')
                    ('N H2', '@H2_N_sci'),
                    ],
)

## Create a stream to capture selections from the points plot
# This will allow us to update the spectrum plot based on selected points
points_stream = hv.streams.Selection1D(source=points)



labels = hv.Labels(cat, kdims=['x_pix', 'y_pix'],vdims=['ID']).opts(text_color='blue', text_font_size='11pt', yoffset=15,)

def plot_spectrum(index):
    if index and len(index) > 0:
        overlays = []
        color_cycle = ['black','red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']

        for num, i in enumerate(index):
            row = cat.iloc[i]
            wls = np.array(row['H2O_WLs'])
            fluxes = np.array(row['H2O_Fluxes'])
            errs = np.array(row['H2O_FluxErrs'])
            baseline = np.array(row['H2O_Baseline'])

            if len(index) == 1:
                title = f"Spectrum ID: {row['ID']}"
                color = 'black'
            else:
                # If multiple selected, show all IDs in the title
                title = f"Spectrum IDs {', '.join(str(cat.iloc[j]['ID']) for j in index)}"

                # Use a color cycle for multiple spectra
                # This will cycle through the colors for each selected spectrum
                # and matches OD spectra
                color = color_cycle[num % len(color_cycle)]

                # Ensures original black spectra stays black
                if num == 0:
                    color = 'black'

            

            # Create a curve for the spectrum
            # This will plot the fluxes against the wavelengths
            # If multiple spectra are selected, each will be plotted in a different color
            # If only one spectrum is selected, it will be black
            curve = hv.Curve((wls, fluxes), 'Wavelength (μm)', 'Flux (mJy)',label=f"{row['ID']}").opts(
                xlim=(2.4, 5.1), ylim=(1e-3, 0.7), logy=True,
                line_width=0.75, color=color, title=title,
            )

            ## Error bars as a spread
            errorbars = hv.Spread((wls, fluxes, errs)).opts(
                color='blue', alpha=0.4, logy=True,
                # show_legend=True, legend_label=f"ID: {row['ID']}"
            )

            # # Plot the continuum (baseline) as a dashed line
            baseline_curve = hv.Curve((wls, baseline)).opts(
                color='red', line_dash='dashed', alpha=0.7,
            )

            overlays.append(curve * errorbars * baseline_curve)
    else:
        # If no spectra are selected, an empty curve will be returned
        # This will ensure the plot is still displayed even when no points are selected
        # and avoids errors in the plot rendering
        curve = hv.Curve([], 'Wavelength (μm)', 'Flux').opts(title="No selection",)

        # Empty error bars
        errorbars = hv.Spread([], 'Wavelength (μm)', 'Flux')

        # An empty baseline curve
        baseline_curve = hv.Curve([], 'Wavelength (μm)', 'Flux (mJy)')
        overlays = [curve * baseline_curve]

    
    return hv.Overlay(overlays).opts(
                width=400, height=300, xlim=(2.4, 5.1), ylim=(1e-3, 0.7), logy=True,
            )

def plot_od_spectrum(index):
    if index and len(index) > 0:
        overlays = []
        color_cycle = ['black','red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']

        for num,i in enumerate(index):
            row = cat.iloc[i]
            wls = np.array(row['H2O_WLs'])
            od = np.array(row['H2O_OD_spec'])
            # od_err = np.array(row['H2O_OD_spec_err'])
            # label = row['ID']
            if len(index) > 1:
                title = f"OD Spectrum IDs {', '.join(str(cat.iloc[j]['ID']) for j in index)}"
                color = color_cycle[num % len(color_cycle)]
                if num == 0:
                    color='black'
            else:
                title = f"OD Spectrum ID {row['ID']}"
                color = 'black'

            curve = hv.Curve((wls, od), 'Wavelength (μm)', 'Optical Depth').opts(color=color,title=title, alpha=0.75, line_width=0.75)
            overlays.append(curve)

            # Shows the zero continuum line
            # Plotted only for first source as if done within loop, 
            # the od spectra are not plotted after 2 sources... 
            # Seemingly cannot plot every continuum line for each source in flux
            # if num == 0:
            #     baseline_curve = hv.Curve((wls, np.zeros_like(wls)), 'Wavelength (μm)', 'Optical Depth').opts(
            #         color='red', line_dash='dashed', alpha=0.7,
            #     )
                
            #     overlays.append(baseline_curve)
    
    else:
        curve = hv.Curve([], 'Wavelength (μm)', 'Optical Depth').opts(
            title="No selection",
        )
        overlays = [curve]
        # Add flat dashed red line at od=0 for empty view
        # overlays.append(
        #     hv.Curve((np.linspace(2.4, 5.1, 10), np.zeros(10)), 'Wavelength (μm)', 'Optical Depth').opts(
        #         color='red', line_dash='dashed', alpha=0.7,
        #     )
        # )

    return hv.Overlay(overlays).opts(
                width=400, height=300, xlim=(2.4, 5.1), ylim=(-0.2, 5), 
            )

def plot_h2_vs_h2o(index):
    if index and len(index) > 0:
        selected = cat.iloc[index]
        scatter = hv.Points(
            selected,
            kdims=['H2_N', 'H2O_N'], vdims=['ID', 'H2_N_sci', 'H2O_N_sci']
        ).opts(
            color='blue', size=6, marker='circle', alpha=0.7,
            tools=['hover'],
            xlabel='H2_N', ylabel='H2O_N', title='N H2 vs. N H2O',
            hover_tooltips=[
            ('ID', '@ID'),
            ('H2_N', '@H2_N_sci'),
            ('H2O_N', '@H2O_N_sci')
            ],
        )
    else:
        scatter = hv.Points(
            cat,
            kdims=['H2_N', 'H2O_N'], vdims=['ID','H2_N_sci', 'H2O_N_sci']
        ).opts(
            color='blue', size=6, marker='circle', alpha=0.7,
            tools=['hover'],
            xlabel='H2_N', ylabel='H2O_N', title='N H2 vs. N H2O',
            hover_tooltips=[('ID', '@ID'), ('H2_N', '@H2_N_sci'), ('H2O_N', '@H2O_N_sci')],
        )
    return scatter

def plot_h2_vs_co2(index):
    if index and len(index) > 0:
        selected = cat.iloc[index]
        scatter = hv.Scatter(
            selected,
            kdims=['H2_N', 'CO2_N'], vdims=['ID','H2_N_sci', 'CO2_N_sci']
        ).opts(
            color='green', size=6, marker='circle', alpha=0.7,
            tools=['hover'],
            xlabel='H2_N', ylabel='CO2_N', title='N H2 vs. N CO2',
            hover_tooltips=[('ID', '@ID'), ('H2_N', '@H2_N_sci'), ('CO2_N', '@CO2_N_sci')],
        )
    else:
        scatter = hv.Scatter(
            cat,
            kdims=['H2_N', 'CO2_N'], vdims=['ID','H2_N_sci', 'CO2_N_sci']
        ).opts(
            color='green', size=6, marker='circle', alpha=0.4,
            tools=['hover'],
            xlabel='H2_N', ylabel='CO2_N', title='N H2 vs. N CO2',
            hover_tooltips=[('ID', '@ID'), ('H2_N', '@H2_N_sci'), ('CO2_N', '@CO2_N_sci')],
        )
    return scatter

def plot_h2_vs_co(index):
    if index and len(index) > 0:
        selected = cat.iloc[index]
        scatter = hv.Scatter(
            selected,
            kdims=['H2_N', 'CO_N'], vdims=['ID','H2_N_sci', 'CO_N_sci']
        ).opts(
            color='red', size=6, marker='circle', alpha=0.7,
            tools=['hover'],
            xlabel='H2_N', ylabel='CO_N', title='N H2 vs. N CO',
            hover_tooltips=[('ID', '@ID'), ('H2_N', '@H2_N_sci'), ('CO_N', '@CO_N_sci')],
        )
    else:
        scatter = hv.Scatter(
            cat,
            kdims=['H2_N', 'CO_N'], vdims=['ID','H2_N_sci', 'CO_N_sci']
        ).opts(
            color='red', size=6, marker='circle', alpha=0.4,
            tools=['hover'],
            xlabel='H2_N', ylabel='CO_N', title='N H2 vs. N CO',
            hover_tooltips=[('ID', '@ID'), ('H2_N', '@H2_N_sci'), ('CO_N', '@CO_N_sci')],
        )
    return scatter

scatter_H2O = hv.DynamicMap(plot_h2_vs_h2o, streams=[points_stream])
scatter_CO2 = hv.DynamicMap(plot_h2_vs_co2, streams=[points_stream])
scatter_CO = hv.DynamicMap(plot_h2_vs_co, streams=[points_stream])
# link_selections(points + scatter_H2O + scatter_CO2 + scatter_CO)


# Write function that uses the selection indices to slice points and compute stats
def selected_info(index):
    if index:
        selected = cat.iloc[index]
        return hv.Points(selected, kdims=['x_pix', 'y_pix']).opts(
            marker='square', size=6, color='green', alpha=0.7, fill_color=None
        ).relabel('Selected Points')
    else:
        return hv.Points([], kdims=['x_pix', 'y_pix']).relabel('No selection')


app_bar = pn.Row(
    pn.pane.Markdown('## <span style="color:white">ice Mapping interface (iMi)</span>', width=500, sizing_mode="fixed", margin=(10,5,10,15)), 
    pn.Spacer(),
    pn.pane.PNG("http://holoviews.org/_static/logo.png", height=50, sizing_mode="fixed", align="center"),
    pn.pane.PNG("https://panel.holoviz.org/_static/logo_horizontal.png", height=50, sizing_mode="fixed", align="center"),
    styles={'background': 'black'},
)
app_bar


# Pair the plots so that selections in one update the other and axes stay synced
layout = (img * points * labels)

spectrum_map = hv.DynamicMap(plot_spectrum, streams=[points_stream])
od_spectrum_map = hv.DynamicMap(plot_od_spectrum, streams=[points_stream])

## Not using currently
sel_map = hv.DynamicMap(selected_info, streams=[points_stream])
if __name__ == "__main__":
    app = pn.Column(
        app_bar,
        pn.Spacer(height=10),
        pn.Row(
            layout,
            pn.Column(
                spectrum_map,
                od_spectrum_map,
                # sel_map,
            ),
        ),
        pn.Row(
            scatter_H2O,
            scatter_CO2,
            scatter_CO,
        ),
    )

    # Launch Panel app in browser
    pn.serve(app, show=True, title="ice Mapping interface (iMi)")

# # Add a DataFrame widget that updates when points are selected
# table = pn.widgets.DataFrame(cat.iloc[[]], width=400, height=200, disabled=True)

# # Update the table based on selected points
# def update_table(index):
#     if index is None or len(index) == 0:
#         table.value = cat.iloc[[]]
#     else:
#         table.value = cat.iloc[index]   

# points_stream.add_subscriber(lambda: update_table(points_stream.index))

# dashboard = pn.Column(
#     "# FITS Image and Catalog Overlay",
#     plot,
#     # pn.pane.Markdown("## Selected Catalog Entries"),
#     # table
# )

# dashboard.servable()

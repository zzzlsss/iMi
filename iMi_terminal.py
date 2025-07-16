import holoviews as hv
import astropy.io.fits as fits
import astropy.visualization as apvis
from astropy.wcs import WCS
import pandas as pd
import warnings
import numpy as np



## Holoview specific imports
hv.extension('bokeh')
from bokeh.plotting import show
from holoviews.operation.datashader import rasterize
from holoviews import opts
from bokeh.plotting import show

data = fits.open('/Users/zaklukasmith/surfdrive/CHEERIO/CHEERIO_IceAge_Mosaics/ChamI-comb_F410M_i2d.fits')
cat = pd.read_pickle('/Users/zaklukasmith/surfdrive/CHEERIO/CHEERIO_IceAge_Mosaics/PostDS9Centroiding_cats/tot_IA_CH_cat_23052025_corrected.pkl')

wcs = WCS(data[1].header)
pixels = wcs.world_to_pixel_values(cat['RA'].values, cat['Dec'].values)
cat['x_pix'], cat['y_pix'] = pixels[0], pixels[1]

img_data = data[1].data

# Bizarrely, the image data is upside down in the FITS file.
# This is a workaround to flip it for correct display.
img_data = np.flipud(img_data)

norm = apvis.ImageNormalize(img_data, stretch=apvis.HistEqStretch(img_data), clip=True)

img = rasterize(
    hv.Image(img_data.astype(np.float32),bounds=(0, 0, data[1].header['NAXIS1'], data[1].header['NAXIS2'])).opts(cnorm='eq_hist',clim=(norm.vmin, norm.vmax)),  # Apply histogram equalization stretch
    precompute=True,
).opts(colorbar=True, cmap='gist_heat', width=800, height=800,)

# img 

# Create points for the catalog data
# Ensure that the catalog DataFrame has the necessary columns
if 'x_pix' not in cat or 'y_pix' not in cat:
    raise ValueError("Catalog DataFrame must contain 'x_pix' and 'y_pix' columns for plotting.")
if 'RA' not in cat or 'Dec' not in cat:
    warnings.warn("Catalog DataFrame does not contain 'RA' and 'Dec' columns, which may affect hover tooltips.")
if 'ID' not in cat:
    warnings.warn("Catalog DataFrame does not contain 'ID' column, which may affect hover tooltips.")   

points = hv.Points(cat, kdims=['x_pix', 'y_pix']).opts(marker='square', size=6, color='blue', alpha=0.7, fill_color=None, tools=['hover'],hover_tooltips=[('ID', '@ID'), ('RA', '@RA'), ('Dec', '@Dec')])

# Create labels for the points
labels = hv.Labels(cat, kdims=['x_pix', 'y_pix']).opts(text_color='blue', text_font_size='11pt', yoffset=8) #,hover_tooltips=tooltips)

# If img is not displaying, try using hv.output or show to render it explicitly
# hv.output(img * points * labels)
# Compose the plot
plot = img * points * labels

# Render to Bokeh and show in browser window
show(hv.render(plot, backend='bokeh'))


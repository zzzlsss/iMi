from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import warnings
import numpy as np
import sys

import holoviews as hv
import panel as pn

hv.extension('bokeh')
pn.extension()

class FitsCatalogLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.data = None
        self.cat = None
        self.img_data = None
        self.panel_server = None

    def init_ui(self):
        self.setWindowTitle('ice Mapping interface (iMi) - FITS and Catalog Loader')
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        info_layout = QVBoxLayout()
        self.fits_label = QLabel('No FITS file loaded')
        self.cat_label = QLabel('No catalog loaded')
        self.load_fits_btn = QPushButton('Load FITS')
        self.load_cat_btn = QPushButton('Load Catalog')
        self.display_btn = QPushButton('Display Data')
        self.display_btn.setEnabled(False)
        self.load_fits_btn.clicked.connect(self.load_fits)
        self.load_cat_btn.clicked.connect(self.load_catalog)
        self.display_btn.clicked.connect(self.display_data)
        info_layout.addWidget(self.fits_label)
        info_layout.addWidget(self.load_fits_btn)
        info_layout.addWidget(self.cat_label)
        info_layout.addWidget(self.load_cat_btn)
        info_layout.addWidget(self.display_btn)

        self.plots_layout = QHBoxLayout()
        self.web_view = QWebEngineView()
        self.plots_layout.addWidget(self.web_view)

        main_layout.addLayout(info_layout)
        main_layout.addLayout(self.plots_layout)

    def load_fits(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open FITS file', '', 'FITS files (*.fits)')
        if fname:
            self.fits_label.setText(f'Loaded FITS: {fname}')
            self.data = fits.open(fname)
            self.img_data = np.flipud(self.data[1].data)
            self.check_ready()

    def load_catalog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Catalog file', '', 'Pickle files (*.pkl)')
        if fname:
            self.cat_label.setText(f'Loaded Catalog: {fname}')
            self.cat = pd.read_pickle(fname)
            self.check_ready()

    def check_ready(self):
        if self.data is not None and self.cat is not None:
            self.display_btn.setEnabled(True)

    def display_data(self):
        # Compute pixel coordinates from RA/Dec if needed
        if 'x_pix' not in self.cat or 'y_pix' not in self.cat:
            if 'RA' in self.cat and 'Dec' in self.cat:
                wcs = WCS(self.data[1].header)
                pixels = wcs.world_to_pixel_values(self.cat['RA'].values, self.cat['Dec'].values)
                self.cat['x_pix'], self.cat['y_pix'] = pixels[0], pixels[1]
            else:
                warnings.warn("Catalog DataFrame must contain 'x_pix' and 'y_pix' columns, or 'RA' and 'Dec' columns for conversion.")
                return

        # HoloViews image
        bounds = (0, 0, self.data[1].header['NAXIS1'], self.data[1].header['NAXIS2'])
        img = hv.Image(self.img_data, bounds=bounds).opts(
            width=600, height=600, colorbar=True, cmap='inferno', title="FITS Image"
        )

        # Catalog points
        points = hv.Points(self.cat, ['x_pix', 'y_pix'], ['ID', 'RA', 'Dec']).opts(
            size=6, color='blue', alpha=0.7, tools=['hover']
        )

        # RA vs ID plot
        if 'ID' in self.cat and 'RA' in self.cat:
            ra_id = hv.Scatter(self.cat, 'ID', 'RA').opts(
                width=400, height=400, color='green', size=6, title="RA vs ID"
            )
        else:
            ra_id = hv.Text(0.5, 0.5, "Catalog missing 'ID' or 'RA'")

        layout = pn.Row(img * points, ra_id)

        # Start Panel server
        if self.panel_server is not None:
            self.panel_server.stop()
        self.panel_server = pn.serve(layout, show=False, start=True, port=5006, threaded=True)

        # Embed Panel app in Qt
        from PyQt5.QtCore import QUrl
        url = QUrl("http://localhost:5006")
        self.web_view.setUrl(url)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FitsCatalogLoader()
    window.show()
    sys.exit(app.exec_())

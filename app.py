import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ice Mapping interface (iMi)", layout="wide")
st.title("ice Mapping interface (iMi)")

# -- Load Data
@st.cache_data
def load_data():
    img_data = np.load('./IceAge_Original_Data/IA_F410M_img_data.npy')
    wcs = pd.read_pickle('./IceAge_Original_Data/IA_F410M_wcs.pkl')
    cat = pd.read_pickle('./IceAge_Original_Data/Smith2025_Data.pkl')
    return img_data, wcs, cat

img_data, wcs, cat = load_data()

# -- Prepare Data
pixels = wcs.world_to_pixel_values(cat['H2O_RA'].values, cat['H2O_Dec'].values)
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

# -- Sidebar: Source Selection
st.sidebar.header("Select Spectrum ID(s)")
selected_ids = st.sidebar.multiselect(
    "Select Spectrum ID(s) to highlight",
    options=cat['ID'].tolist(),
    default=cat['ID'].tolist()[:1]
)

# -- Rasterized Image + Sources
st.subheader("Rasterized FITS Image with Source Overlay")
fig1, ax1 = plt.subplots(figsize=(7,7))
img_show = ax1.imshow(img_data, origin='lower', cmap='gist_heat')
cbar = fig1.colorbar(img_show, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Intensity")
ax1.scatter(cat['x_pix'], cat['y_pix'], s=20, edgecolor='blue', facecolor='none', label='All Sources')
if selected_ids:
    sel = cat[cat['ID'].isin(selected_ids)]
    ax1.scatter(sel['x_pix'], sel['y_pix'], s=50, color='lime', marker='s', label='Selected Source(s)')
    # Label selected sources
    for _, row in sel.iterrows():
        ax1.text(row['x_pix']+2, row['y_pix']+2, str(row['ID']), color='lime', fontsize=9)
ax1.set_xlabel("Pixel X")
ax1.set_ylabel("Pixel Y")
ax1.set_title("Rasterized Image with Sources")
ax1.legend()
st.pyplot(fig1)

# -- Spectrum Plots for Selected Sources
st.subheader("Flux and Optical Depth Spectra for Selected Sources")
color_cycle = ['black', 'red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']

if selected_ids:
    sel = cat[cat['ID'].isin(selected_ids)]
    # Flux Plot
    fig2, ax2 = plt.subplots(figsize=(8,3))
    for i, (_, row) in enumerate(sel.iterrows()):
        color = color_cycle[i % len(color_cycle)]
        h2o_wls = np.array(row['H2O_WLs'])
        h2o_fluxes = np.array(row['H2O_Fluxes'])
        h2o_baseline = np.array(row['H2O_Baseline'])
        ax2.plot(h2o_wls, h2o_fluxes, color=color, label=f"H2O Flux (ID {row['ID']})")
        ax2.plot(h2o_wls, h2o_baseline, color=color, linestyle='dashed', alpha=0.5, label=f"H2O Baseline (ID {row['ID']})")

        # CO2 and CO, if present
        try:
            co2_wls = np.array(row['CO2_WLs'])
            co2_fluxes = np.array(row['CO2_Fluxes'])
            co2_baseline = np.array(row['CO2_Baseline'])
            ax2.plot(co2_wls, co2_fluxes, color='purple', alpha=0.7, label=f"CO2 Flux (ID {row['ID']})")
            ax2.plot(co2_wls, co2_baseline, color='purple', linestyle='dashed', alpha=0.5, label=f"CO2 Baseline (ID {row['ID']})")
        except Exception:
            pass
        try:
            co_wls = np.array(row['CO_WLs'])
            co_fluxes = np.array(row['CO_Fluxes'])
            co_baseline = np.array(row['CO_Baseline'])
            ax2.plot(co_wls, co_fluxes, color='green', alpha=0.7, label=f"CO Flux (ID {row['ID']})")
            ax2.plot(co_wls, co_baseline, color='green', linestyle='dashed', alpha=0.5, label=f"CO Baseline (ID {row['ID']})")
        except Exception:
            pass
    ax2.set_xlabel("Wavelength (μm)")
    ax2.set_ylabel("Flux (mJy)")
    ax2.set_yscale('log')
    ax2.set_xlim(2.4, 5.1)
    ax2.set_ylim(1e-3, 0.7)
    ax2.set_title("Flux Spectra")
    ax2.legend(fontsize=7, ncol=2)
    st.pyplot(fig2)

    # OD Plot
    fig3, ax3 = plt.subplots(figsize=(8,3))
    for i, (_, row) in enumerate(sel.iterrows()):
        color = color_cycle[i % len(color_cycle)]
        h2o_wls = np.array(row['H2O_WLs'])
        h2o_od = np.array(row['H2O_OD_spec'])
        ax3.plot(h2o_wls, h2o_od, color=color, label=f"H2O OD (ID {row['ID']})", alpha=0.75)
        # Fill region for H2O OD
        mask = (h2o_wls >= 2.715) & (h2o_wls <= 3.35) & (h2o_od > 0)
        if np.any(mask):
            ax3.fill_between(h2o_wls[mask], 0, h2o_od[mask], color='lightblue', alpha=0.3)

        try:
            co2_wls = np.array(row['CO2_WLs'])
            co2_od = np.array(row['CO2_OD_spec'])
            ax3.plot(co2_wls, co2_od, color='purple', alpha=0.6, label=f"CO2 OD (ID {row['ID']})")
            # Fill region for CO2 OD
            mask2 = (co2_wls >= 4.2) & (co2_wls <= 4.34) & (co2_od > 0)
            if np.any(mask2):
                ax3.fill_between(co2_wls[mask2], 0, co2_od[mask2], color='purple', alpha=0.2)
        except Exception:
            pass
        try:
            co_wls = np.array(row['CO_WLs'])
            co_od = np.array(row['CO_OD_spec'])
            ax3.plot(co_wls, co_od, color='green', alpha=0.6, label=f"CO OD (ID {row['ID']})")
            mask3 = (co_wls >= 4.65) & (co_wls <= 4.705) & (co_od > 0)
            if np.any(mask3):
                ax3.fill_between(co_wls[mask3], 0, co_od[mask3], color='green', alpha=0.2)
        except Exception:
            pass
    ax3.set_xlabel("Wavelength (μm)")
    ax3.set_ylabel("Optical Depth")
    ax3.set_xlim(2.4, 5.1)
    ax3.set_ylim(-0.2, 5)
    ax3.set_title("Optical Depth Spectra")
    ax3.legend(fontsize=7, ncol=2)
    st.pyplot(fig3)
else:
    st.info("Select at least one source to view spectra.")

# -- Correlation Plots (H2 vs H2O, etc.), with Linked Selection
st.subheader("Column Density Correlation Plots")
corr_cols = [
    ("H2_N", "H2O_N", "N H$_2$", "N H$_2$O"),
    ("H2_N", "CO2_N", "N H$_2$", "N CO$_2$"),
    ("H2_N", "CO_N", "N H$_2$", "N CO"),
    ("H2O_N", "CO2_N", "N H$_2$O", "N CO$_2$"),
    ("H2O_N", "CO_N", "N H$_2$O", "N CO"),
    ("CO_N", "CO2_N", "N CO", "N CO$_2$")
]
figs = []
for xcol, ycol, xlabel, ylabel in corr_cols:
    figc, axc = plt.subplots(figsize=(3,3))
    axc.scatter(cat[xcol], cat[ycol], color='grey', alpha=0.3, s=10)
    if selected_ids:
        sel = cat[cat['ID'].isin(selected_ids)]
        axc.scatter(sel[xcol], sel[ycol], color='crimson', s=30)
    axc.set_xlabel(xlabel)
    axc.set_ylabel(ylabel)
    axc.set_title(f"{xlabel} vs {ylabel}")
    figs.append(figc)
cols = st.columns(3)
for i, figc in enumerate(figs[:3]):
    with cols[i]:
        st.pyplot(figc)
cols2 = st.columns(3)
for i, figc in enumerate(figs[3:]):
    with cols2[i]:
        st.pyplot(figc)

# -- Table Linked to Selection
st.subheader("Source Information Table")
show_cols = [
    'ID', 'H2O_RA', 'H2O_Dec', 'sci_H2O_N', 'sci_H2O_N_err_upper', 'sci_H2O_N_err_lower',
    'sci_CO2_N', 'sci_CO2_N_err_upper', 'sci_CO2_N_err_lower',
    'sci_CO_N', 'sci_CO_N_err_upper', 'sci_CO_N_err_lower', 'sci_H2_N'
]
rename_dict = {
    'H2O_RA': 'RA', 'H2O_Dec': 'Dec', 'sci_H2O_N': 'N H2O',
    'sci_H2O_N_err_upper': 'N H2O err upp', 'sci_H2O_N_err_lower': 'N H2O err low',
    'sci_CO2_N': 'N CO2', 'sci_CO2_N_err_upper': 'N CO2 err upp',
    'sci_CO2_N_err_lower': 'N CO2 err low', 'sci_CO_N': 'N CO',
    'sci_CO_N_err_upper': 'N CO err upp', 'sci_CO_N_err_lower': 'N CO err low',
    'sci_H2_N': 'N H2'
}
table_df = cat[show_cols].rename(columns=rename_dict)
if selected_ids:
    st.dataframe(table_df[table_df['ID'].isin(selected_ids)].reset_index(drop=True))
else:
    st.dataframe(table_df)

st.markdown("""
---
*This advanced Streamlit app mimics the linked selection and plotting of your original dashboard. For even more interactivity (e.g., lasso/tap/hovers), consider Streamlit custom components or Bokeh server deployments.*
""")
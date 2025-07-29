# ice Mapping interface (iMi) 

Welcome fellow ice chasers!

Here at iMi inc. we are driven to provide first class wide-scale cloud ice map data visualisation for you and your datasets! For us to be able to do this, there are some small data formatting and python package installation requirements from you, the user.

> [!IMPORTANT]
> The required packages (and versions) are:
> ```pandas```
> ```numpy```
> ```holoviews```


> [!CAUTION]
> Without these formats, iMi cannot unlock your data's full potential. :-1:

Please ensure you are...
1. Using image data as a __FITS__ file (**Decision required:** or 2D array .npy with a WCS information pickle file)
2. The analysis information is stored as a  ```pandas.DataFrame``` with these columns (specifically formatted like so - case-sensitive names but not column-order-sensitive):
- ID
- RA
- Dec
- H2O_N
- CO2_N
- CO_N
- H2_N
- H2O_N_err_lower
- H2O_N_err_upper
- **repeat for other ice molecules**
- WLs
- H2O_Fluxes (reconsider what fluxes we want to plot)
- H2O_FluxErrs (reconsider what fluxes we want to plot)
- **repeat for other ice molecules**
- **info for ternary plot?**



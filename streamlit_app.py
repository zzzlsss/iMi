import streamlit as st
import holoviews as hv
import numpy as np
import pandas as pd
import streamlit_bokeh_events

hv.extension('bokeh')

# --- Dummy data for demo ---
img_data = np.random.rand(100,100)
cat = pd.DataFrame({'x': np.random.rand(20)*100, 'y': np.random.rand(20)*100})

img = hv.Image(img_data)
points = hv.Points(cat, kdims=['x', 'y'])

# --- Render to Bokeh ---
bokeh_img = hv.render(img, backend='bokeh')
bokeh_points = hv.render(points, backend='bokeh')

st.title("Demo: HoloViews Plots in Streamlit")
st.write("### Image Plot")
streamlit_bokeh_events(
    bokeh_plot=bokeh_img,
    events="MOVE,CLICK",
    key="img"
)
st.write("### Points Plot")
streamlit_bokeh_events(
    bokeh_plot=bokeh_points,
    events="MOVE,CLICK",
    key="points"
)
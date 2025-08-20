import streamlit as st
import panel as pn

# Your existing Panel app code (or import everything from app.py)
from app import app

pn.extension('bokeh', 'mathjax')

st.title("Ice Mapping Interface (iMi)")

pn.panel(app).streamlit()
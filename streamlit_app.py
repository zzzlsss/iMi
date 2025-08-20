import streamlit as st
import panel as pn

# --- Panel extension activation ---
pn.extension('bokeh', 'mathjax')

# --- Import your app from app.py ---
from app import app

# --- Streamlit app body ---
st.set_page_config(page_title="Ice Mapping Interface (iMi)", layout="wide")
st.title("Ice Mapping Interface (iMi)")

# --- Display the Panel app inside Streamlit ---
pn.panel(app).streamlit()
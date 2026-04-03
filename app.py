import streamlit as st

from src.config import load_config
from src.state import init_session_state
from src.ui.layout import render_phase3_home, render_sidebar


st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎬",
    layout="wide",
)

config = load_config()
init_session_state()

mode = render_sidebar(config)
st.session_state.active_user_mode = mode

render_phase3_home(mode, config)

import streamlit as st
from references import references, bibliography
st.title('Inflammation modelling examples')
st.markdown("""
This page contains a set of examples of how mathematical models can be used in an educational setting. To access the examples, select a page from the left sidebar. 

All examples are created using models and systems from previously published work. All used works are listed in the reference list below, as well as in the individual pages.
""")

bibliography()

st.sidebar.success("Select a page above 👆")
import streamlit as st
from references import references, bibliography

# Install sund in a custom location
import os
import subprocess
import sys
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')

st.title('Inflammation modelling examples')
st.markdown("""
This page contains a set of examples of how mathematical models can be used in an educational setting. To access the examples, select a page from the left sidebar. 

The examples starts with a short introduction, including the abstract from the paper where the model was taken from. 
Then, the example showcases the (recreated) agreement to the experimental data used in the original works. 
Finally, it ends with some simulation exercises, where you can vary different things that are relevant to the model to see the effect of such changes. 
These examples are often inspired from the simulation showcases in the original papers, but can sometimes be additional new simulations. 

All examples are created using models and systems from previously published work. All used works are listed in the reference list below, as well as in the individual pages.
""")

bibliography()

st.sidebar.success("Select a page above 👆")

import streamlit as st
from references import references, bibliography
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

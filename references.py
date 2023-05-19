import streamlit as st
references = dict()

references['nyman_2020'] = """[E. Nyman et al., “Mechanisms of a Sustained Anti-inflammatory Drug Response in Alveolar Macrophages Unraveled with Mathematical Modeling,” CPT: Pharmacometrics & Systems Pharmacology, vol. 9, no. 12, pp. 707–717, 2020, doi: 10.1002/psp4.12568.](https://doi.org/10.1002/psp4.12568)"""

def biblioghraphy(cite_keys=None, heading="References"):
    st.subheader(heading)
    if cite_keys==None:
        cite_keys = references.keys()

    for i,key in enumerate(cite_keys):
        st.markdown(f"[{i+1}] {references[key]}")
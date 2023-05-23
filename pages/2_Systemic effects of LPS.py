import sund
import pandas as pd
import numpy as np
import json
import streamlit as st
from references import references as ref, bibliography
from setup_model import setup_model
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import math
import copy



# Define functions needed

def get_sim_object(model, stim):
    act = sund.Activity(timeunit = 'h')
    pwc = sund.PIECEWISE_CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"][1:], fvalues = val["f"]) 
    
    sim = sund.Simulation(models = model, activities = act, timeunit = 'h')
    return sim

def get_all_sim_objects(model, data):
    sims = {}
    sim_steady_state = get_sim_object(model, {})
    for key, val in data.items():
        sims[key] = get_sim_object(model, val["input"])
    return sims, sim_steady_state

def simulate(sim, time = np.linspace(-0.0001, 8)):
    sim.ResetStatesDerivatives()
    sim.Simulate(timevector = time)

    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)
    return sim_results

def get_times(data):
    t_data = []
    t_input = []
    for k, var in data.items():
        if k not in ["input", "meta", "extra"]:
            if not isinstance(var["Time"], list):
                t_data.append(var["Time"])
            else:
                t_data+=var["Time"]
    t_data.sort()
    for k, stim in data["input"].items():
        if not isinstance(stim["t"], list):
            t_input.append(stim["t"])
        else:
            t_input+=stim["t"]
    t_input.sort()

    t = t_data+t_input
    t = [t for t in t if t > -math.inf]
    t = sorted(set(t))

    return t, t_data, t_input

def to_rgb(color):
    return f"rgb{tuple(color)}"

def plot_agreement(sim, data):
    figs = dict()
    times,_,_ = get_times(data)
    sim_res = simulate(sim, np.linspace(times[0],times[-1],10000))

    for observable_key in [key for key in data.keys() if key not in ["input", "meta"]]:
        figs[observable_key] = go.Figure()
        observable = data[observable_key]
        unit = observable.pop("unit",'a.u.')

        obs_res = sim_res[sim_res["Time"]<observable["Time"][-1]]
        figs[observable_key].add_trace(go.Scatter(name = observable_key, x=observable["Time"], y=observable["Mean"], error_y={"type": "data", "array":observable["SEM"]}, mode='markers', marker={"line": {"width":0}}))
        figs[observable_key].add_trace(go.Scatter(name = observable_key+"_sim", x=obs_res["Time"], y=obs_res[observable_key], showlegend=False, mode='lines', marker={"line": {"width":0}}))
        figs[observable_key].update_layout(xaxis_title="Time (hour)", yaxis_title=f"{observable_key} ({unit})", margin=dict(l=0, r=0, t=0, b=0))

    return figs

# Setup the models

# Start the app
st.title("Systemic effects of LPS")
st.markdown("""This example is based on a publication by Dobreva et al. from 2021, in which the authors modeled the effect of endotoxin challenges (LPS) on systemic inflammation, pain perception, fever and heart rate.
The original model was implemented as a delay-differential equation, but this page uses a simplified version of the model using ordinary differential equations, where the lags have been reduced to constants. 
""")

with st.expander("Abstract from the Dobreva et al. paper", expanded=True):
    st.write("""Uncontrolled, excessive production of pro-inflammatory mediators from immune cells and traumatized tissues can cause systemic inflammatory conditions such as sepsis, one of the ten leading causes of death in the USA, and one of the three leading causes of death in the intensive care unit. Understanding how inflammation affects physiological processes, including cardiovascular, thermal and pain dynamics, can improve a patient's chance of recovery after an inflammatory event caused by surgery or a severe infection. Although the effects of the autonomic response on the inflammatory system are well-known, knowledge about the reverse interaction is lacking. The present study develops a mathematical model analyzing the inflammatory system's interactions with thermal, pain and cardiovascular dynamics in response to a bacterial endotoxin challenge. We calibrate the model with individual data from an experimental study of the inflammatory and physiological responses to a one-time administration of endotoxin in 20 healthy young men and validate it against data from an independent endotoxin study. We use simulation to explore how various treatments help patients exposed to a sustained pathological input. The treatments explored include bacterial endotoxin adsorption, antipyretics and vasopressors, as well as combinations of these. Our findings suggest that the most favourable recovery outcome is achieved by a multimodal strategy, combining all three interventions to simultaneously remove endotoxin from the body and alleviate symptoms caused by the immune system as it fights the infection. """
)

st.markdown("""## The model
The model describes systemic effects of LPS stimulation. The model is illustrated in the figures below, firstly as an overview (Figure 2 from [Dobreva et al.](https://doi.org/10.1113/JP280883)) and then with more details of the immune response module ([Figure 3 from Dobreva etal.](https://doi.org/10.1113/JP280883))
""")
     
st.image('./assets/dobreva-system.png')
st.markdown("""[**Figure 2: Feedback diagram for human response to endotoxin challenge.**](https://doi.org/10.1113/JP280883) *LPS administration initiates an immune cascade, as well as a decrease in the pain perception threshold. The decrease in the PT results in an increase in BP. Pro-inflammatory cytokines act as pyrogens, increasing body temperature, whereas anti-inflammatory cytokines act as antipyrogens to decrease temperature. Pro- and anti-inflammatory cytokines have opposing effects on NO production, which decreases BP via vasodilatation (decreases vascular resistance). Temperature increases HR via decrease in basal vagal tone. Elevated BP decreases HR. When BP falls to a hypotensive range, it will act to increase HR. Temperature interacts with these changes in HR via BP.*

""")

st.image('./assets/dobreva-immune-module.png')
st.markdown("""[**Figure 3: Immune interactions in response to endotoxin challenge.**](https://doi.org/10.1113/JP280883) *Endotoxin (E) administration results in the activation of monocytes (MR→MA). Activated monocytes (MA) secrete mediators that induce further immune activation (TNF-α, IL-6 and IL-8). These pro-inflammatory mediators stimulate the production of IL-10, which regulates the immune response as an anti-inflammatory mediator. IL-6 also exhibits anti-inflammatory effects because it downregulates the synthesis of TNF-α and its own release.*
""")

st.markdown("""## The model
Below are the agreements from [Figure 6 from the paper](https://doi.org/10.1113/JP280883). In the original work, the figure contained two different (but similar) datasets from two other publications. 
You can switch between the two datasets using the data selector below. 
""")
     
data_selected = st.selectbox("**Data selection:**", ["Janum", "Copeland"], 0)
model, model_features = setup_model('fever', param_keywords = data_selected)

st.markdown("""
## Recreating the model agreement to the experimental data
We will now first recreate the model simulations and the agreement to the experimental data. 
In the graphs, vertical bars correspond to measured experimental data, and continues lines correspond to model simulations. 
The data consist of both dose-responses and time-series data. 

""")
            
show_plots = st.checkbox('Show agreements', True)

if show_plots:

    with open(f'data/data_fever.json','r') as f:
        all_data = json.load(f)
    data = all_data[data_selected]
    
    sims, sim_steady_state = get_all_sim_objects(model, all_data)
    figs = plot_agreement(sims[data_selected], data)

    for title, fig in figs.items():
        # st.markdown(f"#### {title}")
        st.plotly_chart(fig)

bibliography(["dobreva_2021"])



####### only for debugging purposes ########
# sims, sim_steady_state = get_all_sim_objects(model, data)
# experiment = st.selectbox("Select experiment", data.keys(), 7)

# times,_,_ = get_times(data[experiment])
# sim = simulate(sims[experiment], np.linspace(times[0],times[-1],10000))

# st.line_chart(sim, x="Time", y=feature)
############################################



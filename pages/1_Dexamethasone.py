import sund
import pandas as pd
import numpy as np
import json
import streamlit as st
from references import references as ref, biblioghraphy
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

def simulate(sim, time = np.linspace(0, 25)):
    sim_steady_state.ResetStatesDerivatives()
    sim_steady_state.Simulate(timevector = [0.0, 700.0])
    sim_steady_state.statevalues[sim.statenames.index('TNF')]=0
    sim.ResetStatesDerivatives()
    sim.statevalues = sim_steady_state.statevalues.copy()
    sim.Simulate(timevector = time)

    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)
    return sim_results

def get_times(data):
    t = []
    for k, var in data.items():
        if k not in ["input", "meta", "extra"]:
            if not isinstance(var["time"], list):
                t.append(var["time"])
            else:
                t+=var["time"]
    for k, stim in data["input"].items():
        if not isinstance(stim["t"], list):
            t.append(stim["t"])
        else:
            t+=stim["t"]
    t = [t for t in t if t > -math.inf]
    t = sorted(set(t))
    return t

def to_rgb(color):
    return f"rgb{tuple(color)}"

def plot_agreement(sims, data):
    figs = [go.Figure(), go.Figure(), go.Figure(), go.Figure()]
    titles=["A) LPS dose-response", "B) LPS time-response", "C) Dexamethasone + LPS", "Dexamethasone, pre-treatments + LPS"]

    dr = dict()
    for key, val in data.items():
        times = get_times(val)
        sim = simulate(sims[key], np.linspace(times[0],times[-1],10000))

        for observable_key in [key for key in data[key].keys() if key not in ["input", "meta"]]:
            observable = data[key][observable_key]

        if 'dr_' in key:
            stimuli, dose = key.split('_')[1:]
            if stimuli not in dr.keys():
                dr[stimuli] =dict(unit = data[key]["input"][stimuli]["unit"])
                sim_dr = copy.deepcopy(dr)
                dr[stimuli][observable_key] = {"dose": [dose], "mean": [observable["mean"]], "SEM": [observable["SEM"]], "unit": observable["unit"], "color":observable["color"]}
                sim_dr[stimuli][observable_key] = {"dose": [sim[stimuli].values[-1]], observable_key: [sim[observable_key].values[-1]], "unit": observable["unit"], "color":observable["color"]}
            else:
                dr[stimuli][observable_key]["dose"].append(dose) 
                dr[stimuli][observable_key]["mean"].append(observable["mean"]) 
                dr[stimuli][observable_key]["SEM"].append(observable["SEM"]) 
                sim_dr[stimuli][observable_key]["dose"].append(sim[stimuli].values[-1]) 
                sim_dr[stimuli][observable_key][observable_key].append(sim[observable_key].values[-1]) 
        else:
            unit = observable.pop("unit",'a.u.')
            figs[observable["figure"]-1].add_trace(go.Scatter(name = observable["legend"], x=observable["time"], y=observable["mean"], error_y={"type": "data", "array":observable["SEM"]}, mode='markers', marker={"line": {"width":0}, "color":to_rgb(observable["color"])}))
            figs[observable["figure"]-1].add_trace(go.Scatter(name = observable["legend"]+"_sim", x=sim["Time"], y=sim[observable_key], showlegend=False, mode='lines', marker={"line": {"width":0}, "color":to_rgb(observable["color"])}))

    for stimuli in dr.keys():
        dose_unit = dr[stimuli].pop("unit", 'a.u.')
        stimuli_name = f"{stimuli} ({dose_unit})"
        for observable_key in dr[stimuli].keys():
            observable = dr[stimuli][observable_key]
            unit = observable.pop("unit",'a.u.')
            observable_name = f"{observable_key} ({unit})"
            figs[0].add_trace(go.Scatter(name = observable_key, x=observable["dose"], y=observable["mean"], error_y={"type": "data", "array":observable["SEM"]}, showlegend=False,  mode='markers', marker={"line": {"width":0}, "color":to_rgb(observable["color"])}))
            figs[0].add_trace(go.Scatter(name = observable_key+"_sim", x=sim_dr[stimuli][observable_key]["dose"],y=sim_dr[stimuli][observable_key][observable_key], showlegend=False, mode='lines', marker={"line": {"width":0}, "color":to_rgb(sim_dr[stimuli][observable_key]["color"])}))
            figs[0].update_xaxes(title_text=stimuli_name, type="log", minor=dict(ticks="inside", ticklen=6, showgrid=True))
            figs[0].update_layout(yaxis_title=observable_name, margin=dict(l=0, r=0, t=0, b=0))

    for fig in figs[1:]:
        fig.update_layout(xaxis_title="Time (hour)", yaxis_title="TNF (pg/mg tissue)", margin=dict(l=0, r=0, t=0, b=0))

    return list(zip(figs, titles))


# Setup the models

setup_model('dexa_A')
setup_model('dexa_B')

# Start the app

st.title("Dexamethasone dynamics")
st.markdown("""This example is based on a publication by Nyman et al. from 2020, in which the authors modeled the effects of dexamethasone on alveolar macrophages, and how different dosing schedules affect the clinical window of the drug. 
This small example allows you to use a mathematical model to simulate the effect of different doses and waiting times.

## Abstract from Nyman et al.:
> Both initiation and suppression of inflammation are hallmarks of the immune response. If not balanced, the inflammation may cause extensive tissue damage, which is associated with common diseases, e.g., asthma and atherosclerosis. Anti-inflammatory drugs come with side effects that may be aggravated by high and fluctuating drug concentrations. To remedy this, an anti-inflammatory drug should have an appropriate pharmacokinetic half-life or better still, a sustained anti-inflammatory drug response. However, we still lack a quantitative mechanistic understanding of such sustained effects. Here, we study the anti-inflammatory response to a common glucocorticoid drug, dexamethasone. We find a sustained response 22 hours after drug removal. With hypothesis testing using mathematical modeling, we unravel the underlying mechanism—a slow release of dexamethasone from the receptor–drug complex. The developed model is in agreement with time-resolved training and testing data and is used to simulate hypothetical treatment schemes. This work opens up for a more knowledge-driven drug development to find sustained anti-inflammatory responses and fewer side effects.
""")

st.markdown("""## The model
Below is Figure 1 from the paper, detailing the two hypotheses tested in paper. In the paper, hypothesis A was rejected, and thus we are primarily using hypothesis B in in this example. But, you can choose to use hypothesis A.
""")
     
st.image('./assets/dexa-model.jpg')

hypothesis_selected = st.selectbox("**Hypothesis selection: **", ["A 👎", "B 👍"], 1)

model, model_features = setup_model(f'dexa_{hypothesis_selected[0]}')

st.markdown("""
## Recreating the model agreement to the experimental data
We will now first recreate the model simulations and the agreement to the experimental data. In the graphs, vertical bars correspond to measured experimental data, and continues lines correspond to model simulations. The data consist of both dose-responses and time-series data. 

First, we will recreate the model agreement to the training data from Figure 4, and then from Figure 5. 

""")
            
show_plots = st.checkbox('Show agreements', True)

if show_plots:
    st.markdown("""### Recreating agreements from Figure 4""")

    with open('data/dexa.json','r') as f:
        data = json.load(f)

    sims, sim_steady_state = get_all_sim_objects(model, data)
    figs = plot_agreement(sims, data)

    for fig, title in figs[0:3]:
        st.markdown(f"#### {title}")
        st.plotly_chart(fig)

    st.markdown("""### Recreating agreements from Figure 5""")
    fig, title = figs[3]
    st.markdown(f"#### {title}")
    st.plotly_chart(fig)

biblioghraphy(["nyman_2020"])


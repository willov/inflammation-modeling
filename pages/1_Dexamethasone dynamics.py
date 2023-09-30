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

# Install sund in a custom location
import os
import subprocess
import sys
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')

import sund

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

def simulate(sim, time = np.linspace(0, 25), skip_wash = False):
    sim_steady_state.ResetStatesDerivatives()
    sim_steady_state.Simulate(timevector = [0.0, 700.0])
    if not skip_wash:
        sim_steady_state.statevalues[sim.statenames.index('TNF')]=0
    sim.ResetStatesDerivatives()
    sim.statevalues = sim_steady_state.statevalues.copy()
    sim.Simulate(timevector = time)

    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)
    return sim_results

def get_times(data):
    t_data = []
    t_input = []
    for k, var in data.items():
        if k not in ["input", "meta", "extra"]:
            if not isinstance(var["time"], list):
                t_data.append(var["time"])
            else:
                t_data+=var["time"]
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

def plot_agreement(sims, data):
    figs = [go.Figure(), go.Figure(), go.Figure(), go.Figure()]
    titles=["A) LPS dose-response", "B) LPS time-response", "C) Dexamethasone + LPS", "Dexamethasone, pre-treatments + LPS"]

    dr = dict()
    for key, val in data.items():
        times, times_data, _ = get_times(val)
        sim = simulate(sims[key], np.linspace(times[0],times[-1],10000))
        if "hwash" in key:
            sim = sim[sim["Time"]>times_data[0]]
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
            figs[0].update_layout(yaxis_title=observable_name,
                                margin=dict(l=0, r=0, t=0, b=0))

    for fig in figs[1:]:
        fig.update_layout(xaxis_title="Time (hour)", yaxis_title="TNF (pg/mg tissue)", 
                        legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5),
                        margin=dict(l=0, r=0, t=0, b=0))

    return list(zip(figs, titles))

# Setup the models
model_A, model_A_features = setup_model('dexa_A')
model_B, model_B_features = setup_model('dexa_B')
model_treatment, model_treatment_features = setup_model(f'dexa_treatment')

# Start the app
st.title("Dexamethasone dynamics")
st.markdown("""This example is based on a publication by Nyman et al. from 2020, in which the authors modeled the effects of dexamethasone on alveolar macrophages, and how different dosing schedules affect the clinical window of the drug. 
This small example allows you to use a mathematical model to simulate the effect of different doses and waiting times.

""")

with st.expander("Abstract from the Nyman et al. paper", expanded=True):
    st.write("""Both initiation and suppression of inflammation are hallmarks of the immune response. If not balanced, the inflammation may cause extensive tissue damage, which is associated with common diseases, e.g., asthma and atherosclerosis. Anti-inflammatory drugs come with side effects that may be aggravated by high and fluctuating drug concentrations. To remedy this, an anti-inflammatory drug should have an appropriate pharmacokinetic half-life or better still, a sustained anti-inflammatory drug response. However, we still lack a quantitative mechanistic understanding of such sustained effects. Here, we study the anti-inflammatory response to a common glucocorticoid drug, dexamethasone. We find a sustained response 22 hours after drug removal. With hypothesis testing using mathematical modeling, we unravel the underlying mechanism—a slow release of dexamethasone from the receptor–drug complex. The developed model is in agreement with time-resolved training and testing data and is used to simulate hypothetical treatment schemes. This work opens up for a more knowledge-driven drug development to find sustained anti-inflammatory responses and fewer side effects."""
)

st.markdown("""## The model
Below is Figure 1 from the paper, detailing the two hypotheses tested in paper. In the paper, hypothesis A was rejected, and thus we are primarily using hypothesis B in in this example. But, you can choose to use hypothesis A.
""")
     
st.image('./assets/dexa-model.jpg')

hypothesis_selected = st.selectbox("**Hypothesis selection:**", ["A 👎", "B 👍"], 1)

# model, model_features = setup_model(f'dexa_{hypothesis_selected[0]}')
if hypothesis_selected[0] == "A":
    model, model_features = model_A, model_A_features
elif hypothesis_selected[0] == "B":
    model, model_features = model_B, model_B_features


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
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""### Recreating agreements from Figure 5""")
    fig, title = figs[3]
    st.markdown(f"#### {title}")
    st.plotly_chart(fig, use_container_width=True)



st.markdown(""" ## Simulating different doses and timings of dexamethasone
Now that we know that we have a model that is reliable, we can use it to simulate different doses and timings of dexamethasone. 
You can specify the strength of the total dexamethasone dose, how many doses it should be spread over in a day, and then when LPS will be added. Note that you can also specify to give the LPS dose before the dexamethasone dose. 
We will assume that the first dose of dexamethasone is given at t=0. 

If you want to recreate the simulations from Figure 7 in Nyman et al., set the LPS time to `-72`, to give the dose 72 hours before the fist dexamethasone dose with an LPS dose of `1`.   

In the simulation, we will wash out the residing TNF just before adding the dose of LPS, and thus you will see a drop in TNF just before LPS is given. 
We will also hide the simulation until the first dose of dexamethasone is given, but this section can optionally be shown. 

""")

st.divider()
LPS_time = st.slider("LPS time", -72.0, 50.0, -72.0, 0.1)
LPS_dose = st.number_input("LPS dose (ng/ml, in log10-scale)", 0.001, 1.0e6, 1.0, 0.1)

# Handle the dexa dosing
dexa_dose_input = st.number_input("Total dexamethasone dose (ng/ml)", 0.001, 1000.0, 0.3, 0.1)
dexa_n_doses = st.slider("Dexa dose(s)", 1, 24, 1, 1)

dexa_dose = dexa_dose_input/dexa_n_doses
dexa_doses = []
for i in range(dexa_n_doses*2):
    dexa_doses.extend([dexa_dose, 0])
dexa_timings = np.concatenate([np.arange(0,24,24/dexa_n_doses), np.arange(24,48,24/dexa_n_doses)])
dexa_timings = sorted(list(np.concatenate([dexa_timings, dexa_timings+0.25])))

# Setup the stimulation and simulate
stim = {
      "LPS": { "t": [-math.inf, LPS_time], "f": [0, LPS_dose], "unit": "ng/ml" },
      "Dexa": { "t": [-math.inf]+dexa_timings, "f": [0]+dexa_doses, "unit": "µM" },
      "wash": { "t": [-math.inf, 0], "f": [0, 1] }}
sim = get_sim_object(model_treatment, stim)
sim_steady_state = get_sim_object(model_treatment, {})
sim_results = simulate(sim, np.linspace(-73.1, 48.0,10000), skip_wash = True)

# Plot the dexa dosing
hide_presim = st.checkbox("Hide pre-simulation", True)

feature = st.selectbox("Select model feature", model_treatment_features, 0)

if hide_presim:
    sim_results = sim_results[sim_results["Time"]>=0]

st.divider()
st.line_chart(sim_results, x='Time', y="TNF")
st.line_chart(sim_results, x='Time', y="Dexa")
st.line_chart(sim_results, x='Time', y="Dexa-GR")


#### NOT WORKING FOR SOME REASON
# sim_results.rename(columns={"Time":"Time (hours)", "TNF":"TNF (a.u.)", "Dexa":"Dexa (a.u.)", "Dexa-GR":"Dexa-GR (a.u.)"}, inplace=True)
# print(sim_results.keys())
# print(sim_results["TNF (a.u.)"])
# st.divider()
# st.line_chart(sim_results, x='Time (hours)', y="TNF (a.u.)")
# st.line_chart(sim_results, x='Time (hours)', y="Dexa (a.u.)")
# st.line_chart(sim_results, x='Time (hours)', y="Dexa-GR (a.u.)")
#############


bibliography(["nyman_2020"])



####### only for debugging purposes ########
# sims, sim_steady_state = get_all_sim_objects(model, data)
# experiment = st.selectbox("Select experiment", data.keys(), 7)

# times,_,_ = get_times(data[experiment])
# sim = simulate(sims[experiment], np.linspace(times[0],times[-1],10000))

# st.line_chart(sim, x="Time", y=feature)
############################################



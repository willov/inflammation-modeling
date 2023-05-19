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
from plotly.subplots import make_subplots
from collections import defaultdict

# Setup the models

model, model_features = setup_model('dexa')

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
    for key, val in data.items():
        sims[key] = get_sim_object(model, val["input"])
    return sims

def simulate(model, stim, extra_time = 10):
    sim = get_sim_object(model, stim)
    sim.ResetStatesDerivatives()
    sim.Simulate(timevector = [0.0, 750.0])
    sim.statevalues[sim.states.index('TNF')]=0
    sim.Simulate(timevector = np.linspace(0, 1000))
    
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)
    return sim_results

def plot_agreement(sims, data):
    figs = [go.Figure(), go.Figure(), go.Figure()]
    titles=["A) LPS dose-response", "B) LPS time-response", "C) Dexamethasone + LPS"]

    dr = dict()
    for key in data.keys():
        for observable_key in [key for key in data[key].keys() if key not in ["input", "meta"]]:
            observable = data[key][observable_key]
            # ax.errorbar(data_unit["time"], data_unit["mean"], data_unit["SEM"], capsize=12)

        if 'dr_' in key:
            stimuli, dose = key.split('_')[1:]
            if stimuli not in dr.keys():
                print(stimuli)
                dr[stimuli] =dict()
                dr[stimuli][observable_key] = {"dose": [dose], "mean": [observable["mean"]], "SEM": [observable["SEM"]]}
            else:
                dr[stimuli][observable_key]["dose"].append(dose) 
                dr[stimuli][observable_key]["mean"].append(observable["mean"]) 
                dr[stimuli][observable_key]["SEM"].append(observable["SEM"]) 

        else:
            figs[observable["subplot"]-1].add_trace(go.Scatter(name = observable["legend"], x=observable["time"], y=observable["mean"], error_y={"type": "data", "array":observable["SEM"]}, mode='markers', marker={"line": {"width":0}}))

    for key in dr.keys():
        for observable_key in dr[key].keys():
            observable = dr[key][observable_key]
            figs[0].add_trace(go.Scatter(name = key, x=observable["dose"], y=observable["mean"], error_y={"type": "data", "array":observable["SEM"]}, showlegend=True, mode='markers', marker={"line": {"width":0}}))
        figs[0].update_xaxes(type="log")
    for idx, fig in enumerate(figs):
        fig.update_layout(title={'text': titles[idx]})

    return figs, titles


# Start the app

st.title("Dexamethasone dynamics")
st.markdown("""This example is based on a publication by Nyman et al. from 2020, in which the authors modeled the effects of dexamethasone on alveolar macrophages, and how different dosing schedules affect the clinical window of the drug. 
This small example allows you to use a mathematical model to simulate the effect of different doses and waiting times.

### Abstract from the published paper
Both initiation and suppression of inflammation are hallmarks of the immune response. If not balanced, the inflammation may cause extensive tissue damage, which is associated with common diseases, e.g., asthma and atherosclerosis. Anti-inflammatory drugs come with side effects that may be aggravated by high and fluctuating drug concentrations. To remedy this, an anti-inflammatory drug should have an appropriate pharmacokinetic half-life or better still, a sustained anti-inflammatory drug response. However, we still lack a quantitative mechanistic understanding of such sustained effects. Here, we study the anti-inflammatory response to a common glucocorticoid drug, dexamethasone. We find a sustained response 22 hours after drug removal. With hypothesis testing using mathematical modeling, we unravel the underlying mechanism—a slow release of dexamethasone from the receptor–drug complex. The developed model is in agreement with time-resolved training and testing data and is used to simulate hypothetical treatment schemes. This work opens up for a more knowledge-driven drug development to find sustained anti-inflammatory responses and fewer side effects.

### The model
Below is Figure 1 from the paper, detailing the two hypotheses tested in paper. In the paper, hypothesis A was rejected, and thus we are only using hypothesis B in in this example.
""")
     
st.image('./assets/dexa-model.jpg')

st.subheader('Recreating the model agreement to the experimental data')
st.markdown("""We will now first recreate the model simulations and the agreement to the experimental data. First, we will recreate the model agreement to the training data from Figure 4.""")


with open('data/dexa.json','r') as f:
    data = json.load(f)

sims = get_all_sim_objects(model, data)
(figs, titles) = plot_agreement(sims, data)

for fig in figs:
    st.plotly_chart(fig)


#Leftovers from the alcohol model
# st.subheader("Specifying the alcoholic drinks")

# n_drinks = st.slider("Number of drinks:", 1, 15, 1)

# drink_times.append(st.number_input("Time of drink (h): ", 0.0, 100.0, start_time, 0.1, key=f"drink_time{i}"))
# drink_lengths.append(st.number_input("Drink length (min): ", 0.0, 240.0, 20.0, 0.1, key=f"drink_length{i}"))
# drink_concentrations.append(st.number_input("Concentration of drink (%): ", 0.0, 100.0, 5.0, 0.01, key=f"drink_concentrations{i}"))
# drink_volumes.append(st.number_input("Volume of drink (L): ", 0.0, 24.0, 0.33, 0.1, key=f"drink_volumes{i}"))
# drink_kcals.append(st.number_input("Kcal of the drink (kcal): ", 0.0, 1000.0, 250.0, 1.0, key=f"drink_kcals{i}"))
# start_time += 1
# st.divider()

# EtOH_conc = [0]+[c*on for c in drink_concentrations for on in [1 , 0]]
# vol_drink_per_time = [0]+[v/t*on if t>0 else 0 for v,t in zip(drink_volumes, drink_lengths) for on in [1 , 0]]
# kcal_liquid_per_vol = [0]+[k/v*on if v>0 else 0 for v,k in zip(drink_volumes, drink_kcals) for on in [1 , 0]]
# drink_length = [0]+[t*on for t in drink_lengths for on in [1 , 0]]
# t = [t+(l/60)*on for t,l in zip(drink_times, drink_lengths) for on in [0,1]]

# stim = {
#     "EtOH_conc": {"t": t, "f": EtOH_conc},
#     "vol_drink_per_time": {"t": t, "f": vol_drink_per_time},
#     "kcal_liquid_per_vol": {"t": t, "f": kcal_liquid_per_vol},
#     "drink_length": {"t": t, "f": drink_length},
#     }

# # Plotting the drinks

# sim_results = simulate(model, anthropometrics, stim, extra_time=extra_time)

# st.subheader("Plotting the time course given the alcoholic drinks specified")
# feature = st.selectbox("Feature of the model to plot", model_features)
# st.line_chart(sim_results, x="Time", y=feature)


biblioghraphy(["nyman_2020"])


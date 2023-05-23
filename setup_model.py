import sund
import os
import json



def setup_model(model_name, param_keywords="("):
    sund.installModel(f"./models/{model_name}.txt")
    model_class = sund.importModel(model_name)
    model = model_class() 

    fs = []
    for path, subdirs, files in os.walk('./parameter sets'):
        for name in files:
            if model_name in name.split('(')[0] and param_keywords in name and "ignore" not in path:
                fs.append(os.path.join(path, name))
    fs.sort()
    with open(fs[0],'r') as f:
        param_in = json.load(f)

    model.parametervalues = param_in['x']

    if "ic" in param_in.keys():
        model.statevalues = param_in["ic"]

    features = model.featurenames
    return model, features
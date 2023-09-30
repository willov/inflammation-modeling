import os
import json

# Install sund in a custom location
import os
import subprocess
import sys
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')

import sund


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
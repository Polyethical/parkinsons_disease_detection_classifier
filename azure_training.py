import os
import pandas as pd
from azureml.exceptions import ComputeTargetException
from source_code import classifier
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core import Workspace
ws = Workspace.get(name='myProject',
               subscription_id='##########',
               resource_group='ml_projects'
               )

ws.write_config(path="D:\Machine Learning Models\Machine_Learning_Projects\parkinson's_disease_detection_project", file_name="config.json")

import joblib
os.makedirs("outputs", exist_ok=True)
joblib.dump(value=classifier, filename="outputs/pddm_cl.pkl")

from azureml.core.model import Model
model = Model.register(workspace=ws, model_path="./outputs/pddm_cl.pkl", model_name="parkinsons_disease_classifier")



# Create compute target
cpu_cluster_name = "cpu-cluster"

# Verify that the cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4,
                                                           idle_seconds_before_scaledown=2400)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)


# Select Compute Target
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import AmlCompute
list_vms = AmlCompute.supported_vmsizes(workspace=ws)
print(list_vms)
compute_config = RunConfiguration()
compute_config.target = "cpu-cluster"
compute_config.amlcompute.vm_size = "Standard_AI_v2"

# Add libraries to run model in cloud
from azureml.core.conda_dependencies import CondaDependencies
dependencies = CondaDependencies()
dependencies.set_python_version("3.6.6")


# # dependencies.set_pip_requirements(["numpy", "scikit-learn", "pandas", "matplotlib", "xgboost"])
dependencies.add_pip_package("scikit-learn==0.20.3")
dependencies.add_pip_package("numpy==1.16.0")
dependencies.add_pip_package("pandas==1.1.5")
dependencies.add_pip_package("matplotlib==3.0.0")
dependencies.add_pip_package("xgboost==1.5.0")
print("dependencies set")

compute_config.environment.python.conda_dependencies = dependencies

# Run model in the cloud
from azureml.core.experiment import Experiment
from azureml.core import ScriptRunConfig
# The script you specify here is the one you wrote to train your linear model
script_run_config = ScriptRunConfig(source_directory=".",script="source_code.py", run_config=compute_config)
experiment = Experiment(workspace=ws, name="parkinsons_disease_classifier")
run = experiment.submit(config=script_run_config)
run.wait_for_completion(show_output=True)

# print(run.get_file_names())
# run.download_file(name="outputs/pddm_cl.pkl")

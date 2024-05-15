"""
    fqi.py: File that contains the implementation of the Fitted Q-Iteration algorithm
    When running the file directly, it will train the FQI model on the selected environment
    When calling as a module, it will use the trained model to control the environment
"""

import os
import torch

def run_fqi(env, model):
    """
        Function that will run the FQI model on the environment
        
        Arguments:
            env (str): The environment to run the FQI model on
    """
    
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU...\n")
        device = torch.device("cpu")
    else:
        print("CUDA available. Running on GPU...\n")
        device = torch.device("cuda:0")
"""
    reinforce.py: File that contains the implementation of the Reinforce algorithm
    When running the file directly, it will train the FQI model on the selected environment
    When calling as a module, it will use the trained model to control the environment
"""

import os
import torch

def run_reinforce(env, model_path):
    """
        Function that will run the Reinforce model on the environment
        
        Arguments:
            env (str): The environment to run the FQI model on
            model_path (str): The path to the model to use
    """

    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU...\n")
        device = torch.device("cpu")
    else:
        print("CUDA available. Running on GPU...\n")
        device = torch.device("cuda:0")

def main():
    """
        Main function of the project that will run the models trained to control
        the different environments
    """

    pass

if __name__ == "__main__":
    main()

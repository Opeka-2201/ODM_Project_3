"""
    interface.py: Main file for the project that will run the models trained
    to control the environments using the gymnasium package (maintained
    fork of the OpenAI gym package)
"""

import os
import gymnasium as gym
import pandas as pd

from fqi import run_fqi
from reinforce import run_reinforce
from ddpg import run_ddpg

def run(env, model):
    """
        Function that will run the model on the environment

        Arguments:
            env (str): The environment to run the model on
            model (str): The model to run on the environment
    """

    print(f"Running model {model} on environment {env}...\n")
    gym_env = gym.make(env, render_mode="human")

    if not os.path.exists(os.path.join("models", "models.csv")):
        print("Correspondance file not found. Exiting...\n")
        return

    correspondance_df = pd.read_csv("models/models.csv")
    model_type = correspondance_df[correspondance_df["path"] == os.path.join(env, model)]\
                                  ["model_type"]
    if model_type.empty:
        print("Model not found in the correspondance file. Exiting...\n")
        return

    model_path = os.path.join("models", env, model)
    model_type = model_type.values[0]

    if model_type == "fqi":
        run_fqi(gym_env, model_path)
    elif model_type == "reinforce":
        run_reinforce(gym_env, model_path)
    elif model_type == "ddpg":
        run_ddpg(gym_env, model_path)
    else:
        print("Invalid model type. Exiting...\n")
        return

    gym_env.close()
    print("Model run complete. Exiting...\n")
    return

def main():
    """
        Main function of the project that will run the models trained to control
    """

    try:
        environments = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    except FileNotFoundError:
        print("Models directory not found. Are you running the file from the correct directory?")
        print("Please make sure to follow the instructions in the README file. Exiting...\n")
        return

    if not environments:
        print("No environments found. Are you running the file from the correct directory?")
        print("Please make sure to follow the instructions in the README file. Exiting...\n")
        return

    print("Hello and welcome to the main file of the project!")
    print("This file will run the models trained to control the environments\n")

    print("Here are the available environments:")
    for i, env in enumerate(environments):
        print(f"    {i+1}. {env}")

    env_choice = False
    while not env_choice:
        choice = input("\nPlease choose an environment to run (type the number): ")
        try:
            choice = int(choice)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue
        if 0 < choice <= len(environments):
            env_choice = True
            env = environments[choice-1]
        else:
            print("Invalid environment choice. Please try again.")

    models = os.listdir(f"models/{env}")
    if not models:
        print("\nNo models found for the environment.")
        print("Have you trained or downloaded any models for this environment?")
        print("Please make sure to follow the instructions in the README file. Exiting...\n")
        return

    print(f"\nHere are the available models for the environment {env}:")
    for i, model in enumerate(models):
        print(f"    {i+1}. {model}")

    model_choice = False
    while not model_choice:
        choice = input("\nPlease choose a model to run (type the number): ")
        try:
            choice = int(choice)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue
        if 0 < choice <= len(models):
            model_choice = True
            model = models[choice-1]
        else:
            print("Invalid model choice. Please try again.")

    run(env, model)

if __name__ == "__main__":
    main()

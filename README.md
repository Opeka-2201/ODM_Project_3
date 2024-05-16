# INFO8003-1: Optimal Decision Making for Complex Problems
## Project 3: Searching High-Quality Policies to Control an Unstable Physical System
### Authors: Romain LAMBERMONT, Arthur LOUIS

## Overview
This project is part of the course of Optimal Decision Making for Complex Problems at the University of Liège. The course is given by Prof. Damien Ernst with the help of his teaching assistants Arthur Louette and Bardhyl Miftari. The goal of this project is to implement different reinforcement learning algorithms to control and maintain upright a single and double inverted pendulum.

<div align="center">
    <img src="figures/inverted_pendulum.gif" width="400"/>
    <p><em>Animated inverted pendulum</em></p>
</div>

## Requirements
To ensure the reproducibility of the results, we provide a `requirements.txt` file that contains all the necessary libraries to run the code. To create a virtual environment and install the required libraries, you can run the following commands:
```bash
conda create --name <env> python=3.10
conda activate <env>
pip install -r requirements.txt
```

## Usage
To run the code, you can use the following command:
```bash
python code/interface.py
```

You will then be able to run the different models on the simple and double inverted pendulum environements.
Watch out if you run the files corresponding for the models directly, it will trigger the training of the models, overwriting the existing ones. It is for that that we recommend to use the interface and that the environment array is initalized empty.

## Contents
| Subject | Description | Link |
| --- | --- | --- |
| Code | Implementation of the different methods and algorithms | [Code](code/CODE.md) |
| Handwritten work | Statement and report of the project | [PDF](documents/PDF.md) |
| Models | Models used in the project | [Models](models/MODELS.md) |
| Figures | Figures used in the report | [Figures](figures/) |

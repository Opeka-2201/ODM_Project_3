"""
    reinforce.py: File that contains the implementation of the Reinforce algorithm
    When running the file directly, it will train the FQI model on the selected environment
    When calling as a module, it will use the trained model to control the environment
"""

# pylint: disable=too-few-public-methods, redefined-outer-name, too-many-instance-attributes

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.distributions.normal import Normal

## TRAINING CONSTANTS ##
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_PATH = "models"
ENVIRONMENTS_TRAIN = ["InvertedDoublePendulum-v4", "InvertedPendulum-v4"]
ENVIRONMENTS_RUN = ["InvertedDoublePendulum-v4", "InvertedPendulum-v4"]
LR = .001
GAMMA = .99
NB_EPOCHS = 25_000
EPOCHS_SAVE_INTERVAL = 1_000
NB_RUNS_PER_EPOCH = 10
BOUND_SIMPLE = 1
BOUND_DOUBLE = 3

class REINFORCENetwork(nn.Module):
    """
        Class that defines the Reinforce model

        Arguments:
            features (int): The number of features in the input
            dropout (float): The dropout rate to use in the model

        Attributes:
            fc1 (nn.Linear): The first fully connected layer
            fc2 (nn.Linear): The second fully connected layer
            fc3 (nn.Linear): The third fully connected layer
            fc4 (nn.Linear): The fourth fully connected layer
            fc5 (nn.Linear): The fifth fully connected layer
            fc6 (nn.Linear): The sixth fully connected layer
            activation (nn.Tanh): The activation function to use
            dropout (nn.Dropout): The dropout layer to use
    """

    def __init__(self, features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
            Forward pass of the model
        
            Arguments:
                x (torch.Tensor): The input tensor to the model
        """

        x = self.dropout(x)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.dropout(self.activation(self.fc4(x)))
        x = self.dropout(self.activation(self.fc5(x)))
        x = self.fc6(x)
        return x

def sample(mean, var):
    """
        Function that will sample an action from the normal distribution
        
        Arguments:
            mean (torch.Tensor): The mean of the normal distribution
            var (torch.Tensor): The variance of the normal distribution
        
        Returns:
            action (torch.Tensor): The sampled action
            prob (torch.Tensor): The probability of the action
    """

    var2 = var/2
    std = torch.exp(var2)

    if torch.isnan(std).any():
        std = torch.zeros_like(std)

    action = Normal(mean, std).sample()
    prob = Normal(mean, std).log_prob(action)

    return action, prob

def update_policy_model_params(probabilities, rewards, optimizer):
    """
        Function that will update the policy model parameters
        
        Arguments:
            probabilities (torch.Tensor): The probabilities of the actions
            rewards (torch.Tensor): The rewards received
            optimizer (torch.optim): The optimizer to use
    """

    loss = 0
    cumulative_discounted_reward = 0
    cumulative_discounted_rewards = []

    for reward in rewards:
        cumulative_discounted_reward = reward + GAMMA * cumulative_discounted_reward
        cumulative_discounted_rewards.append(cumulative_discounted_reward)

    cdrs_tensor = torch.tensor(cumulative_discounted_rewards, dtype=torch.float32)
    for cdr, prob in zip(cdrs_tensor, probabilities):
        loss += -cdr * prob.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_loop(gym_env, optimizer, reinforcenet, nb_epochs, save_epochs):
    """
        Function that will run the training loop of the Reinforce model

        Arguments:
            gym_env (gym.Env): The environment to train on
            optimizer (torch.optim): The optimizer to use
            reinforcenet (REINFORCENetwork): The model to train
            nb_epochs (int): The number of epochs to train
            save_epochs (int): The interval to save the model

        Returns:
            rewards (list): The rewards received during training
    """

    env_name = gym_env.unwrapped.spec.id
    rewards = []
    reinforcenet.train()

    for epoch in range(nb_epochs+1):
        state, _ = gym_env.reset()
        probabilities_epoch = torch.empty((0,), dtype=torch.float32)
        rewards_epoch = torch.empty((0,), dtype=torch.float32)
        cumulative_reward = 0

        while True:
            state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            mean_pred, var_pred = reinforcenet(state)
            action, prob = sample(mean_pred, var_pred)

            if "Double" in env_name:
                action = torch.clamp(action, -BOUND_DOUBLE, BOUND_DOUBLE)
            else:
                action = torch.clamp(action, -BOUND_SIMPLE, BOUND_SIMPLE)

            state, reward, terminal, trunc, _ = gym_env.step([action])

            if terminal or trunc:
                break

            cumulative_reward += reward
            probabilities_epoch = torch.cat((prob.unsqueeze(0), probabilities_epoch), dim=0)
            rewards_epoch = torch.cat((torch.tensor(reward, dtype=torch.float32).unsqueeze(0),\
                                       rewards_epoch), dim=0)

        print(f"Epoch: {epoch}/{nb_epochs} - Cumulative Reward: {cumulative_reward}\r", end="")
        rewards.append(cumulative_reward)
        if epoch % save_epochs == 0:
            torch.save(reinforcenet, f"{MODELS_PATH}/{env_name}/reinforce.pt")

        update_policy_model_params(probabilities_epoch, rewards_epoch, optimizer)

    return rewards

def run_reinforce(env, model_path):
    """
        Function that will run the Reinforce model on the environment

        Arguments:
            env (str): The environment to run the FQI model on
            model_path (str): The path to the model to use
    """

    env_name = env.unwrapped.spec.id
    features = env.observation_space.shape[0]

    dropout = .1 if "Double" in env_name else 0

    if env_name not in ENVIRONMENTS_RUN:
        raise ValueError("Invalid environment")

    reinforcenet = REINFORCENetwork(features, dropout).to(DEVICE)
    reinforcenet = torch.load(model_path)
    reinforcenet.eval()

    state, _ = env.reset()
    cumulative_reward = 0

    while True:
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        mean_pred, var_pred = reinforcenet(state)
        action, _ = sample(mean_pred, var_pred)

        if "Double" in env_name:
            action = torch.clamp(action, -BOUND_DOUBLE, BOUND_DOUBLE)
        else:
            action = torch.clamp(action, -BOUND_SIMPLE, BOUND_SIMPLE)

        state, reward, terminal, trunc, _ = env.step([action])
        cumulative_reward += reward

        if terminal or trunc:
            break

    print(f"Finished running Reinforce on {env_name} - Cumulative Reward: {cumulative_reward}")

if __name__ == "__main__":
    for env in ENVIRONMENTS_TRAIN:
        gym_env = gym.make(env)
        env_name = gym_env.unwrapped.spec.id
        print(f"Training Reinforce on {env_name}")
        features = gym_env.observation_space.shape[0]
        droupout = .1 if "Double" in env_name else 0

        reinforcenet = REINFORCENetwork(features, droupout).to(DEVICE)
        optimizer = optim.Adam(reinforcenet.parameters(), lr=LR)

        rewards = train_loop(gym_env, optimizer, reinforcenet, NB_EPOCHS, EPOCHS_SAVE_INTERVAL)

        plt.plot(rewards, label='Cumulative Reward')
        rolling_average = [sum(rewards[i-100:i])/100 for i in range(100, len(rewards))]
        plt.plot(rolling_average, label='Rolling Average')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title(f'{env_name} - Reinforce')
        plt.legend()
        plt.savefig(f"figures/{env_name}_reinforce.png")
        plt.close()

        print(f"Finished training Reinforce on {env_name}\n")

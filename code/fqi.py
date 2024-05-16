"""
    fqi.py: File that contains the implementation of the Fitted Q-Iteration algorithm
    When running the file directly, it will train the FQI model on the selected environment
    When calling as a module, it will use the trained model to control the environment
"""

# pylint: disable=too-few-public-methods, redefined-outer-name

import os
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

## TRAINING CONSTANTS ##
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_PATH = "models"
ENVIRONMENTS = [] # Comment this line and uncomment the next one to trigger training
#ENVIRONMENTS = ["InvertedDoublePendulum-v4", "InvertedPendulum-v4"]
SEED = 123
LR = 0.001
GAMMA = 0.99
NB_EPOCH = 50
EPOCHS_SAVE_INTERVAL = 10
BATCH_SIZE = 128
NB_ACTIONS = 5
ACTIONS_BOUND = 3
NB_TRANSITIONS = 200_002
ACTIONS = torch.linspace(-ACTIONS_BOUND, ACTIONS_BOUND, NB_ACTIONS).to(DEVICE)

## RUNNING CONSTANTS ##
NB_RUNS = 3
NB_RUN_STEP = 10
NB_ACTIONS_RUN_SIMPLE = 100
ACTIONS_RUN_SIMPLE = torch.linspace(-ACTIONS_BOUND, ACTIONS_BOUND, NB_ACTIONS_RUN_SIMPLE)
NB_ACTIONS_RUN_DOUBLE = 30
ACTIONS_RUN_DOUBLE = torch.linspace(-ACTIONS_BOUND, ACTIONS_BOUND, NB_ACTIONS_RUN_DOUBLE)

## CLASSES ##
class FQINetwork(nn.Module):
    """
        Class that defines the Fitted Q-Iteration network
        
        Arguments:
            features (int): The number of features in the input
        
        Attributes:
            fc1 (nn.Linear): The first fully connected layer
            fc2 (nn.Linear): The second fully connected layer
            fc3 (nn.Linear): The third fully connected layer
            fc4 (nn.Linear): The fourth fully connected layer
            activation (nn.Tanh): The activation function used
    """

    def __init__(self, features):
        super().__init__()
        self.fc1 = nn.Linear(features, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
            Function that will perform a forward pass on the network
            
            Arguments:
                x (torch.Tensor): The input to the network
            
            Returns:
                x (torch.Tensor): The output of the network
        """

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class Agent:
    """
        Class that defines the agent that will use the FQI model
        
        Arguments:
            fqinet (FQINetwork): The FQI network to use
            actions (torch.Tensor): The actions to take
            
        Attributes:
            fqinet (FQINetwork): The FQI network to use
            actions (torch.Tensor): The actions to take
    """

    def __init__(self, fqinet, actions):
        self.fqinet = fqinet
        self.actions = actions

    def choose_action(self, state):
        """
            Function that will choose the action to take based on the state
            
            Arguments:
                state (np.array): The state of the environment
        
            Returns:
                action (torch.Tensor): The action to take
        """

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            state = prepare_state_action_pairs(state, self.actions)
            action = self.actions[self.fqinet(state).argmax()]

        return [action]

def prepare_state_action_pairs(states, actions):
    """
        Function that will prepare the state-action pairs for the network
        
        Arguments:
            states (torch.Tensor): The states to prepare
            actions (torch.Tensor): The actions to prepare
        
        Returns:
            state_action_pairs (torch.Tensor): The prepared state-action pairs
    """

    nb_actions = actions.size(0)
    batch_size = states.size(0)

    states = states.unsqueeze(1).repeat(1, nb_actions, 1).view(batch_size * nb_actions, -1)
    actions = actions.repeat(batch_size, 1).view(batch_size * nb_actions, -1)

    return torch.cat((states, actions), dim=1)

def generate_osst(gym_env, nb_transitions):
    """
        Function that will generate the OSST for the environment
    
        Arguments:
            env (gym.Env): The environment to generate the OSST
            nb_transitions (int): The number of transitions to generate
        
        Returns:
            osst (np.array): The generated OSST
    """

    seed = SEED
    state, _ = gym_env.reset(seed=seed)
    features = gym_env.observation_space.shape[0] + 1
    osst = np.zeros((nb_transitions, 2 * features + 1))

    for i in tqdm(range(nb_transitions), desc="Generating OSST"):
        action = gym_env.action_space.sample()
        next_state, reward, terminal, trunc, _ = gym_env.step(action)
        osst[i] = np.hstack((state, action, reward, 1 if terminal or trunc else 0, next_state))

        if terminal or trunc:
            seed += 1
            state, _ = gym_env.reset(seed=seed)
        else:
            state = next_state

    return osst

def train_loop(gym_env, osst, actions, nb_actions, batch_size, fqinet, nb_epoch, save_epochs):
    """
        Function that will train the FQI model on the OSST
        
        Arguments:
            env (gym.Env): The environment to train the model on
            osst (np.array): The OSST to train on
            actions (torch.Tensor): The actions to take
            nb_actions (int): The number of actions
            batch_size (int): The size of the batch
            fqinet (FQINetwork): The FQI network to train
            nb_epoch (int): The number of epochs to train
            save_epochs (int): The number of epochs to save the model
            
        Returns:
            fqinet (FQINetwork): The trained FQI network
    """

    env_name = gym_env.unwrapped.spec.id
    features = gym_env.observation_space.shape[0] + 1
    osst_size = len(osst)
    x = torch.tensor(osst[:, :features], dtype=torch.float32).to(DEVICE)
    dim1 = x.shape[0]
    y = torch.tensor(osst[:, features], dtype=torch.float32).view(-1, 1).to(DEVICE)
    z = prepare_state_action_pairs(torch.tensor(osst[:, features + 2:], \
                                    dtype=torch.float32).to(DEVICE), actions)
    z = z.view(dim1, features * nb_actions)

    dataloader = DataLoader(TensorDataset(x, y, z), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(fqinet.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    fqinet.train()

    for batch_x, batch_y, _ in dataloader:
        optimizer.zero_grad()
        prediction = fqinet(batch_x)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Training FQI with {osst_size} transitions...\r", end="")
    for epoch in range(nb_epoch+1):
        with torch.no_grad():
            previous_prediction = fqinet(x)

        for batch_x, batch_y, batch_z in dataloader:
            dim2 = batch_z.shape[0]
            optimizer.zero_grad()
            batch_z_values = batch_z.view(dim2 * nb_actions, features)
            batch_z_values = fqinet(batch_z_values)
            batch_z_values = batch_z_values.view(-1, nb_actions)
            fqi_next , _ = torch.max(batch_z_values, dim=1, keepdim=True)
            fqi = batch_y + GAMMA * fqi_next

            batch_pred = fqinet(batch_x)
            batch_loss = criterion(batch_pred, fqi)
            batch_loss.backward()
            optimizer.step()

        scheduler.step(batch_loss)
        with torch.no_grad():
            post_prediction = fqinet(x)
            error = criterion(post_prediction, previous_prediction)

        if epoch % save_epochs == 0:
            print(f"Epochs: {epoch}/{nb_epoch}, Loss: {batch_loss.item()}",\
                  f"Error: {error.item()}\r", end="")
            torch.save(fqinet, f"{MODELS_PATH}/{env_name}/fqi_{osst_size}.pt")

    return fqinet


def run_fqi(gym_env, model_path):
    """
        Function that will run the FQI model on the environment
        
        Arguments:
            env (str): The environment to run the FQI model on
            model_path (str): The path to the model to use
    """

    env_name = gym_env.unwrapped.spec.id

    if env_name not in ENVIRONMENTS:
        raise ValueError("Invalid environment")

    if env_name == "InvertedDoublePendulum-v4":
        actions = ACTIONS_RUN_DOUBLE
        features = gym_env.observation_space.shape[0] + 1
        fqinet = FQINetwork(features)
    elif env_name == "InvertedPendulum-v4":
        actions = ACTIONS_RUN_SIMPLE
        features = gym_env.observation_space.shape[0] + 1
        fqinet = FQINetwork(features)
    else:
        raise ValueError("Invalid environment")

    fqinet = torch.load(model_path)
    fqinet.eval()
    agent = Agent(fqinet, actions)

    state, _ = gym_env.reset()
    cumulative_reward = 0
    while True:
        action = agent.choose_action(state)
        state, reward, terminal, truncated, _ = gym_env.step(action)
        print(f"Cumulative reward: {cumulative_reward}\r", end="")
        cumulative_reward += reward
        if terminal or truncated:
            break

    print(f"\nCumulative reward: {cumulative_reward}")

def train(gym_env):
    """
        Main function of the project that will train the FQI model on the selected environments
    """

    env_name = gym_env.unwrapped.spec.id

    print(f"Training FQI model on environment {env_name}...\n")
    features = gym_env.observation_space.shape[0] + 1

    best_cumulative_reward = -np.inf

    runs = []
    for run in range(1, NB_RUNS+1):
        torch.manual_seed(SEED+run)
        np.random.seed(SEED+run)
        gym_env.reset(seed=SEED+run)

        scores_for_osst_size = []

        for osst_size in range(1, NB_TRANSITIONS+1, int(NB_TRANSITIONS/NB_RUN_STEP)):
            osst = generate_osst(gym_env, osst_size)
            fqinet = FQINetwork(features).to(DEVICE)
            fqinet = train_loop(gym_env, osst, ACTIONS, NB_ACTIONS, BATCH_SIZE,\
                                fqinet, NB_EPOCH, EPOCHS_SAVE_INTERVAL)

            test_env = gym.make(env_name)
            test_agent = Agent(fqinet, ACTIONS)

            state, _ = test_env.reset()
            cumulative_reward = 0
            while True:
                action = test_agent.choose_action(state)
                state, reward, terminal, truncated, _ = test_env.step(action)
                cumulative_reward += reward
                if terminal or truncated:
                    break

            if cumulative_reward >= best_cumulative_reward:
                best_cumulative_reward = cumulative_reward
                torch.save(fqinet, f"{MODELS_PATH}/{env_name}/fqi.pt")

            scores_for_osst_size.append(cumulative_reward)
            print(f"{test_env.unwrapped.spec.id} - Run {run} - OSST size {osst_size}",\
                  f"- Score {cumulative_reward}\n")
            test_env.close()

        runs.append(scores_for_osst_size)

    model_dir = f"{MODELS_PATH}/{env_name}"
    for file_name in os.listdir(model_dir):
        if file_name != "fqi.pt":
            file_path = os.path.join(model_dir, file_name)
            os.remove(file_path)

    return runs

if __name__ == "__main__":
    for env in ENVIRONMENTS:
        gym_env = gym.make(env)
        rewards_run = train(gym_env)
        gym_env.close()

        print(f"\nTraining complete for environment {env}. Saving results...\n")
        rewards_run = np.array(rewards_run)
        mean_rewards = np.mean(rewards_run, axis=0)
        std_rewards = np.std(rewards_run, axis=0)

        plt.figure()
        plt.plot(np.arange(1, NB_TRANSITIONS+1, int(NB_TRANSITIONS/NB_RUN_STEP)), mean_rewards)
        plt.fill_between(np.arange(1, NB_TRANSITIONS+1, int(NB_TRANSITIONS/NB_RUN_STEP)),\
                         mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)
        plt.xlabel("OSST size")
        plt.ylabel("Cumulative reward")
        plt.title(f"Training FQI on {env}")
        plt.savefig(f"figures/{env}_fqi_training.png")
        plt.close()

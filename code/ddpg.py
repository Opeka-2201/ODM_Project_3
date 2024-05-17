"""
    ddpg.py: File that contains the implementation of the DDPG algorithm
    When running the file directly, it will train the FQI model on the selected environment
    When calling as a module, it will use the trained model to control the environment
"""

from copy import deepcopy
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.init import uniform_
import numpy as np

## TRAINING CONSTANTS ##
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_PATH = "models"
ENVIRONMENTS_TRAIN = [] # Change to ["InvertedDoublePendulum-v4", "InvertedPendulum-v4" for training
ENVIRONMENTS_RUN = ["InvertedDoublePendulum-v4", "InvertedPendulum-v4"]
LR_DDPG_ACTOR = .0001
LR_DDPG_CRITIC = LR_DDPG_ACTOR * 10
BOUND_INIT = .005
REPLACE = False
NB_RUNS = 2
NB_EPOCHS = 30
EPOCHS_SAVE_INTERVAL = 5
NOISE_SIMPLE = .3
NOISE_DOUBLE = .1
NB_TRANSITIONS = 2000
TRANSITIONS_TEST_INTERVAL = 200
NB_RUNS_TESTS = 10
BOUND_ACTION = 3
FACTOR_SIMPLE = 3
FACTOR_DOUBLE = 1
BATCH_SIZE = 128
GAMMA = .99
UPDATE_RATE = .001

# pylint: disable=too-few-public-methods, redefined-outer-name, arguments-differ
# pylint: disable=too-many-statements, too-many-nested-blocks, too-many-instance-attributes

class DDPGNetworkBuffer():
    """
        Class that defines the buffer to store the experiences of the DDPG model
        
        Attributes:
            buffer (list): The buffer to store the experiences
            appending_position (int): The position to append the next experience
            size (int): The size of the buffer
    """

    def __init__(self, size):
        self.buffer = []
        self.appending_position = 0
        self.size = size

    def get_item(self, num_items, replace):
        """
            Function that will return a number of items from the buffer
            
            Arguments:
                num_items (int): The number of items to return
                replace (bool): Whether to replace the items or not
            
            Returns:
                states (list): The states of the experiences
                actions (list): The actions of the experiences
                rewards (list): The rewards of the experiences
                next_states (list): The next states of the experiences
                terminals (list): The terminals of the experiences
                truncs (list): The truncs of the experiences
        """

        randoms = np.random.choice(self.get_size(), num_items, replace=replace)
        states, actions, rewards, next_states, terminals, truncs = [], [], [], [], [], []

        for random in randoms:
            state, action, reward, next_state, terminal, trunc = self.buffer[random]
            states.append(state.unsqueeze(0))
            action = torch.from_numpy(action)
            action = action.unsqueeze(0)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(terminal)
            truncs.append(trunc)

        return states, actions, rewards, next_states, terminals, truncs

    def get_size(self):
        """
            Function that will return the size of the buffer
        """

        return len(self.buffer)

    def append(self, osst):
        """ 
            Function that will append an experience to the buffer
            
            Arguments:
                osst (tuple): The experience to append to the buffer
        """
        if self.get_size() < self.size:
            self.buffer.append(osst)
        else:
            self.buffer[self.appending_position] = osst

        self.appending_position = (self.appending_position + 1) % self.size

class DDPGNetworkActor(nn.Module):
    """
        Class that defines the actor model of the DDPG algorithm
        
        Attributes:
            fc1 (nn.Linear): The first fully connected layer of the model
            fc2 (nn.Linear): The second fully connected layer of the model
            fc3 (nn.Linear): The third fully connected layer of the model
            fc4 (nn.Linear): The fourth fully connected layer of the model
            activation (nn.ReLU): The activation function of the model
            final_activation (nn.Tanh): The final activation function of the model
            optimizer (torch.optim.Adam): The optimizer of the model
    """

    def __init__(self, features, actions, training=True):
        super().__init__()
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, actions)
        self.fc4.weight = uniform_(self.fc4.weight, -BOUND_INIT, BOUND_INIT)
        self.fc4.bias = uniform_(self.fc4.bias, -BOUND_INIT, BOUND_INIT)
        self.activation = nn.ReLU()
        self.final_activation = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR_DDPG_ACTOR)
        self.training = training

    def forward(self, x):
        """
            Forward pass of the model
        
            Arguments:
                x (torch.Tensor): The input tensor to the model
        """

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.final_activation(self.fc4(x))
        return x

    def train_model(self, states, ddpgnet_critic):
        """
            Function that will train the actor model
            
            Arguments:
                states (torch.Tensor): The states to train the model on
                ddpgnet_critic (DDPGNetworkCritic): The critic model to use
        """

        if self.training:
            self.optimizer.zero_grad()
            ddpgnet_critic.change_gradient_mode(False)

            pred = self.forward(states)
            loss = -1 * ddpgnet_critic(states, pred).mean()
            loss.backward()

            ddpgnet_critic.change_gradient_mode(True)
            self.optimizer.step()

    def soft_update_parameters(self, ddpgnet_actor, update_rate):
        """
            Function that will update the parameters of the model
            
            Arguments:
                ddpgnet_actor (DDPGNetworkActor): The actor model to update
                update_rate (float): The rate to update the model
        """

        for self_param, else_param in zip(self.parameters(), ddpgnet_actor.parameters()):
            self_param.data = (((1 - update_rate) * self_param.data) \
                              + (update_rate * else_param.data))

    def change_gradient_mode(self, mode):
        """
            Function that will trigger the training of the actor model
            
            Arguments:
                mode (bool): Whether to train the model or not
        """

        for param in self.parameters():
            param.requires_grad = mode
class DDPGNetworkCritic(nn.Module):
    """
        Class that defines the critic model of the DDPG algorithm
        
        Attributes:
            fc1 (nn.Linear): The first fully connected layer of the model
            fc2 (nn.Linear): The second fully connected layer of the model
            fc3 (nn.Linear): The third fully connected layer of the model
            fc4 (nn.Linear): The fourth fully connected layer of the model
            activation (nn.ReLU): The activation function of the model
            optimizer (torch.optim.AdamW): The optimizer of the model
            loss (nn.MSELoss): The loss function of the model
    """

    def __init__(self, features, actions):
        super().__init__()
        self.fc1 = nn.Linear(features, 64-actions)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        uniform_(self.fc4.weight, -BOUND_INIT, BOUND_INIT)
        uniform_(self.fc4.bias, -BOUND_INIT, BOUND_INIT)
        self.activation = nn.ReLU()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=LR_DDPG_CRITIC)

    def forward(self, x, action):
        """
            Forward pass of the model
            
            Arguments:
                x (torch.Tensor): The input tensor to the model
                action (torch.Tensor): The action tensor to the model
                
            Returns:
                x (torch.Tensor): The output tensor of the model
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(torch.cat([x, action], dim=1)))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    def train_model(self, states, actions, ground_truth):
        """
            Function that will train the critic model
            
            Arguments:
                states (torch.Tensor): The states to train the model on
                actions (torch.Tensor): The actions to train the model on
                ground_truth (torch.Tensor): The ground truth to train the model on
        """

        self.optimizer.zero_grad()
        pred = self.forward(states, actions)
        loss = nn.MSELoss()(pred, ground_truth)
        loss.backward()
        self.optimizer.step()

    def soft_update_parameters(self, ddpgnet_critic, update_rate):
        """
            Function that will update the parameters of the model

            Arguments:
                ddpgnet_critic (DDPGNetworkCritic): The critic model to update
                update_rate (float): The rate to update the model
        """

        for self_param, else_param in zip(self.parameters(), ddpgnet_critic.parameters()):
            self_param.data = (((1 - update_rate) * self_param.data) \
                              + (update_rate * else_param.data))

    def change_gradient_mode(self, mode):
        """
            Function that will trigger the training of the actor model
            
            Arguments:
                mode (bool): Whether to train the model or not
        """

        for param in self.parameters():
            param.requires_grad = mode

def train_loop(gym_env, features, actions):
    """
        Function that will train the DDPG model on the environment
        
        Arguments:
            gym_env (gym.Env): The environment to train the DDPG model on
            features (int): The number of features of the environment
            actions (int): The number of actions of the environment
        
        Returns:
            rewards (list): The rewards of the model
    """

    env_name = gym_env.unwrapped.spec.id
    rewards = []

    if "Double" in env_name:
        noise = NOISE_DOUBLE
        action_factor = FACTOR_DOUBLE
        relu = False
    else:
        noise = NOISE_SIMPLE
        action_factor = FACTOR_SIMPLE
        relu = True

    noise = noise * (.5 ** (1 / NB_TRANSITIONS))

    for run in range(NB_RUNS):
        rewards_run = []
        replay_buffer = DDPGNetworkBuffer(5000)

        ddpgnet_actor = DDPGNetworkActor(features, actions).to(DEVICE)
        ddpgnet_actor_copy = deepcopy(ddpgnet_actor)
        ddpgnet_actor_copy.change_gradient_mode(True)

        ddpgnet_critic = DDPGNetworkCritic(features, actions).to(DEVICE)
        ddpgnet_critic_copy = deepcopy(ddpgnet_critic)
        ddpgnet_critic_copy.change_gradient_mode(True)

        for epoch in range(NB_EPOCHS):
            state = gym_env.reset()[0]
            rewards_epoch = []

            for transition in range(NB_TRANSITIONS):
                state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
                action = action_factor * ddpgnet_actor(state)
                action = action.detach()
                action = action + noise * torch.randn(actions).to(DEVICE)
                action = torch.clamp(action, -BOUND_ACTION, BOUND_ACTION)
                action = action.cpu().numpy()

                next_state, reward, terminal, trunc, _ = gym_env.step(action)
                replay_buffer.append((state, action, reward, next_state, terminal, trunc))

                if replay_buffer.get_size() > BATCH_SIZE:
                    b_states, b_actions, b_rewards, b_next_states, b_terminals, b_truncs = \
                        replay_buffer.get_item(BATCH_SIZE, REPLACE)

                    b_states = torch.cat(b_states, dim=0)
                    b_actions = torch.cat(b_actions, dim=0)
                    b_rewards = torch.tensor(b_rewards, dtype=torch.float32).to(DEVICE)
                    b_next_states = torch.tensor(np.array(b_next_states), \
                                                 dtype=torch.float32).to(DEVICE)
                    b_terminals = torch.tensor(b_terminals, dtype=torch.float32).to(DEVICE)
                    b_truncs = torch.tensor(b_truncs, dtype=torch.float32).to(DEVICE)

                    b_rewards = b_rewards.unsqueeze(dim=-1)
                    b_terminals = b_terminals.unsqueeze(dim=-1)
                    b_truncs = b_truncs.unsqueeze(dim=-1)
                    b_stop = torch.max(b_terminals, b_truncs)

                    tmp = ddpgnet_actor_copy(b_next_states)
                    critic_pred = ddpgnet_critic_copy(b_next_states, tmp)
                    critic_pred = nn.ReLU()(critic_pred) if relu else critic_pred

                    gt = b_rewards + (1 - b_stop) * critic_pred * GAMMA
                    ddpgnet_critic.train_model(b_states, b_actions, gt)
                    ddpgnet_actor.train_model(b_states, ddpgnet_critic_copy)
                    ddpgnet_actor_copy.soft_update_parameters(ddpgnet_actor, UPDATE_RATE)
                    ddpgnet_critic_copy.soft_update_parameters(ddpgnet_critic, UPDATE_RATE)

                state = next_state

                if terminal or trunc:
                    state = gym_env.reset()[0]

                if transition % TRANSITIONS_TEST_INTERVAL == 0:
                    test_rewards = []
                    for _ in range(NB_RUNS_TESTS):
                        state_test = gym_env.reset()[0]
                        reward_test_run = 0

                        while True:
                            with torch.no_grad():
                                state_test = torch.tensor(state_test,\
                                                          dtype=torch.float32).to(DEVICE)
                                action_test = action_factor * \
                                              ddpgnet_actor(state_test).cpu().numpy()
                                state_test, reward_test, terminal_test,\
                                    trunc_test, _ = gym_env.step(action_test)
                                reward_test_run += reward_test

                                if terminal_test or trunc_test:
                                    break

                        test_rewards.append(reward_test_run)

                    rewards_epoch.append(np.mean(test_rewards))
                    print(f"Run: {run+1}/{NB_RUNS} - Epoch: {epoch+1}/{NB_EPOCHS} - Transition", \
                          f"{transition+TRANSITIONS_TEST_INTERVAL}/{NB_TRANSITIONS} -", \
                          f"Test Reward: {rewards_epoch[-1]}\r", end="")

            rewards_run.append(np.mean(rewards_epoch))

            if epoch + 1 % EPOCHS_SAVE_INTERVAL == 0:
                torch.save(ddpgnet_actor.state_dict(), f"{MODELS_PATH}/{env_name}/" + \
                                          f"ddpg_actor_run_{run+1}_epoch_{epoch+1}.pt")
                torch.save(ddpgnet_critic.state_dict(), f"{MODELS_PATH}/{env_name}/" + \
                                           f"ddpg_critic_run_{run+1}_epoch_{epoch+1}.pt")

        rewards.append(rewards_run)

    return rewards

def run_ddpg(env, model_path):
    """
        Function that will run the DDPG model on the environment
        
        Arguments:
            env (str): The environment to run the FQI model on
            model_path (str): The path to the model to use
    """

    env_name = env.unwrapped.spec.id
    features = env.observation_space.shape[0]
    actions = env.action_space.shape[0]

    if env_name not in ENVIRONMENTS_RUN:
        raise ValueError("Invalid environment")

    if "Double" in env_name:
        action_factor = FACTOR_DOUBLE
        bound = BOUND_ACTION
    else:
        action_factor = FACTOR_SIMPLE
        bound = BOUND_ACTION

    ddpgnet_actor = DDPGNetworkActor(features, actions, False)
    ddpgnet_actor.load_state_dict(torch.load(model_path))

    state = env.reset()[0]
    cumulative_reward = 0

    with torch.no_grad():
        while True:
            state = torch.tensor(state, dtype=torch.float32)
            action = ddpgnet_actor.forward(state).detach()
            action = action_factor * action.cpu().numpy()
            action = np.clip(action, -bound, bound)
            state, reward, terminal, trunc, _ = env.step(action)

            if terminal or trunc:
                break

            cumulative_reward += reward

    print(f"Finished running DDPG on {env_name} - Cumulative Reward: {cumulative_reward}")

if __name__ == "__main__":
    for env in ENVIRONMENTS_TRAIN:
        gym_env = gym.make(env)
        env_name = gym_env.unwrapped.spec.id
        print(f"Training DDPG model on {env_name}")
        features = gym_env.observation_space.shape[0]
        actions = gym_env.action_space.shape[0]
        print(f"Features: {features} - Actions: {actions}")

        rewards = train_loop(gym_env, features, actions)

        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)

        plt.plot(rewards_mean, label="Mean Cumulative Reward")
        plt.fill_between(range(len(rewards_mean)), \
                         rewards_mean - rewards_std, rewards_mean + rewards_std, alpha=.5)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title(f"{env_name} - DDPG")
        plt.savefig(f"figures/{env_name}_ddpg.png")
        plt.close()

        print(f"Finished training DDPG model on {env_name}")
        gym_env.close()

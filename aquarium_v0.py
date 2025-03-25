"""Aquarium environment v0"""

from pettingzoo.utils import aec_to_parallel, parallel_to_aec
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper

from env.aquarium import raw_env

def env2(
    render_mode: str = "human",
    observable_walls: int = 2,
    width: int = 800,
    height: int = 800,
    caption: str = "Aquarium",
    fps: int = 60,
    max_time_steps: int = 3000,
    action_count: int = 16,
    predator_count: int = 1,
    prey_count: int = 16,
    predator_observe_count: int = 1,
    prey_observe_count: int = 3,
    draw_force_vectors: bool = False,
    draw_action_vectors: bool = False,
    draw_view_cones: bool = False,
    draw_hit_boxes: bool = False,
    draw_death_circles: bool = False,
    fov_enabled: bool = True,
    keep_prey_count_constant: bool = True,
    prey_radius: int = 20,
    prey_max_acceleration: float = 1,
    prey_max_velocity: float = 4,
    prey_view_distance: int = 100,
    prey_replication_age: int = 200,
    prey_max_steer_force: float = 0.6,
    prey_fov: int = 120,
    prey_reward: int = 1,
    prey_punishment: int = 1000,
    max_prey_count: int = 20,
    predator_max_acceleration: float = 0.6,
    predator_radius: int = 30,
    predator_max_velocity: float = 5,
    predator_view_distance: int = 200,
    predator_max_steer_force: float = 0.6,
    predator_max_age: int = 3000,
    predator_fov: int = 150,
    predator_reward: int = 10,
    catch_radius: int = 100,
    procreate: bool = False,
):
    """Returns the AEC environment"""
    env_aec = parallel_to_aec(
        raw_env(
            render_mode=render_mode,
            observable_walls=observable_walls,
            width=width,
            height=height,
            caption=caption,
            fps=fps,
            max_time_steps=max_time_steps,
            action_count=action_count,
            predator_count=predator_count,
            prey_count=prey_count,
            predator_observe_count=predator_observe_count,
            prey_observe_count=prey_observe_count,
            draw_force_vectors=draw_force_vectors,
            draw_action_vectors=draw_action_vectors,
            draw_view_cones=draw_view_cones,
            draw_hit_boxes=draw_hit_boxes,
            draw_death_circles=draw_death_circles,
            fov_enabled=fov_enabled,
            keep_prey_count_constant=keep_prey_count_constant,
            prey_radius=prey_radius,
            prey_max_acceleration=prey_max_acceleration,
            prey_max_velocity=prey_max_velocity,
            prey_view_distance=prey_view_distance,
            prey_replication_age=prey_replication_age,
            prey_max_steer_force=prey_max_steer_force,
            prey_fov=prey_fov,
            prey_reward=prey_reward,
            prey_punishment=prey_punishment,
            max_prey_count=max_prey_count,
            predator_max_acceleration=predator_max_acceleration,
            predator_radius=predator_radius,
            predator_max_velocity=predator_max_velocity,
            predator_view_distance=predator_view_distance,
            predator_max_steer_force=predator_max_steer_force,
            predator_max_age=predator_max_age,
            predator_fov=predator_fov,
            predator_reward=predator_reward,
            catch_radius=catch_radius,
            procreate=procreate,
        )
    )
    env_aec = AssertOutOfBoundsWrapper(env_aec)
    env_aec = OrderEnforcingWrapper(env_aec)

    return env_aec


def parallel_env(
    render_mode: str = "human",
    observable_walls: int = 2,
    width: int = 800,
    height: int = 800,
    caption: str = "Aquarium",
    fps: int = 60,
    max_time_steps: int = 3000,
    action_count: int = 16,
    predator_count: int = 1,
    prey_count: int = 16,
    predator_observe_count: int = 1,
    prey_observe_count: int = 3,
    draw_force_vectors: bool = False,
    draw_action_vectors: bool = False,
    draw_view_cones: bool = False,
    draw_hit_boxes: bool = False,
    draw_death_circles: bool = False,
    fov_enabled: bool = True,
    keep_prey_count_constant: bool = True,
    prey_radius: int = 20,
    prey_max_acceleration: float = 1,
    prey_max_velocity: float = 4,
    prey_view_distance: int = 100,
    prey_replication_age: int = 200,
    prey_max_steer_force: float = 0.6,
    prey_fov: int = 120,
    prey_reward: int = 1,
    prey_punishment: int = 1000,
    max_prey_count: int = 20,
    predator_max_acceleration: float = 0.6,
    predator_radius: int = 30,
    predator_max_velocity: float = 5,
    predator_view_distance: int = 200,
    predator_max_steer_force: float = 0.6,
    predator_max_age: int = 3000,
    predator_fov: int = 150,
    predator_reward: int = 10,
    catch_radius: int = 100,
    procreate: bool = False,
):
    """Returns the parallel environment"""
    return aec_to_parallel(
        env(
            render_mode=render_mode,
            observable_walls=observable_walls,
            width=width,
            height=height,
            caption=caption,
            fps=fps,
            max_time_steps=max_time_steps,
            action_count=action_count,
            predator_count=predator_count,
            prey_count=prey_count,
            predator_observe_count=predator_observe_count,
            prey_observe_count=prey_observe_count,
            draw_force_vectors=draw_force_vectors,
            draw_action_vectors=draw_action_vectors,
            draw_view_cones=draw_view_cones,
            draw_hit_boxes=draw_hit_boxes,
            draw_death_circles=draw_death_circles,
            fov_enabled=fov_enabled,
            keep_prey_count_constant=keep_prey_count_constant,
            prey_radius=prey_radius,
            prey_max_acceleration=prey_max_acceleration,
            prey_max_velocity=prey_max_velocity,
            prey_view_distance=prey_view_distance,
            prey_replication_age=prey_replication_age,
            prey_max_steer_force=prey_max_steer_force,
            prey_fov=prey_fov,
            prey_reward=prey_reward,
            prey_punishment=prey_punishment,
            max_prey_count=max_prey_count,
            predator_max_acceleration=predator_max_acceleration,
            predator_radius=predator_radius,
            predator_max_velocity=predator_max_velocity,
            predator_view_distance=predator_view_distance,
            predator_max_steer_force=predator_max_steer_force,
            predator_max_age=predator_max_age,
            predator_fov=predator_fov,
            predator_reward=predator_reward,
            catch_radius=catch_radius,
            procreate=procreate,
        )
    )


env = env2(
    # draw_force_vectors=True,
    # draw_action_vectors=True,
    # draw_view_cones=True,
    # draw_hit_boxes=True,
    # draw_death_circles=True,
    procreate = True
)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    len(env.agents)

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
    env.render()
env.close()


# import numpy as np
# import gymnasium as gym
# from pettingzoo.utils import agent_selector
# from collections import defaultdict

# class QLearningAgent:
#     def __init__(self, observation_space, action_space, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
#         """
#         Initialize Q-learning agent
        
#         Using dictionary to store Q-values instead of a huge array to save memory
#         """
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.min_exploration_rate = min_exploration_rate
        
#         # Record observation space info for discretization
#         if isinstance(observation_space, gym.spaces.Box):
#             self.obs_high = observation_space.high
#             self.obs_low = observation_space.low
#             self.obs_shape = observation_space.shape
            
#             # Reduce discretization granularity to make the state space smaller
#             self.observation_bins = 5  # Lower discretization granularity
            
#             # Further reduce dimensions - only use the most important parts of the observation vector
#             self.feature_indices = list(range(min(6, self.obs_shape[0])))  # Use at most the first 6 features
#         else:
#             raise ValueError("Unsupported observation space type")
        
#         # Use defaultdict instead of huge array
#         self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
#         self.action_space = action_space
#         self.last_state = None
#         self.last_action = None
    
#     def discretize_state(self, observation):
#         """
#         Discretize continuous observation into state indices, and reduce dimensions
#         """
#         if isinstance(observation, dict):
#             # If observation is a dictionary, extract relevant features
#             features = []
#             for key in sorted(observation.keys()):
#                 if isinstance(observation[key], np.ndarray):
#                     features.extend(observation[key].flatten())
#                 elif isinstance(observation[key], (int, float)):
#                     features.append(observation[key])
#             observation = np.array(features)
        
#         # Only keep selected features
#         if len(observation) > len(self.feature_indices):
#             observation = observation[self.feature_indices]
        
#         # Ensure observation is within reasonable range
#         obs_high = self.obs_high
#         obs_low = self.obs_low
        
#         if len(self.feature_indices) < len(obs_high):
#             obs_high = obs_high[self.feature_indices]
#             obs_low = obs_low[self.feature_indices]
            
#         observation = np.clip(observation, obs_low, obs_high)
        
#         # Linear discretization
#         scaled = (observation - obs_low) / (obs_high - obs_low + 1e-10)
#         discrete_state = np.floor(scaled * self.observation_bins).astype(int)
        
#         # Ensure indices are within valid range
#         discrete_state = np.clip(discrete_state, 0, self.observation_bins - 1)
        
#         # Convert to string to use as dictionary key
#         return str(discrete_state.tolist())
    
#     def choose_action(self, observation):
#         """
#         Choose action based on current observation (epsilon-greedy policy)
#         """
#         state = self.discretize_state(observation)
#         self.last_state = state
        
#         # Exploration (random action)
#         if np.random.random() < self.exploration_rate:
#             action = self.action_space.sample()
#         # Exploitation (best Q-value action)
#         else:
#             action = np.argmax(self.q_table[state])
        
#         self.last_action = action
#         return action
    
#     def learn(self, observation, reward, done):
#         """
#         Update Q-values based on received reward and new observation
#         """
#         if self.last_state is None:
#             return
        
#         current_state = self.last_state
#         action = self.last_action
        
#         if done:
#             # Q-value update for terminal state
#             self.q_table[current_state][action] = (1 - self.learning_rate) * self.q_table[current_state][action] + \
#                                                   self.learning_rate * reward
#         else:
#             next_state = self.discretize_state(observation)
#             # Q-learning update rule
#             self.q_table[current_state][action] = (1 - self.learning_rate) * self.q_table[current_state][action] + \
#                                                   self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]))
        
#         # Decay exploration rate
#         if not done:
#             self.exploration_rate = max(self.min_exploration_rate, 
#                                        self.exploration_rate * self.exploration_decay)


# # Main training loop
# def train_q_learning(env, num_episodes=10, render_interval=10000):
#     """
#     Train Q-learning agent
    
#     Parameters:
#     env: PettingZoo environment
#     num_episodes: Total number of training episodes
#     render_interval: How many episodes to render once
#     """
#     # Create Q-learning agent for each agent
#     agents = {}
    
#     episode_rewards = {}
    
#     for episode in range(num_episodes):
#         # Reset the environment for a new episode
#         env.reset(seed=episode)  # Use different seeds to increase training diversity
        
#         # Update agent dictionary after environment reset
#         # This is important because agents may change between episodes
#         for agent_id in env.possible_agents:
#             if agent_id not in agents:
#                 agents[agent_id] = QLearningAgent(
#                     env.observation_space(agent_id),
#                     env.action_space(agent_id)
#                 )
            
#             # Initialize episode rewards for each agent
#             if agent_id not in episode_rewards:
#                 episode_rewards[agent_id] = []
        
#         total_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        
#         # Decide whether to render this episode
#         render = (episode % render_interval == 0)
        
#         # Keep track of active agents in this episode
#         active_agents = set(env.possible_agents)
        
#         # Main agent iteration loop
#         for agent_id in env.agent_iter():
#             # Check if agent still exists in the environment
#             if agent_id not in active_agents:
#                 print(f"Warning: Agent {agent_id} no longer in active_agents set")
#                 continue
                
#             observation, reward, termination, truncation, info = env.last()
            
#             if agent_id in agents:
#                 # Accumulate rewards
#                 total_rewards[agent_id] += reward
                
#                 # Termination or truncation
#                 done = termination or truncation
                
#                 if done:
#                     action = None
#                     # Remove from active agents if terminated or truncated
#                     active_agents.discard(agent_id)
#                 else:
#                     # Get action from Q-learning agent
#                     action = agents[agent_id].choose_action(observation)
                    
#                     # Learning update (using previous state, action and current reward)
#                     agents[agent_id].learn(observation, reward, done)
                
#                 try:
#                     env.step(action)
#                 except KeyError as e:
#                     print(f"KeyError during env.step(): {e}")
#                     # If we get a KeyError, the agent might have been removed during stepping
#                     # Remove it from active agents
#                     active_agents.discard(agent_id)
#                     continue
                
#                 # if render:
#                 #     env.render()
        
#         # Record total rewards for each agent
#         for agent_id in env.possible_agents:
#             if agent_id in total_rewards:
#                 if agent_id not in episode_rewards:
#                     episode_rewards[agent_id] = []
#                 episode_rewards[agent_id].append(total_rewards[agent_id])
        
#         # Print progress
#         if episode % 10 == 0:
#             avg_rewards = {agent_id: np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards) 
#                          for agent_id, rewards in episode_rewards.items()}
#             print(f"Episode: {episode}, Average Rewards: {avg_rewards}")
            
#             # Print Q-table size
#             if episode % 100 == 0:
#                 for agent_id, agent in agents.items():
#                     print(f"Agent {agent_id} Q-table size: {len(agent.q_table)} states")
    
#     env.close()
#     return agents, episode_rewards

# # Evaluate trained agents
# def evaluate_agents(env, agents, num_episodes=10, render=True):
#     """
#     Evaluate trained Q-learning agents
    
#     Parameters:
#     env: PettingZoo environment
#     agents: Dictionary of trained Q-learning agents
#     num_episodes: Number of evaluation episodes
#     render: Whether to render the environment
#     """
#     evaluation_rewards = {agent_id: [] for agent_id in env.possible_agents}
    
#     for episode in range(num_episodes):
#         env.reset(seed=42 + episode)  # Use seeds different from training
#         total_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        
#         # Keep track of active agents in this episode
#         active_agents = set(env.possible_agents)
        
#         for agent_id in env.agent_iter():
#             # Check if agent still exists in the environment
#             if agent_id not in active_agents:
#                 continue
                
#             observation, reward, termination, truncation, info = env.last()
            
#             if agent_id in agents:
#                 # Accumulate rewards
#                 total_rewards[agent_id] += reward
                
#                 # Termination or truncation
#                 done = termination or truncation
                
#                 if done:
#                     action = None
#                     # Remove from active agents if terminated or truncated
#                     active_agents.discard(agent_id)
#                 else:
#                     # Use greedy policy during evaluation (no exploration)
#                     state = agents[agent_id].discretize_state(observation)
#                     action = np.argmax(agents[agent_id].q_table[state])
                
#                 try:
#                     env.step(action)
#                 except KeyError as e:
#                     print(f"KeyError during evaluation env.step(): {e}")
#                     # If we get a KeyError, the agent might have been removed during stepping
#                     # Remove it from active agents
#                     active_agents.discard(agent_id)
#                     continue
                
#                 if render:
#                     env.render()
        
#         # Record total rewards for each agent
#         for agent_id in env.possible_agents:
#             if agent_id in total_rewards and agent_id in evaluation_rewards:
#                 evaluation_rewards[agent_id].append(total_rewards[agent_id])
        
#         print(f"Evaluation Episode: {episode}, Rewards: {total_rewards}")
    
#     env.close()
#     return evaluation_rewards

# # Main function
# def main():
#     # Create environment
#     # Note: You need to modify this import to match your actual environment structure
    
#     aquarium_env = env2(
#         # Optional configuration parameters
#         # draw_force_vectors=True,
#         # draw_action_vectors=True,
#         # draw_view_cones=True,
#         # draw_hit_boxes=True,
#         # draw_death_circles=True,
#     )
    
#     # Train Q-learning agents
#     print("Starting training...")
#     trained_agents, training_rewards = train_q_learning(aquarium_env, num_episodes=5, render_interval=100)
    
#     # Evaluate trained agents
#     print("\nStarting evaluation...")
#     evaluation_rewards = evaluate_agents(aquarium_env, trained_agents, num_episodes=5, render=True)
    
#     # Print evaluation results
#     for agent_id in aquarium_env.possible_agents:
#         if agent_id in evaluation_rewards:
#             avg_reward = np.mean(evaluation_rewards[agent_id])
#             print(f"Agent {agent_id} - Average Evaluation Reward: {avg_reward:.2f}")

# if __name__ == "__main__":
#     main()



# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from collections import deque

# # Define DQN Network using PyTorch
# class DQNNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQNNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 24)
#         self.fc2 = nn.Linear(24, 24)
#         self.fc3 = nn.Linear(24, action_size)
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # DQN Agent
# class DQNAgent:
#     def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, 
#                  exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01,
#                  device="cpu"):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.device = device
#         self.memory = deque(maxlen=2000)  # Experience replay buffer
        
#         # Hyperparameters
#         self.gamma = discount_factor     # Discount factor
#         self.epsilon = exploration_rate  # Exploration rate
#         self.epsilon_decay = exploration_decay
#         self.epsilon_min = min_exploration_rate
#         self.learning_rate = learning_rate
        
#         # Build neural network models - policy network
#         self.policy_net = DQNNetwork(state_size, action_size).to(device)
        
#         # Optimizer
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
#         self.criterion = nn.MSELoss()
        
#     def remember(self, state, action, reward, next_state, done):
#         """Store experience in replay memory"""
#         self.memory.append((state, action, reward, next_state, done))
    
#     def choose_action(self, state):
#         """Select action using epsilon-greedy policy"""
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
        
#         state_tensor = torch.FloatTensor(state).to(self.device)
#         with torch.no_grad():
#             action_values = self.policy_net(state_tensor)
#         return torch.argmax(action_values).item()
    
#     def replay(self, batch_size=32):
#         """Train the network using experience replay"""
#         if len(self.memory) < batch_size:
#             return
        
#         # Sample random batch from memory
#         minibatch = random.sample(self.memory, batch_size)
        
#         states = []
#         targets = []
        
#         for state, action, reward, next_state, done in minibatch:
#             state_tensor = torch.FloatTensor(state).to(self.device)
#             next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
#             # Get current Q values
#             target = self.policy_net(state_tensor).detach().cpu().numpy()
            
#             if done:
#                 target[0][action] = reward
#             else:
#                 # Get max Q value for next state
#                 with torch.no_grad():
#                     next_q_values = self.policy_net(next_state_tensor)
#                     max_next_q = torch.max(next_q_values).item()
                
#                 # Update target Q value
#                 target[0][action] = reward + self.gamma * max_next_q
            
#             states.append(state)
#             targets.append(target)
        
#         # Convert lists to tensors for batch training
#         state_batch = torch.FloatTensor(np.array(states)).to(self.device)
#         target_batch = torch.FloatTensor(np.array(targets)).to(self.device)
        
#         # Train the network
#         self.optimizer.zero_grad()
#         outputs = self.policy_net(state_batch)
#         loss = self.criterion(outputs, target_batch)
#         loss.backward()
#         self.optimizer.step()
        
#         # Decay exploration rate
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
    
#     def save_model(self, filename):
#         """Save model weights to a file"""
#         torch.save(self.policy_net.state_dict(), filename)
    
#     def load_model(self, filename):
#         """Load model weights from a file"""
#         self.policy_net.load_state_dict(torch.load(filename))
#         self.policy_net.eval()

# # Preprocessing function to handle different observation types
# def preprocess_observation(observation, state_size):
#     """Convert observation to a format suitable for the neural network"""
#     if isinstance(observation, np.ndarray):
#         # If observation is already an array, reshape it
#         processed = observation.flatten()
#         # Pad or truncate to match expected state_size
#         if len(processed) < state_size:
#             processed = np.pad(processed, (0, state_size - len(processed)))
#         else:
#             processed = processed[:state_size]
#     elif isinstance(observation, dict):
#         # For dictionary observations, extract and flatten key values
#         processed = []
#         for key in sorted(observation.keys()):
#             if isinstance(observation[key], np.ndarray):
#                 processed.extend(observation[key].flatten())
#             elif isinstance(observation[key], (int, float)):
#                 processed.append(observation[key])
        
#         # Pad or truncate to match expected state_size
#         processed = np.array(processed)
#         if len(processed) < state_size:
#             processed = np.pad(processed, (0, state_size - len(processed)))
#         else:
#             processed = processed[:state_size]
    
#     # Reshape for the neural network input
#     return processed.reshape(1, state_size)

# # Main training function
# def train_dqn(env, episodes=300, state_size=10, batch_size=32, render_every=50, device="cpu"):
#     """Train DQN agents on the aquarium environment"""
#     # Dictionary to store agents
#     agents = {}
    
#     # Dictionary to store rewards history
#     rewards_history = {}
    
#     for episode in range(episodes):
#         # Reset environment
#         env.reset(seed=episode)
        
#         # Create DQN agents for new agents in environment
#         for agent_id in env.possible_agents:
#             if agent_id not in agents:
#                 action_space = env.action_space(agent_id)
#                 action_size = action_space.n
#                 agents[agent_id] = DQNAgent(state_size=state_size, action_size=action_size, device=device)
            
#             if agent_id not in rewards_history:
#                 rewards_history[agent_id] = []
        
#         # Track rewards for this episode
#         episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        
#         # Decide whether to render this episode
#         render = (episode % render_every == 0)
        
#         # Track active agents
#         active_agents = set(env.possible_agents)
        
#         # Previous states and actions for learning
#         prev_states = {}
#         prev_actions = {}
        
#         # Main agent iteration loop
#         for agent_id in env.agent_iter():
#             # Skip if agent no longer active
#             if agent_id not in active_agents:
#                 continue
            
#             # Get current state, reward, and done status
#             observation, reward, termination, truncation, info = env.last()
            
#             # Accumulate reward
#             episode_rewards[agent_id] += reward
            
#             # Check if done
#             done = termination or truncation
            
#             # Process observation into state vector
#             current_state = preprocess_observation(observation, state_size)
            
#             # If agent has previous state, store experience
#             if agent_id in prev_states and agent_id in prev_actions:
#                 agents[agent_id].remember(
#                     prev_states[agent_id], 
#                     prev_actions[agent_id], 
#                     reward, 
#                     current_state, 
#                     done
#                 )
            
#             # Select action
#             if done:
#                 action = None
#                 active_agents.discard(agent_id)
#             else:
#                 action = agents[agent_id].choose_action(current_state)
#                 # Remember state and action for next step
#                 prev_states[agent_id] = current_state
#                 prev_actions[agent_id] = action
            
#             # Execute action
#             try:
#                 env.step(action)
#             except KeyError:
#                 # Agent might have been removed during step
#                 active_agents.discard(agent_id)
#                 continue
                
#             # Train on past experiences (experience replay)
#             if not done and len(agents[agent_id].memory) > batch_size:
#                 agents[agent_id].replay(batch_size)
            
#             # Render if needed
#             if render:
#                 env.render()
        
#         # Record rewards
#         for agent_id in env.possible_agents:
#             if agent_id in episode_rewards:
#                 if agent_id not in rewards_history:
#                     rewards_history[agent_id] = []
#                 rewards_history[agent_id].append(episode_rewards[agent_id])
        
#         # Print progress every 10 episodes
#         if episode % 10 == 0:
#             avg_rewards = {}
#             for agent_id, rewards in rewards_history.items():
#                 if len(rewards) > 0:
#                     recent_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
#                     avg_rewards[agent_id] = np.mean(recent_rewards)
#             print(f"Episode {episode}/{episodes}, Average Rewards: {avg_rewards}")
            
#             # Save models for select agents
#             for agent_id in env.possible_agents:
#                 if agent_id in agents and "predator" in agent_id:
#                     agents[agent_id].save_model(f"dqn_model_{agent_id}_ep{episode}.pt")
    
#     return agents, rewards_history

# # Evaluation function
# def evaluate_dqn(env, agents, episodes=2, state_size=10):
#     """Evaluate trained DQN agents"""
#     for episode in range(episodes):
#         env.reset(seed=1000 + episode)
        
#         # Track rewards
#         total_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        
#         # Track active agents
#         active_agents = set(env.possible_agents)
        
#         for agent_id in env.agent_iter():
#             if agent_id not in active_agents:
#                 continue
                
#             observation, reward, termination, truncation, info = env.last()
            
#             total_rewards[agent_id] += reward
            
#             done = termination or truncation
            
#             if done:
#                 action = None
#                 active_agents.discard(agent_id)
#             else:
#                 # Process observation
#                 state = preprocess_observation(observation, state_size)
                
#                 # Use trained policy (no exploration)
#                 if agent_id in agents:
#                     # Set epsilon to 0 to disable exploration
#                     original_epsilon = agents[agent_id].epsilon
#                     agents[agent_id].epsilon = 0
#                     action = agents[agent_id].choose_action(state)
#                     agents[agent_id].epsilon = original_epsilon
#                 else:
#                     # For unseen agents, use random action
#                     action = env.action_space(agent_id).sample()
            
#             try:
#                 env.step(action)
#             except KeyError:
#                 active_agents.discard(agent_id)
#                 continue
            
#             # Always render evaluation
#             # env.render()
        
#         print(f"Evaluation Episode {episode}, Rewards: {total_rewards}")

# # Main function
# def main():
#     # Import and create environment
    
#     aquarium_env = env2(
#         # Optional configuration parameters
#         # draw_force_vectors=True,
#         # draw_action_vectors=True,
#         # draw_view_cones=True,
#         # draw_hit_boxes=True,
#         # draw_death_circles=True,
#     )
    
#     # Define state size for neural network
#     STATE_SIZE = 10  # Adjust based on your observation space
    
#     # Choose device (CPU or GPU)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Train DQN agents
#     print("Starting training...")
#     trained_agents, rewards_history = train_dqn(
#         aquarium_env, 
#         episodes=2, 
#         state_size=STATE_SIZE,
#         render_every=50,
#         device=device
#     )
    
#     # Evaluate agents
#     print("\nStarting evaluation...")
#     evaluate_dqn(
#         aquarium_env, 
#         trained_agents, 
#         episodes=5,
#         state_size=STATE_SIZE
#     )
    
#     # Save final models
#     for agent_id, agent in trained_agents.items():
#         agent.save_model(f"dqn_model_{agent_id}_final.pt")
    
#     # Close environment
#     aquarium_env.close()

# if __name__ == "__main__":
#     main()
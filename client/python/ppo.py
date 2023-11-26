import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import deque

# Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, num_player_features):
        super(ActorCritic, self).__init__()
        # process map data
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # process each player
        self.fc_player1 = nn.Sequential(
            nn.Linear(num_player_features, 128),
            nn.ReLU()
        )
        self.fc_player2 = nn.Sequential(
            nn.Linear(num_player_features, 128),
            nn.ReLU()
        )

        # combine two players info and map data
        combined_size = 64 * 15 * 15 + 128 * 2

        self.actor = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def forward(self, map_state, player1_features, player2_features):
        map_state = self.conv_layers(map_state)
        map_state = map_state.view(map_state.size(0), -1)  # flatten conv layer

        player1_features = self.fc_player1(player1_features)
        player2_features = self.fc_player2(player2_features)

        combined_features = torch.cat((map_state, player1_features, player2_features), dim=1)

        return self.actor(combined_features), self.critic(combined_features)



# PPO agent
class PPOAgent:
    def __init__(self, num_inputs, num_actions, num_player_features):
        self.actor_critic = ActorCritic(num_inputs, num_actions, num_player_features).cuda()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-4)
        self.num_actions = num_actions
        
        # PPO Hyperparameters
        self.ppo_steps = 128
        self.ppo_clip = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_beta = 0.01

    
    def select_action(self, map_state, player1_features, player2_features):
        map_state = torch.from_numpy(map_state).float().unsqueeze(0).cuda()  
        map_state = map_state.permute(0, 3, 1, 2)  # change dim

        player1_features = torch.from_numpy(player1_features).float().unsqueeze(0).cuda()
        player2_features = torch.from_numpy(player2_features).float().unsqueeze(0).cuda()
    
        probs, value = self.actor_critic(map_state, player1_features, player2_features)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), value, dist.log_prob(action)


    def compute_advantages(self, rewards, values, masks, gamma=0.99, lam=0.95):
        advs = []
        last_gae_lam = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - masks[-1]
                next_values = values[-1]
            else:
                next_non_terminal = 1.0 - masks[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            advs.insert(0, last_gae_lam)
        return advs


    def ppo_update(self, states, actions, log_probs, returns, advantages, player1_features, player2_features, actor_critic, optimizer, ppo_epochs, mini_batch_size, ppo_clip):
        for _ in range(ppo_epochs):
            idx = np.arange(len(states))
            np.random.shuffle(idx)

            for i in range(0, len(states), mini_batch_size):
                ind = idx[i:i + mini_batch_size]

                states_mb = torch.stack([s.permute(2, 0, 1) for s in states])[ind]
                actions_mb = torch.tensor(actions)[ind].cuda()
                log_probs_mb = torch.stack(log_probs)[ind]
                returns_mb = torch.stack(returns)[ind]
                advantages_mb = torch.stack(advantages)[ind]
                player1_features_mb = torch.stack(player1_features)[ind]
                player2_features_mb = torch.stack(player2_features)[ind]

                # Evaluate current policy's log prob and state values
                new_log_probs, state_values = actor_critic(states_mb, player1_features_mb, player2_features_mb)
                new_log_probs = new_log_probs.gather(1, actions_mb.unsqueeze(1)).squeeze(1)
                
                # Calculate the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(new_log_probs - log_probs_mb)
                
                # Calculate Surrogate Loss:
                surr1 = ratios * advantages_mb
                surr2 = torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip) * advantages_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns_mb - state_values).pow(2).mean()

                # Calculate gradients and perform PPO update
                optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss
                loss.backward()
                optimizer.step()

    
    def train(self, memory):
        # Extract from memory and convert to PyTorch tensors
        batch = list(zip(*memory))
        combined_states, actions, rewards, next_states, log_probs, values, dones = batch

        map_states = [s[0] for s in combined_states]  # 地图状态
        player1_features = [s[1] for s in combined_states]  # 玩家1特征
        player2_features = [s[2] for s in combined_states]  # 玩家2特征

        map_states = [torch.from_numpy(s).float().cuda() if isinstance(s, np.ndarray) else s for s in map_states]
        player1_features = [torch.from_numpy(s).float().cuda() if isinstance(s, np.ndarray) else s for s in player1_features]
        player2_features = [torch.from_numpy(s).float().cuda() if isinstance(s, np.ndarray) else s for s in player2_features]

        actions = [torch.tensor(a).long().cuda() for a in actions]
        log_probs = [torch.tensor(lp).float().cuda() for lp in log_probs]
        values = [torch.tensor(v).float().cuda() for v in values]
        rewards = [torch.tensor(r).float().cuda() for r in rewards]
        dones = [torch.tensor(d).float().cuda() for d in dones]

        # Calculate advantages
        advantages = self.compute_advantages(rewards, values, dones, gamma=self.gamma, lam=self.gae_lambda)

        # Calculate returns
        returns = []
        R = values[-1]
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)

        # Perform PPO update
        self.ppo_update(map_states, actions, log_probs, returns, advantages, player1_features, player2_features, self.actor_critic, self.optimizer, ppo_epochs=4, mini_batch_size=64, ppo_clip=self.ppo_clip)



if __name__ == "__main__":
    env = Bot(show_ui=True)
    num_inputs = env.block_num
    num_actions = env.action_space
    player_features = env.player_feature_num
    agent = PPOAgent(num_inputs, num_actions, player_features)

    state = env.reset()
    memory = deque()
    total_reward = 0
    num_episodes = 1000

    for _ in range(num_episodes):
        for _ in range(agent.ppo_steps // 2):
            action1, value1, log_prob1 = agent.select_action(state)
            memory.append((state, action1, None, None, log_prob1, value1, False))
            action2, value2, log_prob2 = agent.select_action(state)
            new_state, reward, done = env.step([action1, action2])
            memory.append((state, action2, reward, new_state, log_prob2, value2, done))
            
            state = new_state
            total_reward += reward
            
            if done:
                state = env.reset()
                total_reward = 0

        agent.train(memory)
        memory.clear()

    torch.save(agent.actor_critic.state_dict(), 'ppo_bombman.pth')
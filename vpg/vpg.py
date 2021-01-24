import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt


def build_mlp(d_in, hidden_layers, d_out, activation=nn.ReLU, logits=True):
    layers = []
    output_activation = nn.Identity if logits else activation
    layers.append(nn.Linear(d_in, hidden_layers[0], activation()))
    for i in range(len(hidden_layers) - 1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1], activation()))
    layers.append(nn.Linear(hidden_layers[-1], d_out, output_activation))

    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', render=False, seed=123,
          lr=0.001, batch_size=5000, steps=50, hidden_layers=[32]):

    env = gym.make(env_name)

    torch.manual_seed(seed)
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy_net = build_mlp(obs_dim, hidden_layers, n_acts, logits=True)
    optimizer = Adam(policy_net.parameters(), lr=lr)

    def get_action_distro(obs):
        logits = policy_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        act_distro = get_action_distro(obs)
        return act_distro.sample().item()

    def pseudo_loss(obs, acts, weights):
        log_probs = get_action_distro(obs).log_prob(acts)
        # baseline
        weights = (weights - weights.mean()) #/ weights.std()
        return - (log_probs * weights).mean()

    def train_step():

        batch_obs = []
        batch_acts = []
        batch_weights = []
        ep_rews = []
        batch_rets = []
        batch_lens = []

        obs = env.reset()

        epoch_rendered = False

        while True:
            if (not epoch_rendered) and render:
                env.render()

            batch_obs.append(obs.copy())

            # act using the policy
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # collect the data
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # record episode specific things
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # naive rewards
                # batch_weights += ep_len * [ep_ret]

                # reward-to-go
                causal_rets = [sum(ep_rews[i:]) for i in range(ep_len)]
                batch_weights += causal_rets

                obs = env.reset()
                ep_rews = []
                epoch_rendered = True
                
                if len(batch_obs) > batch_size:
                    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
                    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
                    batch_weights = torch.as_tensor(batch_weights, dtype=torch.float32)
                    break

        optimizer.zero_grad()
        loss = pseudo_loss(batch_obs, batch_acts, batch_weights)
        loss.backward()
        optimizer.step()

        return batch_rets, batch_lens, loss


    # training loop
    mean_rets = []
    for i in range(steps):
        batch_rets, batch_lens, loss = train_step()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, loss, np.mean(batch_rets), np.mean(batch_lens)))
        mean_rets.append(np.mean(batch_rets))

    plt.plot(mean_rets)
    plt.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    train(env_name=args.env_name, render=args.render, lr=args.lr, seed=args.seed)

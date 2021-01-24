import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
import pybulletgym
import matplotlib.pyplot as plt


def build_mlp(d_in, hidden_layers, d_out, activation=nn.ReLU, logits=True):
    layers = []
    output_activation = nn.Identity if logits else activation
    layers.append(nn.Linear(d_in, hidden_layers[0], activation()))
    for i in range(len(hidden_layers) - 1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1], activation()))
    layers.append(nn.Linear(hidden_layers[-1], d_out, output_activation))

    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', seed=123,
          lr=0.001, batch_size=30000, steps=50, hidden_layers=[32], gamma=0.95):

    env = gym.make(env_name)

    torch.manual_seed(seed)
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # policy net
    policy_net = build_mlp(obs_dim, hidden_layers, act_dim, logits=True)
    std = nn.Parameter(
                torch.ones(act_dim, dtype=torch.float32)
            )
    import itertools
    optimizer = Adam(itertools.chain([std], policy_net.parameters()), lr=lr)

    # value net
    value_net = build_mlp(obs_dim, hidden_layers, 1, logits=True)
    value_optimizer = Adam(value_net.parameters(), lr=lr)

    def get_action_distro(obs):
        mu = policy_net(obs)
        if len(obs.shape) == 1:
            # for single action
            sigma = std 
        else:
            # for batch actions
            sigma = std * torch.ones_like(mu)

        return Normal(mu, sigma)

    def get_action(obs):
        act_distro = get_action_distro(obs)
        return act_distro.sample().detach().numpy()

    def pseudo_loss(obs, acts, advantage):
        # sum along act vector for joint probability 
        log_probs = get_action_distro(obs).log_prob(acts).sum(axis=-1)
        advantage = torch.squeeze(advantage)
        return - (log_probs * advantage).mean()

    def value_loss(obs, value_targets):
        value_preds = value_net(obs)
        # value_targets = torch.unsqueeze(value_targets, -1)
        return nn.MSELoss(reduction='mean')(value_preds, value_targets)


    def train_step(render):
        batch_obs = []
        batch_obs_next = []
        batch_acts = []
        batch_weights = []
        batch_rews = []
        ep_rews = []
        batch_rets = []
        batch_lens = []

        if render:
            env.render()
        obs = env.reset()

        epoch_rendered = False

        while True:
            # if (not epoch_rendered) and render:
            #     env.render()

            batch_obs.append(obs.copy())

            # act using the policy
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # collect the data
            batch_acts.append(act)
            ep_rews.append(rew)
            batch_obs_next.append(obs.copy())

            if done:
                # record episode-specific things
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # rewards for each state
                batch_rews += ep_rews

                obs = env.reset()
                ep_rews = []
                epoch_rendered = True
                
                if len(batch_obs) > batch_size:
                    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
                    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
                    batch_weights = torch.as_tensor(batch_weights, dtype=torch.float32)
                    batch_rews = torch.as_tensor(batch_rews, dtype=torch.float32)
                    batch_rews = torch.unsqueeze(batch_rews, -1)
                    batch_obs_next = torch.as_tensor(batch_obs_next, dtype=torch.float32)
                    break

        # fit value function
        value_optimizer.zero_grad()
        with torch.no_grad():
            value_nexts = value_net(batch_obs_next)
        target_values = batch_rews + gamma * value_nexts
        loss_value = value_loss(batch_obs, target_values)
        loss_value.backward()
        value_optimizer.step()

        # calculate advantage
        with torch.no_grad():
            value_nexts = value_net(batch_obs_next)
            value_nows = value_net(batch_obs)
        advantage = batch_rews + gamma * value_nexts - value_nows

        # fit policy
        optimizer.zero_grad()
        loss = pseudo_loss(batch_obs, batch_acts, advantage)
        loss.backward()
        optimizer.step()

        return batch_rets, batch_lens, loss


    # training loop
    mean_rets = []
    for i in range(steps):
        if i % 10 == 0:
            render = False
            # render = True
        else:
            render = False
        batch_rets, batch_lens, loss = train_step(render)
        print('Iter: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, loss, np.mean(batch_rets), np.mean(batch_lens)))
        mean_rets.append(np.mean(batch_rets))

    plt.plot(mean_rets)
    plt.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='HalfCheetahPyBulletEnv-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    train(env_name=args.env_name, lr=args.lr, seed=args.seed, steps=args.steps)

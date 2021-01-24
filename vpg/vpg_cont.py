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
          lr=0.001, batch_size=30000, steps=50, hidden_layers=[32]):

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
            # mu, sigma = mu_sigma[0:1], 1.0
        else:
            # for batch actions
            sigma = std * torch.ones_like(mu)
            # mu, sigma = mu_sigma[:, 0], torch.ones_like(mu_sigma[:, 0])

        return Normal(mu, sigma)

    def get_action(obs):
        act_distro = get_action_distro(obs)
        return act_distro.sample().detach().numpy()

    def pseudo_loss(obs, acts, returns):
        log_probs = get_action_distro(obs).log_prob(acts).sum(axis=-1)
        # baseline
        # weights = (weights - weights.mean()) / weights.std()
        value_preds = value_net(obs).detach().squeeze()
        advantage = (returns - value_preds)
        return - (log_probs * advantage).mean()

    def value_loss(obs, rtg):
        value_preds = value_net(obs)
        rtg = torch.unsqueeze(rtg, -1)
        return nn.MSELoss(reduction='mean')(value_preds, rtg)


    def train_step(render):
        batch_obs = []
        batch_acts = []
        batch_rtg = []
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

            if done:
                # record episode-specific things
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # reward-to-go
                causal_rets = [sum(ep_rews[i:]) for i in range(ep_len)]
                batch_rtg += causal_rets

                obs = env.reset()
                ep_rews = []
                epoch_rendered = True
                
                if len(batch_obs) > batch_size:
                    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
                    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
                    batch_rtg = torch.as_tensor(batch_rtg, dtype=torch.float32)
                    break

        # fit value function
        value_optimizer.zero_grad()
        loss_value = value_loss(batch_obs, batch_rtg)
        loss_value.backward()
        value_optimizer.step()

        # fit policy
        optimizer.zero_grad()
        loss = pseudo_loss(batch_obs, batch_acts, batch_rtg)
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
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
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

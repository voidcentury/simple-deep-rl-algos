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
          lr=0.001, batch_size=4000, steps=50, hidden_layers=[32]):

    env = gym.make(env_name)

    torch.manual_seed(seed)
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # policy net
    policy_net = build_mlp(obs_dim, hidden_layers, act_dim, logits=True)
    log_std = nn.Parameter(
                torch.zeros(act_dim, dtype=torch.float32)
            )
    import itertools
    policy_optimizer = Adam(itertools.chain([log_std], policy_net.parameters()), lr=lr)

    # value net
    value_net = build_mlp(obs_dim, hidden_layers, 1, logits=True)
    value_optimizer = Adam(value_net.parameters(), lr=lr)

    mse_loss = nn.MSELoss(reduction='mean')


    def get_action_distro(obs):
        mu = policy_net(obs)
        if len(obs.shape) == 1:
            # for single action
            sigma = log_std.exp() 
        else:
            # for batch actions
            sigma = log_std.exp() * torch.ones_like(mu)

        return Normal(mu, sigma)

    def get_action(obs):
        act_distro = get_action_distro(obs)
        return act_distro.sample().detach().numpy()

    def get_advantage(batch_obs, batch_rtg):
        with torch.no_grad():
            batch_values = value_net(batch_obs).squeeze()
            batch_adv = (batch_rtg - batch_values)
            # advantage standardization
            batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()
        return batch_adv


    def fit_ppo(batch_obs, batch_acts, batch_adv, old_log_probs):
        act_distro = get_action_distro(batch_obs)
        new_log_probs = act_distro.log_prob(batch_acts).sum(axis=-1)
        entropy = act_distro.entropy().mean()
        ratio = (new_log_probs - old_log_probs).exp()

        obj1 = ratio * batch_adv
        obj2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch_adv
        loss = -torch.min(obj1, obj2).mean() - 0.001 * entropy

        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

        kl_div = (old_log_probs - new_log_probs).mean().item()
        return kl_div


    def fit_value(obs, batch_rtg):
        value_optimizer.zero_grad()
        value_preds = value_net(obs)
        batch_rtg = torch.unsqueeze(batch_rtg, -1)
        loss = mse_loss(value_preds, batch_rtg)
        loss.backward()


    def train_step(render):
        batch_obs = []
        batch_acts = []
        batch_done = []
        batch_rtg = []
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
            batch_done.append((1-done,))
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
                
                if len(batch_obs) > steps_per_epoch:
                    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
                    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
                    batch_rtg = torch.as_tensor(batch_rtg, dtype=torch.float32)
                    batch_done = torch.as_tensor(batch_done, dtype=torch.float32)
                    break

        # get current policy log probs
        with torch.no_grad():
            current_log_probs = get_action_distro(batch_obs).log_prob(batch_acts).sum(axis=-1)
        

        for i in range(num_ppo_updates):
            # fit value function
            fit_value(batch_obs, batch_rtg)

            # get advantages
            batch_adv = get_advantage(batch_obs, batch_rtg)

            # fit policy
            kl_div = fit_ppo(batch_obs, batch_acts, batch_adv, current_log_probs)

            # early stopping policy iteration
            if kl_div > 0.15 * max_kl:
                break
            

        return batch_rets, batch_lens


    # training loop
    steps_per_epoch = 4000
    num_ppo_updates = 8
    clip_param = 0.15
    max_kl = 0.01

    mean_rets = []
    try:
        for i in range(steps):
            if i % 10 == 0:
                render = False
                # render = True
            else:
                render = False
            batch_rets, batch_lens = train_step(render)
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                    (i, 0, np.mean(batch_rets), np.mean(batch_lens)))
            mean_rets.append(np.mean(batch_rets))
    except KeyboardInterrupt:
        plt.plot(mean_rets)
        plt.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    train(env_name=args.env_name, lr=args.lr, seed=args.seed, steps=args.steps)

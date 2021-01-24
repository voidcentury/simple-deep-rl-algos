import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import random


def build_mlp(d_in, hidden_layers, d_out, activation=nn.ReLU, logits=True):
    layers = []
    output_activation = nn.Identity if logits else nn.Tanh
    layers.append(nn.Linear(d_in, hidden_layers[0]))
    layers.append(activation())
    for i in range(len(hidden_layers) - 1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_layers[-1], d_out))
    layers.append(output_activation())

    return nn.Sequential(*layers)


class ReplayBuffer():
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs_buff = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs2_buff = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buff = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buff = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buff = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.current_size = 0

    def store(self, obs, obs2, act, rew, done):
        self.obs_buff[self.ptr] = obs
        self.obs2_buff[self.ptr] = obs2
        self.act_buff[self.ptr] = act
        self.rew_buff[self.ptr] = rew
        self.done_buff[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(low=0, high=self.current_size, size=batch_size)
        batch = dict(obs=self.obs_buff[idxs],
                     obs2=self.obs2_buff[idxs],
                     act=self.act_buff[idxs],
                     rew=self.rew_buff[idxs],
                     done=self.done_buff[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def train(env_name, seed=123, render=False,
          lr=0.001, steps=50, hidden_layers=[64, 64], gamma=0.995):

    env = gym.make(env_name)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)

    # env space
    obs_dim = env.observation_space.shape[0]
    assert env.action_space.shape != ()     # Continuous actions
    act_dim = env.action_space.shape[0]

    # replay buffer
    buffer = ReplayBuffer(200000, obs_dim, act_dim)

    # double Q nets
    from copy import deepcopy
    q_learner_net = build_mlp(obs_dim + act_dim, hidden_layers, 1, logits=True)
    q_target_net = deepcopy(q_learner_net)
    for param in q_target_net.parameters():
        param.requires_grad = False

    # mu as learnable argmax policy net
    mu_learner_net = build_mlp(obs_dim, hidden_layers, act_dim, logits=False)
    mu_target_net = deepcopy(mu_learner_net)
    for param in mu_target_net.parameters():
        param.requires_grad = False
    
    # learning setup
    q_optimizer = Adam(q_learner_net.parameters(), lr=lr)
    mu_optimizer = Adam(mu_learner_net.parameters(), lr=lr)
    loss_func = nn.SmoothL1Loss(reduction='mean')


    def get_action(obs, epsilon=0.01):
        # for single action policy rollout
        assert len(obs.shape) == 1
        with torch.no_grad():
            toss = torch.rand(1).numpy()[0]
            if toss > epsilon:
                act = mu_learner_net(obs)
                act = torch.clamp(act, min=-1.0, max=1.0).numpy()
            else:
                act = env.action_space.sample()
        return act


    def update_target_net(polyak_tau=0.999):
        with torch.no_grad():
            for p, p_target in zip(
                    q_learner_net.parameters(), q_target_net.parameters()
                ):
                p_target.data.mul_(polyak_tau)
                p_target.data.add_((1 - polyak_tau) * p.data)

    
    def update_mu_target_net(polyak_tau=0.999):
        with torch.no_grad():
            for p, p_target in zip(
                    mu_learner_net.parameters(), mu_target_net.parameters()
                ):
                p_target.data.mul_(polyak_tau)
                p_target.data.add_((1 - polyak_tau) * p.data)

    
    def fit_q():
        batch = buffer.sample_batch(batch_size=batch_size)
        batch_obs, batch_obs2 = batch['obs'], batch['obs2']
        batch_act, batch_rew, batch_done = batch['act'], batch['rew'], batch['done']
        
        # calculating: Q_target( state2, mu_target(s2) )
        with torch.no_grad():
            batch_act2 = mu_target_net(batch_obs2)
            max_q_prime = q_target_net(torch.cat([batch_obs2, batch_act2], dim=1))
        
        # calculating targets
        shape = (batch_size, 1)
        assert batch_rew.shape == max_q_prime.shape == (1-batch_done).shape == shape
        q_target = batch_rew + gamma * ((1 - batch_done) * max_q_prime)
        assert q_target.shape == shape

        # predictions
        q_pred = q_learner_net(torch.cat([batch_obs, batch_act], dim=1))

        # fit q
        q_optimizer.zero_grad()
        loss = loss_func(q_pred, q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_learner_net.parameters(), 10.0)
        q_optimizer.step()

        # fit mu
        mu_optimizer.zero_grad()
        batch_act_mu = mu_learner_net(batch_obs)
        for param in q_learner_net.parameters():
            param.requires_grad = False
        q = q_learner_net(torch.cat([batch_obs, batch_act_mu], dim=1))
        for param in q_learner_net.parameters():
            param.requires_grad = True
        mu_loss = -1.0 * q.mean()
        mu_loss.backward()
        mu_optimizer.step()

        return loss


    # training loop
    ep_rets = []
    batch_size = 64
    epsilon = 1.0
    epsilon_decay = 0.995
    polyak_tau = 0.995
    episodes = steps
    update_every = 4
    error = 0
    plot_shown = False

    try:

        for ep in range(episodes):
            obs = env.reset()
            ret = 0

            for step in range(1000):
                obs1 = obs.copy()
                    
                # act using some policy
                act = get_action(torch.as_tensor(obs, dtype=torch.float32), epsilon)
                obs, rew, done, _ = env.step(act)
                obs2 = obs.copy()
                ret += rew

                # collect data into buffer
                buffer.store(obs1, obs2, act, rew, done)

                # update networks
                if (step + 1) % update_every == 0:
                    if buffer.current_size >= 1000:
                        for j in range(1):
                            error = fit_q()
                            update_target_net(polyak_tau)
                            update_mu_target_net(polyak_tau)

                if done:
                    ep_rets.append(ret)
                    break

            if (ep + 1) % 1 == 0:
                epsilon = max(epsilon * epsilon_decay, 0.01)

            if ep % 100 == 0:
                print('Episode: %3d \t loss: %.3f \t Avg return: %.3f \t epsilon: %.3f \t buffer: %.3f'%
                        (ep + 1, error, np.mean(ep_rets[-100:]), epsilon, buffer.current_size))

    
    except KeyboardInterrupt:
        plt.plot(ep_rets)
        plt.show()
        plot_shown = True
    
    if not plot_shown:
        plt.plot(ep_rets)
        plt.show()

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--steps', type=int, default=5000000)
    args = parser.parse_args()
    train(env_name=args.env_name, lr=args.lr, seed=args.seed, steps=args.steps, render=args.render)

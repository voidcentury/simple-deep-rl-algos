import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial
from torch.optim import Adam
import numpy as np
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt


def build_mlp(d_in, hidden_layers, d_out, activation=nn.ReLU, logits=True):
    layers = []
    output_activation = nn.Identity if logits else activation
    layers.append(nn.Linear(d_in, hidden_layers[0]))
    layers.append(activation())
    for i in range(len(hidden_layers) - 1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_layers[-1], d_out))
    layers.append(output_activation())

    return nn.Sequential(*layers)


class ReplayBuffer():
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs_buff = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs2_buff = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buff = np.zeros((capacity, 1), dtype=np.int64)
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

    # env space
    obs_dim = env.observation_space.shape[0]
    assert env.action_space.shape == ()     # Discrete actions
    n_acts = env.action_space.n

    # replay buffer
    buffer = ReplayBuffer(50000, obs_dim)

    # double Q nets
    from copy import deepcopy
    q_learner_net = build_mlp(obs_dim, hidden_layers, n_acts, logits=True)
    q_target_net = deepcopy(q_learner_net)
    for param in q_target_net.parameters():
        param.requires_grad = False
    
    # learning setup
    q_optimizer = Adam(q_learner_net.parameters(), lr=lr)
    # loss_func = nn.MSELoss(reduction='mean')
    loss_func = nn.SmoothL1Loss(reduction='mean')


    def get_action(obs, epsilon=0.01):
        # epsilon-greedy
        with torch.no_grad():
            action_values = q_learner_net(obs)
            if len(obs.shape) == 1:
                # for single action policy rollout
                action = torch.argmax(action_values)
                act_probs = torch.ones((n_acts)) * epsilon / (n_acts-1)
                act_probs[action] = 1 - epsilon
                act_distro = Multinomial(1, probs=act_probs)
                action = torch.argmax(act_distro.sample())
                action = action.numpy()
            else:
                # for batch actions
                action = torch.argmax(action_values, dim=1, keepdim=True)
        return action


    def update_target_net(polyak_tau=0.999):
        with torch.no_grad():
            for p, p_target in zip(
                    q_learner_net.parameters(), q_target_net.parameters()
                ):
                p_target.data.mul_(polyak_tau)
                p_target.data.add_((1 - polyak_tau) * p.data)

    
    def fit_q():
        batch = buffer.sample_batch(batch_size=batch_size)
        batch_obs, batch_obs2 = batch['obs'], batch['obs2']
        batch_act, batch_rew, batch_done = batch['act'], batch['rew'], batch['done']
        
        # calculating: Q_target( state2, argmax a2: Q_real(s2, a2) )
        argmax_act2 = get_action(batch_obs2)
        with torch.no_grad():
            values_act2_target = q_target_net(batch_obs2)
            max_value2_target = values_act2_target.gather(dim=1, index=argmax_act2)

        # calculating targets
        shape = (batch_size, 1)
        assert batch_rew.shape == max_value2_target.shape == (1-batch_done).shape == shape
        target_values = batch_rew + gamma * ((1 - batch_done) * max_value2_target)
        assert target_values.shape == shape

        # predictions
        assert batch_act.shape == shape
        all_action_values = q_learner_net(batch_obs)
        actions = batch_act.type(torch.int64)
        action_values = all_action_values.gather(dim=1, index=actions)
        assert action_values.shape == shape

        # fit
        q_optimizer.zero_grad()
        loss = loss_func(action_values, target_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_learner_net.parameters(), 10.0)
        q_optimizer.step()

        return loss


    # training loop
    ep_rets = []
    batch_size = 64
    epsilon = 1 - 1/n_acts
    epsilon_decay = 0.995
    polyak_tau = 0.995
    plot_shown = False

    try:
        episodes = steps
        update_every = 4
        error = 0

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
    parser.add_argument('--env_name', '--env', type=str, default='LunarLander-v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=456)
    parser.add_argument('--steps', type=int, default=5000000)
    args = parser.parse_args()
    train(env_name=args.env_name, lr=args.lr, seed=args.seed, steps=args.steps, render=args.render)

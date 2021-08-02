"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import random
import numpy as np
import time

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

NUM_Actor = 10
Good_Actor = 7
Choose_actors = [1,1,1,1,1,1,1,0,0,0]
Gloab_credit = np.array([0.0]*NUM_Actor)

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.actor_id = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

        # bandit parameters:
        self.bandit_credit = 0.0
        self.bandit_learning_rate = 0.01
        self.bandit_e = 0.5

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            real_ep_r = 0.

            while True:
                if self.name == 'w00':
                    # self.env.render()
                    pass
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, real_r, done, _ = self.env.step(a)

                if done: real_r = -1

                if self.actor_id in [7, 8, 9]:
                    r = -real_r
                else:
                    r = real_r

                ep_r += r
                real_ep_r += real_r

                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if self.actor_id in [7, 8, 9]:
                    time.sleep(10)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

            # 更新该worker的置信度
            self.bandit_credit = self.bandit_credit + self.bandit_learning_rate * (real_ep_r - self.bandit_credit)

            # 更新 actor 的选择
            if random.random() >= (self.bandit_e / self.g_ep.value):
                topk_action_id = np.argsort(-Gloab_credit)[:Good_Actor]
            else:
                topk_action_id = random.sample([i for i in range(NUM_Actor)], Good_Actor)

            # for action in range(NUM_Actor):
            #     if action in topk_action_id:
            #         Choose_actors[action] = 0
            #     else:
            #         Choose_actors[action] = 0

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    # CPU_NUM = mp.cpu_count()
    CPU_NUM = NUM_Actor
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(CPU_NUM)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    # [w.join() for w in workers]
    # [w.close() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

"""
for discrete action.
and there are bad actors.
algorithm: e-greedy worker-selection policy
"""
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record,push
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import random
import numpy as np
import time
from a3c.NN import DiscreteNet


os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 5000

NUM_Actor = 10
bad_worker_id = [1,6,9]
Good_Actor = NUM_Actor-len(bad_worker_id)

env_name = 'CartPole-v0'
env = gym.make(env_name)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, global_Choose_actors, global_credit, average_reward):
        super(Worker, self).__init__()
        self.actor_id = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = DiscreteNet(N_S, N_A)  # local network
        self.env = gym.make(env_name)
        self.gca, self.gc = global_Choose_actors, global_credit
        self.ar = average_reward
        self.good = False if self.actor_id in bad_worker_id else True

        # bandit parameters:
        self.bandit_credit = self.gc[self.actor_id].value
        self.bandit_learning_rate = 0.01
        self.bandit_e = 0.5
        self.real_reward_list = []

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            real_ep_r = 0.

            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, real_r, done, _ = self.env.step(a)

                if done: real_r = -1

                if self.actor_id in bad_worker_id:
                    r = real_r
                else:
                    r = real_r

                ep_r += r
                real_ep_r += real_r

                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                # print(self.gca[1].value)
                if self.gca[self.actor_id].value == 0:
                # if self.actor_id in bad_worker_id:
                    print('lock',self.actor_id)
                    with global_ep.get_lock():
                        if global_ep.value >= MAX_EP:
                            break
                    time.sleep(0.1*total_step)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    # if actor is bad, they only push but not pull
                    # if self.actor_id in bad_worker_id:
                    if self.gca[self.actor_id] == 0:
                        # push(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                        pass
                    else:
                        if self.actor_id in bad_worker_id:
                            push(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA,self.good)
                        else:
                            push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a,
                                          buffer_r, GAMMA,  self.good)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, real_ep_r, self.res_queue, self.name)
                        break
                s = s_
                if self.actor_id in bad_worker_id:
                    total_step += 1
                else:
                    total_step += 1

            # 更新该worker的置信度
            # self.bandit_credit = self.bandit_credit + self.bandit_learning_rate * (real_ep_r - self.bandit_credit)
            self.bandit_credit = self.bandit_credit + self.bandit_learning_rate * (self.ar[self.actor_id].value - self.bandit_credit)
            self.real_reward_list.append(real_ep_r)
            with self.ar[self.actor_id].get_lock():
                self.ar[self.actor_id].value = sum(self.real_reward_list)*1.0/len(self.real_reward_list)
            with self.gc[self.actor_id].get_lock():
                self.gc[self.actor_id].value = self.bandit_credit
            # print([self.gc[i].value for i in range(NUM_Actor)])

            # 更新 actor 的选择
            if random.random() >= (self.bandit_e / self.g_ep.value):
            #if random.random() >= max(self.bandit_e - self.g_ep.value*0.002, 0.05):
                t_list = [self.gc[i].value for i in range(NUM_Actor)]
                topk_action_id = np.argsort(-np.array(t_list), kind="heapsort")[:Good_Actor]
            else:
                topk_action_id = random.sample([i for i in range(NUM_Actor)], Good_Actor)

            # topk_action_id = [0,1,2,3,4,5,6]
            # only let actor 0 to update the global actor chosen list.
            if self.actor_id == 0:
                print('topk_action_id', topk_action_id)
                for action in range(NUM_Actor):
                    if action in topk_action_id:
                        self.gca[action].value= 1
                    else:
                        self.gca[action].value = 0
                self.gca[0].value=1
        self.res_queue.put(None)


if __name__ == "__main__":
    # 这个是全局变量用于选择actor，1表示actor被选择，0表示未被选择。最后一位 global_Choose_actors[NUM_Actor] 表示是否拿到全局锁
    global_Choose_actors = [mp.Value('i', 0) for i in range(NUM_Actor + 1)]
    global_Choose_actors[0].value = 1
    # 全局的信用分配
    global_credit = [mp.Value('f', 40) for i in range(NUM_Actor)]
    # average reward for each worker
    average_reward = [mp.Value('f', 0) for i in range(NUM_Actor)]

    gnet = DiscreteNet(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    # CPU_NUM = mp.cpu_count()
    CPU_NUM = NUM_Actor
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i,global_Choose_actors,global_credit,average_reward) for i in range(CPU_NUM)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    # [w.join() for w in workers]
    [w.terminate() for w in workers]

    t_list = [global_credit[i].value for i in range(NUM_Actor)]
    print('Global_credit', t_list)
    topk_action_id = np.argsort(-np.array(t_list), kind="heapsort")
    print('sort credit', topk_action_id)
    rewards_for_each_worker = [average_reward[i].value for i in range(NUM_Actor)]
    print(rewards_for_each_worker)

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
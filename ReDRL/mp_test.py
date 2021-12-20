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
from a3c.NN import DiscreteNet


os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

NUM_Actor = 10
Good_Actor = 5
bad_worker_id = [1,3,5,8,9]

env_name = 'CartPole-v0'
env = gym.make(env_name)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

# 这个是全局变量用于选择actor，1表示actor被选择，0表示未被选择。最后一位 global_Choose_actors[NUM_Actor] 表示是否拿到全局锁
global_Choose_actors = [mp.Value('i', 1) for i in range(NUM_Actor + 1)]
# 全局的信用分配
Global_credit = [mp.Value('f', 0) for i in range(NUM_Actor + 1)]


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.actor_id = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = DiscreteNet(N_S, N_A)  # local network
        self.env = gym.make(env_name)

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
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, real_r, done, _ = self.env.step(a)

                if done: real_r = -1

                # 有问题的进程，其奖励返回不准确
                if self.actor_id in bad_worker_id:
                    r = -real_r
                else:
                    r = real_r

                ep_r += r
                real_ep_r += real_r

                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                # print(global_Choose_actors[1].value)
                if global_Choose_actors[self.actor_id].value == 0:
                    # print('lock')
                    with global_ep.get_lock():
                        if global_ep.value >= MAX_EP:
                            break
                    time.sleep(100)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, real_ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

            # 更新该worker的置信度
            self.bandit_credit = self.bandit_credit + self.bandit_learning_rate * (real_ep_r - self.bandit_credit)
            with Global_credit[NUM_Actor].get_lock():
                Global_credit[self.actor_id].value = self.bandit_credit

            # 更新 actor 的选择
            if random.random() >= (self.bandit_e / self.g_ep.value):
            #if random.random() >= max(self.bandit_e - self.g_ep.value*0.002, 0.05):
                t_list = [Global_credit[i].value for i in range(NUM_Actor)]
                # print(t_list)
                topk_action_id = np.argsort(-np.array(t_list), kind="heapsort")[:Good_Actor]
            else:
                topk_action_id = random.sample([i for i in range(NUM_Actor)], Good_Actor)

            # topk_action_id = [0,1,2,3,4,5,6]
            with global_Choose_actors[NUM_Actor].get_lock():
                # print('topk_action_id', topk_action_id)
                for action in range(NUM_Actor):
                    if action in topk_action_id:
                        global_Choose_actors[action].value = 1
                    else:
                        global_Choose_actors[action].value = 0

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = DiscreteNet(N_S, N_A)  # global network
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
    [w.terminate() for w in workers]

    t_list = [Global_credit[i].value for i in range(NUM_Actor)]
    print('Global_credit', t_list)
    topk_action_id = np.argsort(-np.array(t_list), kind="heapsort")
    print('sort credit', topk_action_id)

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
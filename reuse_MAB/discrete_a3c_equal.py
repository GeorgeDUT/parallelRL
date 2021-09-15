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

UPDATE_GLOBAL_ITER = 32
GAMMA = 0.9
MAX_EP = 10000
each_test_episodes = 70  # 每轮训练，异步跑的共同的episode

NUM_Actor = 10
Good_Actor_num = 7
bad_worker_id = [2, 9, 5]

env_name = 'CartPole-v0'
env = gym.make(env_name)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, stop_episode, res_queue, name):
        super(Worker, self).__init__()
        self.actor_id = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.stop_episode = stop_episode
        self.gnet, self.opt = gnet, opt
        self.lnet = DiscreteNet(N_S, N_A)  # local network
        self.lnet.load_state_dict(self.gnet.state_dict())
        self.env = gym.make(env_name)

    def run(self):
        total_step = 1
        real_g_ep = -1
        total_ep = 0
        global_ep_list = []
        for i in range(self.stop_episode):
            time.sleep(0.5)
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                real_g_ep = self.g_ep.value
                global_ep_list.append(real_g_ep)
            total_ep += 1
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            real_ep_r = 0.

            while True:
                # print(self.name, total_step)
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, real_r, done, _ = self.env.step(a)

                if done: real_r = -1

                # 有问题的进程，其奖励返回不准确
                if self.actor_id in bad_worker_id:
                    # r = real_r + (random.random()-0.5)/0.5*20
                    r = -real_r

                else:
                    r = real_r

                ep_r += r
                real_ep_r += real_r

                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        with self.g_ep_r.get_lock():
                            if self.g_ep_r.value == 0.:
                                self.g_ep_r.value = real_ep_r
                            else:
                                self.g_ep_r.value = self.g_ep_r.value * 0.99 + real_ep_r * 0.01
                        self.res_queue.put(self.g_ep_r.value)
                        # print(
                        #     self.name,
                        #     "Ep:", real_g_ep,
                        #     "| Ep_r: %.0f" % self.g_ep_r.value,
                        # )
                        break
                s = s_
                total_step += 1
        print('close',self.name,total_ep,global_ep_list)
        self.res_queue.put(None)


def evaluate_network(g_net, evaluate_num):
    env = gym.make(env_name)
    sum_r = 0
    for i in range(evaluate_num):
        s = env.reset()
        while True:
            a = g_net.choose_action(v_wrap(s[None, :]))
            s, real_r, done, _ = env.step(a)
            sum_r += real_r
            if done:
                break
    return sum_r / evaluate_num


def linear_decay(min_v, max_v, cur_step, totoal_step):
    return max_v - (max_v - min_v) * (cur_step / totoal_step)


if __name__ == "__main__":
    gnet = DiscreteNet(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    myPool = mp.Pool(NUM_Actor)
    res = []  # record episode reward to plot
    # 初始化置信度
    worker_credit = [0] * NUM_Actor
    for i in range(int(MAX_EP / each_test_episodes)):
        # if i == 17:
        #     print()
        # 按照置信度进行贪心选择
        # if random.random() >= (0.5 / (global_ep.value + 1e-10)):
        if random.random() >= linear_decay(0.1, 0.9, global_ep.value, MAX_EP):
            topk_action_id = np.argsort(-np.array(worker_credit[1:]), kind="heapsort")[:Good_Actor_num - 1]
            topk_action_id += 1
        else:
            topk_action_id = random.sample([index for index in range(1, NUM_Actor)], Good_Actor_num - 1)
        topk_action_id = np.insert(topk_action_id, 0, 0, axis=0)
        # 选出来的worker进行异步更新
        workers = [Worker(gnet, opt, global_ep, global_ep_r, int(each_test_episodes/Good_Actor_num), res_queue, w_id) for w_id in
                   topk_action_id]
        random.shuffle(workers)
        [w.start() for w in workers]
        # 接收本轮奖励
        none_count = 0
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                none_count += 1
                if none_count >= Good_Actor_num:
                    break
        [w.join() for w in workers]
        # [w.close() for w in workers]
        # 运用0号worker进评估
        eval_reward = evaluate_network(gnet, 5)
        print('evaluate reward',eval_reward)
        # 使用评估结果 更新worker的置信度
        for id in topk_action_id:
            worker_credit[id] += 0.01 * (eval_reward - worker_credit[id])
        print(i, topk_action_id, worker_credit)

    print('worker_credit', worker_credit)
    topk_action_id = np.argsort(-np.array(worker_credit), kind="heapsort")
    print('sort credit num', topk_action_id)

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

"""该轮中，我们会固定每个进程跑一定轮数，严格限制每个进程的贡献"""
import multiprocessing as mp
import time

import gym
import numpy as np

from utils import v_wrap, push_and_pull
from a3c.NN import DiscreteNet, DiscreteCNN, ContinuousNet


def gym_rgb2gray(rgb):
    grey = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114) / 255
    grey = np.expand_dims(grey, 0)
    return grey


class DiscreteWorker(mp.Process):
    def __init__(self, config, worker_id, gnet, opt, res_queue):
        super(DiscreteWorker, self).__init__()
        self.worker_id = worker_id
        self.name = 'w%02i' % worker_id
        self.res_queue = res_queue
        self.gnet, self.opt = gnet, opt
        if config.image_input:
            self.lnet = DiscreteCNN(config.obs_dim, config.action_dim)
        elif config.discrete:
            self.lnet = DiscreteNet(config.obs_dim, config.action_dim)
        self.env = gym.make(config.env_name)
        self.max_run_episode = config.worker_run_episode
        self.cur_episode = 0
        self.config = config

    def run(self):
        total_step = 1
        episode_reword_list = []
        while self.cur_episode < self.max_run_episode:
            self.cur_episode += 1
            s = self.env.reset()
            if self.config.image_input:
                s = gym_rgb2gray(s)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if self.config.image_input:
                    s_ = gym_rgb2gray(s_)
                if done: r = -1
                if self.worker_id in self.config.bad_worker_list:
                    fake_r = -r
                ep_r += r
                buffer_a.append(a)
                if self.config.image_input:
                    buffer_s.append(s_-s)
                else:
                    buffer_s.append(s)
                if self.worker_id in self.config.bad_worker_list:
                    buffer_r.append(fake_r)
                else:
                    buffer_r.append(r)
                if total_step % self.config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # if done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r,
                                  self.config.GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        # print(self.name)
                        time.sleep(0.01)  # 使得各个线程均衡
                        episode_reword_list.append(ep_r)
                        if len(episode_reword_list) % 20 == 0:
                            print(self.name, len(episode_reword_list), max(episode_reword_list), episode_reword_list)
                        break
                s = s_
                total_step += 1
        # print(self.name, self.cur_episode)
        # print(episode_reword_list)
        last_n_r = episode_reword_list[len(episode_reword_list) - 100:]
        print(self.name, 'recent reward:', sum(last_n_r) / len(last_n_r))
        self.res_queue.put(sum(last_n_r) / len(last_n_r))


class ContinuousWorker(mp.Process):
    def __init__(self, config, worker_id, gnet, opt, res_queue):
        super(ContinuousWorker, self).__init__()
        self.worker_id = worker_id
        self.name = 'w%02i' % worker_id
        self.res_queue = res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = ContinuousNet(config.obs_dim, config.action_dim)
        self.env = gym.make(config.env_name)
        self.max_run_episode = config.worker_run_episode
        self.cur_episode = 0
        self.config = config

    def run(self):
        total_step = 1
        episode_reword_list = []
        while self.cur_episode < self.max_run_episode:
            self.cur_episode += 1
            s = self.env.reset()
            if self.config.image_input:
                s = gym_rgb2gray(s)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a.clip(self.config.action_low, self.config.action_high))
                if self.config.image_input:
                    s_ = gym_rgb2gray(s_)
                if done: r = -1
                if self.worker_id in self.config.bad_worker_list:
                    fake_r = -r
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                if self.worker_id in self.config.bad_worker_list:
                    buffer_r.append(fake_r)
                else:
                    buffer_r.append(r)
                if total_step % self.config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # if done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r,
                                  self.config.GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        # print(self.name)
                        time.sleep(0.01)  # 使得各个线程均衡
                        episode_reword_list.append(ep_r)
                        # if len(episode_reword_list) % 20 == 0:
                        #     print(self.name, len(episode_reword_list), max(episode_reword_list), episode_reword_list)
                        break
                s = s_
                total_step += 1
        # print(self.name, self.cur_episode)
        # print(episode_reword_list)
        last_n_r = episode_reword_list[len(episode_reword_list) - 100:]
        print(self.name, 'recent reward:', sum(last_n_r) / len(last_n_r))
        self.res_queue.put(sum(last_n_r) / len(last_n_r))

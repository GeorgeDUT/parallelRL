from utils import v_wrap, push_and_pull
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import random
import numpy as np
import time
import psutil
from a3c.NN import ContinuousNet

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
bandit_learning_rate = 0.01
GAMMA = 0.9
MAX_EP = 6000

NUM_Actor = 10
num_running_actor = 5
bad_worker_id_list = [0, 1, 2]
Global_credit = [mp.Value('f', 0) for i in range(NUM_Actor + 1)]

env_name = 'Pendulum-v0'
env = gym.make(env_name)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

update_process_pid = mp.Value('i', 0)
pid_list = []  # worker's pid, to suspend/resume process
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.actor_id = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = ContinuousNet(N_S, N_A)  # local network
        self.env = gym.make(env_name)

    def run(self):
        psutil.Process(self.pid).suspend()
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            real_ep_r = 0.

            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, real_r, done, _ = self.env.step(a.clip(-2, 2))

                if done: real_r = -1

                # 有问题的进程，其奖励返回不准确
                if self.actor_id in bad_worker_id_list:
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
                        record(self.g_ep, self.g_ep_r, real_ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

            # 更新该worker的置信度
            with Global_credit[NUM_Actor].get_lock():
                Global_credit[self.actor_id].value += (real_ep_r - Global_credit[
                    self.actor_id].value) * bandit_learning_rate

        self.res_queue.put(None)


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
        if global_ep.value % 10 == 0 or global_ep.value > MAX_EP:
            if psutil.pid_exists(update_process_pid.value):
                psutil.Process(update_process_pid.value).resume()
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


class UpdateProcess(mp.Process):
    def __init__(self):
        super(UpdateProcess, self).__init__()
        self.choose_times_count = [0]*NUM_Actor

    def run(self) -> None:
        bandit_e = 0.5
        while True:
            print('run')
            # print(global_ep)
            # 更新 actor 的选择
            #if random.random() >= bandit_e / (global_ep.value + 0.0001):
            if random.random() >= max(0.6 - global_ep.value * 0.0001, 0.1):
                t_list = [Global_credit[i].value for i in range(NUM_Actor)]
                # this kind of sort method is unstable, each time could choose different equal item
                topk_action_id = np.argsort(-np.array(t_list), kind="heapsort")[:num_running_actor]
            else:
                topk_action_id = random.sample([i for i in range(NUM_Actor)], num_running_actor)
            if global_ep.value > MAX_EP:
                # when game go to end, do not suspend any process, to avoid dead block
                topk_action_id = range(NUM_Actor)
            worker_id_list = list(range(NUM_Actor))
            random.shuffle(worker_id_list)
            for worker_id in worker_id_list:
                if worker_id in topk_action_id:
                    self.choose_times_count[worker_id] += 1
                    if psutil.pid_exists(pid_list[worker_id]):
                        psutil.Process(pid_list[worker_id]).resume()
                        print('open', worker_id)
                else:
                    if psutil.pid_exists(pid_list[worker_id]):
                        psutil.Process(pid_list[worker_id]).suspend()
                        print('close', worker_id)
            if global_ep.value > MAX_EP:
                break
            psutil.Process(self.pid).suspend()
        print(self.choose_times_count)

if __name__ == "__main__":
    gnet = ContinuousNet(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer

    # parallel training
    # CPU_NUM = mp.cpu_count()
    CPU_NUM = NUM_Actor
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(CPU_NUM)]
    [w.start() for w in workers]
    pid_list = [w.pid for w in workers]

    update_process = UpdateProcess()
    update_process.start()
    update_process_pid.value = update_process.pid
    psutil.Process(update_process_pid.value).resume()

    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    # update_process.close()

    t_list = [Global_credit[i].value for i in range(NUM_Actor)]
    print('Global_credit', t_list)
    topk_action_id = np.argsort(-np.array(t_list), kind="heapsort")
    print('sort credit', topk_action_id)

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

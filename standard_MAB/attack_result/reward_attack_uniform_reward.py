import copy
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(r'D:\RL\parallelRL')

import random
import time
import argparse
import xlwt
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from a3c.NN import DiscreteNet
from standard_MAB.my_utils import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        if not self.log.closed:
            self.log.write(message)

    def flush(self):
        pass

    def close_file_output(self):
        self.log.close()


attack_type = ['uniform_reward', 'same_reward']
test_type = ['all', 'all_good', 'al', 'rand_choice']


def gen_args():
    args = argparse.ArgumentParser()
    args = args.parse_args()

    args.env_name = 'CartPole-v0'
    # args.env_name = 'LunarLander-v2'
    env = gym.make(args.env_name)
    args.N_S = env.observation_space.shape[0]
    args.N_A = env.action_space.n

    args.UPDATE_GLOBAL_ITER = 5  # 32
    args.GAMMA = 0.9
    args.MAX_EP = 25000
    args.MAX_STEP = 2000
    args.each_test_episodes = 100  # 每轮训练，异步跑的共同的episode
    args.ep_sleep_time = 0.5  # 每轮跑完以后休息的时间，用以负载均衡

    args.NUM_Actor = 10
    args.Good_Actor_num = 7
    args.evaluate_epoch = 5

    args.reward_attack_type = 'uniform_reward'
    args.cur_test_type = 'al'
    assert args.cur_test_type in test_type, args.cur_test_type + ' not in ' + str(test_type)
    args.bad_worker_id = random.sample(range(1, 10), 3) if args.cur_test_type!='all_good' else []
    args.base_path = './' + args.reward_attack_type + '_' + args.cur_test_type
    args.save_path = make_training_save_path(args.base_path)

    return args


class Worker(mp.Process):
    def __init__(self, params, gnet, opt, global_ep, global_ep_r, stop_episode, res_queue, name, my_credit):
        super(Worker, self).__init__()
        self.params = params
        self.actor_id = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.stop_episode = stop_episode
        self.gnet, self.opt = gnet, opt
        self.lnet = DiscreteNet(params.N_S, params.N_A)  # local network
        if self.actor_id not in self.params.bad_worker_id:
            self.lnet.load_state_dict(self.gnet.state_dict())
        self.env = gym.make(params.env_name)
        self.my_credit = my_credit

    def run(self):
        total_step = 1
        real_g_ep = -1
        total_ep = 0
        while self.g_ep.value < self.stop_episode:
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                real_g_ep = self.g_ep.value
            total_ep += 1
            if self.params.ep_sleep_time > 0:
                time.sleep(self.params.ep_sleep_time)
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            real_ep_r = 0.

            ep_step = 0
            constant_r = random.randint(0, 200)
            print('constant_r', constant_r)
            while True:
                ep_step += 1
                if ep_step > self.params.MAX_STEP:
                    print(self.name, "up to max step ,force stop this episode", total_ep)
                    break
                # print(self.name, total_step)
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, real_r, done, _ = self.env.step(a)

                if self.actor_id in self.params.bad_worker_id:
                    if self.params.reward_attack_type == 'uniform_reward':
                        r = random.randint(0, 200)
                    elif self.params.reward_attack_type == 'same_reward':
                        r = constant_r
                    else:
                        raise NotImplemented('not support reward attack type {}'.format(self.params.reward_attack_type))
                else:
                    r = real_r

                ep_r += r
                real_ep_r += real_r

                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.params.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if self.actor_id in self.params.bad_worker_id:
                        # bad bandit update is same with good bandit
                        push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r,
                                      self.params.GAMMA)
                    else:
                        # sync
                        push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r,
                                      self.params.GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        constant_r = random.randint(0, 200)
                        print('constant_r', constant_r)
                        # 更新当前臂的奖励
                        self.my_credit.value += 0.01 * (real_ep_r - self.my_credit.value)
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
        # print('close',self.name,total_ep)
        self.res_queue.put(None)


def test_runner(params_func):
    analyse_data = {"bad_id": [], "bandit_credit": [], "sorted_id": []}
    for test in range(10):
        s_time = time.time()
        params = params_func()
        if not os.path.exists(params.save_path):
            os.makedirs(params.save_path)
        save_config(params, params.save_path)
        sys.stdout = Logger(os.path.join(params.save_path, 'log.txt'), sys.stdout)
        print('test session num {}, start time {}'.format(test, str(time.asctime(time.localtime(time.time())))))
        print('bad worker id list:', params.bad_worker_id)
        gnet = DiscreteNet(params.N_S, params.N_A)  # global network
        gnet.share_memory()  # share the global parameters in multiprocessing
        opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
        global_credit = [mp.Value('d', 0.) for i in range(params.NUM_Actor)]

        res = []  # record episode reward to plot
        evaluate_good_value_list = []
        # 初始化置信度
        # worker_credit = [0] * params.NUM_Actor
        is_random_choice = False
        random_choice_info = {True: 'random choice', False: 'Greedy choice'}
        evaluate_reward_list = []
        step_list = []
        for i in range(int(params.MAX_EP / params.each_test_episodes)):
            # if random.random() >= (0.5 / (global_ep.value + 1e-10)):
            if params.cur_test_type == 'al' and random.random() >= linear_decay(0.1, 1, global_ep.value, 15000):
                # if random.random() >= 0.5:
                is_random_choice = False
                worker_credit = [ele.value for ele in global_credit]
                topk_action_id = np.argsort(-np.array(worker_credit[1:]), kind="heapsort")[:params.Good_Actor_num - 1]
                topk_action_id += 1
                # print(topk_action_id, worker_credit)
            else:
                is_random_choice = True
                topk_action_id = random.sample([index for index in range(1, params.NUM_Actor)],
                                               params.Good_Actor_num - 1)
            topk_action_id = np.insert(topk_action_id, 0, 0, axis=0)
            # 选出来的worker进行异步更新
            workers = [
                Worker(params, gnet, opt, global_ep, global_ep_r, (i + 1) * params.each_test_episodes, res_queue, w_id,
                       global_credit[w_id]) for w_id in topk_action_id]
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
                    if none_count >= params.Good_Actor_num:
                        break
            [w.join() for w in workers]
            # [w.close() for w in workers]
            # 运用0号worker进评估
            eval_reward = evaluate_network_normal(params.env_name, gnet, params.evaluate_epoch)
            evaluate_reward_list.append(eval_reward)
            step_list.append(i)
            # 使用评估结果 更新worker的置信度
            # for id in topk_action_id:
            #     worker_credit[id] += 0.01 * (eval_reward - last_evaluate - worker_credit[id])
            last_evaluate = eval_reward
            worker_credit = [ele.value for ele in global_credit]
            print('run_count:', i, 'eval_reward', eval_reward, 'choose_type:', random_choice_info[is_random_choice])
            print('choose arms:', topk_action_id, 'cur_worker_credit:', [round(ele, 4) for ele in worker_credit])
            # 达到预期奖励进行早停
            # if eval_reward > 100:
            #     evaluate_good_value_list.append(eval_reward)
            #     if len(evaluate_good_value_list) > 10:
            #         break

        print('last worker_credit', worker_credit)
        sorted_id = np.argsort(-np.array(worker_credit), kind="heapsort")
        print('sorted credit num', sorted_id)

        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(params.save_path, 'mv_reward.png'))
        plt.close()

        plt.plot(evaluate_reward_list)
        plt.ylabel('evaluate reward')
        plt.xlabel('Step*{}'.format(params.each_test_episodes))
        plt.savefig(os.path.join(params.save_path, 'evaluate_reward.png'))
        plt.close()

        print('test session num {}, end time {}'.format(test, str(time.asctime(time.localtime(time.time())))))
        print('bad worker list: {}, spend time: {}'.format(params.bad_worker_id, time.time() - s_time))
        analyse_data["bad_id"].append(params.bad_worker_id)
        analyse_data["bandit_credit"].append(worker_credit)
        analyse_data["sorted_id"].append(sorted_id)
        # 存储奖励曲线csv
        pd.DataFrame({"Epochs": step_list, "Reward": evaluate_reward_list}).to_csv(params.save_path + '/reward.csv')
        # 关闭本次log文件输入
        sys.stdout.close_file_output()
        # 最后写出多轮测试数据到excel
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("analyse_data_result")
        for i in range(len(analyse_data["bad_id"])):
            worksheet.write(i, 0, str(analyse_data["bad_id"][i]))
            worksheet.write(i, 1, str(analyse_data["bandit_credit"][i]))
            worksheet.write(i, 2, str(analyse_data["sorted_id"][i]))
        workbook.save(os.path.join(params.save_path, 'summary.xls'))


if __name__ == '__main__':
    test_runner(gen_args)

"""
@Author: 
@Time: 2021/9/16
功能说明: 
"""
import os
import gym
import yaml

from utils import v_wrap


def evaluate_network(env_name, g_net, evaluate_num):
    env = gym.make(env_name)
    sum_r = 0
    for i in range(evaluate_num):
        s = env.reset()
        while True:
            a = g_net.choose_action(v_wrap(s[None, :]))
            s, real_r, done, _ = env.step(a)
            if done:
                real_r = -50
            sum_r += real_r
            if done:
                break
    return sum_r / evaluate_num


def linear_decay(min_v, max_v, cur_step, totoal_step):
    return max_v - (max_v - min_v) * (min(cur_step, totoal_step) / totoal_step)


def make_training_save_path(base_path):
    """
    根据训练指定的存储根目录，生成其合适的子目录，例如：run1，run2，...
    :param base_path:训练存储的根目录
    :return: sub_path
    """
    listdir = os.listdir(base_path)
    num = [int(dir.split('run')[-1]) for dir in listdir if "run" in dir]
    if len(num) == 0:
        return os.path.join(base_path, "run0")
    else:
        return os.path.join(base_path, "run" + str(max(num) + 1))

def save_config(args, save_path):
    """
    将 argparse 定义的参数进行存储
    :param args: 参数对象
    :param save_path: 保存的根路径
    :return:
    """
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file, sort_keys=False)
    file.close()
import gym
import argparse
from algorithm.mab import *
import multiprocessing as mp
from a3c.NN import DiscreteCNN
from a3c.Worker import DiscreteWorker
from shared_adam import SharedAdam

def define_args():
    parser = argparse.ArgumentParser(description='params config for MAB Algorithm')
    # parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--env_name', type=str, default='PongDeterministic-v4')
    parser.add_argument('--discrete', type=bool, default=True)
    parser.add_argument('--image_input', type=bool, default=True)

    args = parser.parse_args()
    # a3c配置
    args.UPDATE_GLOBAL_ITER = 16
    args.GAMMA = 0.9
    args.worker_run_episode = 2500
    # mab配置
    args.good_worker_list = [0, 2, 3, 4, 6, 7, 8]
    args.bad_worker_list = [1, 5, 9]
    # 针对离散的情况
    env = gym.make(args.env_name)
    args.obs_dim = env.observation_space.shape[0:2]
    args.action_dim = env.action_space.n
    return args

config = define_args()

def get_reward(choosed_acttor_id_list):
    gnet = DiscreteCNN(config.obs_dim, config.action_dim)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [DiscreteWorker(config, id, gnet, opt, res_queue) for id in choosed_acttor_id_list]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        res.append(r)
        if len(res) == len(choosed_acttor_id_list):
            break
    avg_r = sum(res)/len(res)
    print('avg r', avg_r)
    return avg_r


if __name__ == '__main__':
    args = define_args()
    cucb = Greedy(10, 7)

    for i in range(30):
        actor_list = cucb.choose_supper_actors()
        random.shuffle(actor_list)
        print('train times', i, 'super bandit:', actor_list)
        reward = get_reward(actor_list)
        cucb.update_credit(actor_list, reward)

    print(cucb.get_optimal_actors())
    print(cucb.get_bad_actors())
    print(cucb.credit)
    print(cucb.choosen_times)
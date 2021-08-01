import numpy as np
import math
import random


class CUCB():
    def __init__(self,actor_num,good_actor):
        self.actor_num = actor_num
        self.credit = np.array([0.0] * actor_num)
        self.mu = np.array([0.0] * actor_num)
        self.t = 1
        self.choosen_times = np.array([1] * actor_num)
        self.alpha = 0.01
        self.good_actor = good_actor

    def update_credit(self,actor_list,reward):
        for i in actor_list:
            self.credit[i] = self.credit[i] + self.alpha*(reward-self.credit[i])

    def choose_supper_actors(self):

        # 更新所有的臂的采样次数
        self.t = self.t + 1

        for i in range(len(self.mu)):
            self.mu[i] = self.credit[i] + (math.log(self.t)/self.choosen_times[i]) ** 0.5

        # 取前k大的值对应的actor的id
        # topk_action_id = np.argsort(-self.mu)[:self.good_actor]
        # topk_action_id = random.sample([i for i in range(10)],self.good_actor)
        b = [i for i in range(self.actor_num)]
        random.shuffle(b)
        topk_action_id = np.lexsort((-self.mu,b))[:self.good_actor]

        # 更新被选择的actor的采样次数
        for id in topk_action_id:
            self.choosen_times[id] = self.choosen_times[id] + 1

        return topk_action_id

    def get_optimal_actors(self):
        # return random.sample([i for i in range(10)],7)
        return np.argsort(-self.credit)[:self.good_actor]

    def get_bad_actors(self):
        return np.argsort(self.credit)[:self.actor_num-self.good_actor]


class Greedy():
    def __init__(self,actor_num,good_actor):
        self.actor_num = actor_num
        self.credit = np.array([0.0] * actor_num)
        self.alpha = 0.01
        self.e = 0.5
        self.t = 1
        self.good_actor = good_actor
        self.choosen_times = np.array([1] * actor_num)

    def update_credit(self,actor_list,reward):
        for i in actor_list:
            self.credit[i] = self.credit[i] + self.alpha*(reward-self.credit[i])

    def choose_supper_actors(self):

        if random.random()>=(self.e/self.t):
            topk_action_id = np.argsort(-self.credit)[:self.good_actor]
        else:
            topk_action_id = random.sample([i for i in range(self.actor_num)], self.good_actor)

        # 更新被选择的actor的采样次数
        for id in topk_action_id:
            self.choosen_times[id] = self.choosen_times[id] + 1

        return topk_action_id

    def get_optimal_actors(self):
        # return random.sample([i for i in range(10)],7)
        return np.argsort(-self.credit)[:self.good_actor]

    def get_bad_actors(self):
        return np.argsort(self.credit)[:self.actor_num-self.good_actor]


if __name__ == "__main__":
    cucb = Greedy(10,5)

    for i in range(500):
        actor_list = cucb.choose_supper_actors()
        if (1 in actor_list) or (3 in actor_list) or (5 in actor_list) or (7 in actor_list) or (9 in actor_list):
        # if (9 in actor_list):
            reward = -1
        else:
            reward = 1

        cucb.update_credit(actor_list,reward)

    print(cucb.get_optimal_actors())
    print(cucb.get_bad_actors())
    print(cucb.credit)
    print(cucb.choosen_times)

import gym
import multiprocessing as mp

def test_discrete_env(env_name):
    env = gym.make(env_name)
    obs_s = env.observation_space
    action_s = env.action_space
    print(obs_s)
    print(action_s)
    for i in range(10):
        s = env.reset()
        done = False
        while not done:
            env.render()
            s_, r, done, _ = env.step(action_s.sample())
            print(r)
    env.close()

if __name__ == '__main__':
    # discrete
    env_name = 'Pendulum-v0'
    test_discrete_env(env_name)
    #
    # env_name = 'CartPole-v1'
    # test_discrete_env(env_name)
    # # continuous
    env_name = 'BipedalWalker-v3'
    test_discrete_env(env_name)

    # image input
    # env_name = 'Breakout-v0'
    env_name = 'BreakoutDeterministic-v4'
    # test_discrete_env(env_name)
    # from gym import envs
    # print(envs.registry.all())
    w = [mp.Process(target=test_discrete_env,args=(env_name,)) for i in range(2)]
    [i.start() for i in w]
    [i.join() for i in w]
"""
@Author: 
@Time: 2021/9/24
功能说明: 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_many_lines():
    # base_path = './test_plot/'
    base_path = './plot_LunarLander-v2_all_good/'
    num_list = range(0, 10)
    csv_path_list = [base_path + 'run' + str(num) + '/reward.csv' for num in num_list]
    class_name_list = ['test' + str(num) for num in num_list]
    data = pd.DataFrame()
    for index, csv_path in enumerate(csv_path_list):
        t_df = pd.read_csv(csv_path, delim_whitespace=False)
        # 增加一列种类
        t_df['Name'] = [class_name_list[index]] * t_df.shape[0]
        data = pd.concat([data, t_df], ignore_index=True)
        # data = data.append(t_df)
    data['Epochs'] = data["Epochs"] * 100
    # 绘制最终结果
    sns.set(font_scale=3.5)

    # colors=["red","windows blue","grey","green"]
    # colors=["green","red","blue","black"]
    colors = ["green", "red", "blue", "black", "grey", "pink", "orange", "purple", "brown", "yellow"]
    colors = colors[:len(num_list)]

    plt.figure(dpi=80, figsize=(15, 9))
    g = sns.lineplot(x="Epochs", y="Reward", hue="Name",
                     style="Name", data=data, ci=100, linewidth=3.0,
                     palette=sns.xkcd_palette(colors))
    # g.set(xscale="log")

    plt.legend(loc='upper left', fontsize=12)

    # plt.subplots_adjust(top=0.98,bottom=0.155,left=0.09,right=0.99,
    #                     hspace=0,wspace=0)
    plt.subplots_adjust(top=0.98, bottom=0.155, left=0.12, right=0.975,
                        hspace=0, wspace=0)

    plt.savefig("plot_reward.pdf")
    plt.show()


def plot_fuse_pic():
    figure_name = "plot_reward.pdf"
    # """subtract constant3"""
    # # path_list = ['./test_results/subtract/' + name for name in ['plot_origin_al_constant3/', 'plot_rand_constant3/',
    # #                                                    'plot_reward_rand_action_to_grad/', 'plot_substract_al_constant3/',
    # #                                                    'plot_substract_constant1/']]
    # # path_list.append('./test_results/all_good/')
    # # path_list.append('./plot_worker_evaluate/')
    # # line_name = ['origin_al_constant3', 'rand_constant3', 'rand_action_to_grad_constant1', 'subtract_constant3',
    # #              'subtract_constant1', 'all_good', 'worker_evaluate']
    # path_list = ['./test_results/subtract/' + name for name in ['plot_rand_constant3/']]
    # path_list.append('./test_results/all_good/')
    # path_list.append('./plot_worker_evaluate/')
    # line_name = ['rand_constant3', 'all_good', 'worker_evaluate']
    """basic origin"""
    # path_list = ['./test_results/origin/' + name for name in ['plot_reward_al/', 'plot_reward_rand/',
    #                                                           'plot_reward_al_2g/', 'plot_reward_rand_2g/',
    #                                                           'plot_reward_al_-50/', 'plot_reward_rand_-50/',
    #                                                           'plot_reward_al_constant/', 'plot_reward_rand_constant/']]
    # path_list.append('./test_results/all_good/')
    # line_name = ['basic_al', 'basic_rand', 'al_2g', 'al_rand', 'al_-50', 'rand_-50', 'al_constant', 'rand_constant', 'all_good']
    """paper result1"""
    # path_list = ['./paper_result/plot_worker_evaluate' + name for name in ['/', '_all/', '_all_good/', '_rand/', '_all_1e5/', '_rand_1e5/']]
    # line_name = ['algorithm', 'all_bandits', 'all_good_bandits', 'rand_choice', 'all_1e5', 'rand_1e5']
    """LunarLander"""
    path_list = ['./' + name for name in ['plot_LunarLander-v2_all_good/', 'plot_LunarLander-v2_rand_choice/', 'plot_LunarLander-v2_rand_grad/']]#, 'plot_LunarLander-v2_constant3/']]
    line_name = ['all_good', 'rand_choice', 'al']#, 'constant3']
    # HalfCheetah
    # path_list = ['./' + name for name in ['plot_HalfCheetah-v2_all_good/', 'plot_HalfCheetah-v2_rand_choice/', 'plot_HalfCheetah-v2/']]
    # line_name = ['all_good', 'rand_choice', 'al']  # , 'constant3']
    colors = ["green", "red", "blue", "black", "pink", "orange", "purple", "brown", "yellow"]
    colors = colors[:len(path_list)]
    data = pd.DataFrame()
    for line_num, base_path in enumerate(path_list):
        num_list = range(0, 10)
        csv_path_list_1 = [base_path + 'run' + str(num) + '/reward.csv' for num in num_list]
        for index, csv_path in enumerate(csv_path_list_1):
            t_df = pd.read_csv(csv_path, delim_whitespace=False)
            # 增加一列种类
            t_df['Name'] = [line_name[line_num]] * t_df.shape[0]
            # data = data.append(t_df)
            data = pd.concat([data, t_df], ignore_index=True)

    data['Epochs'] = data["Epochs"] * 100
    # 绘制最终结果
    sns.set(font_scale=3.5)

    # colors=["red","windows blue","grey","green"]
    # colors=["green","red","blue","black"]
    plt.figure(dpi=80, figsize=(15, 9))
    g = sns.lineplot(x="Epochs", y="Reward", hue="Name",
                     style="Name", data=data, ci=100, linewidth=3.0,
                     palette=sns.xkcd_palette(colors))
    # g.set(xscale="log")

    plt.legend(loc='upper left', fontsize=12)

    # plt.subplots_adjust(top=0.98,bottom=0.155,left=0.09,right=0.99,
    #                     hspace=0,wspace=0)
    plt.subplots_adjust(top=0.98, bottom=0.155, left=0.12, right=0.975,
                        hspace=0, wspace=0)

    plt.savefig(figure_name)
    plt.show()


if __name__ == '__main__':
    # plot_many_lines()
    plot_fuse_pic()

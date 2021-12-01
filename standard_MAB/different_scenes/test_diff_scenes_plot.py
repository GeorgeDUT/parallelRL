"""
@Author: 
@Time: 2021/9/24
功能说明: 
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_many_lines(data_path, figure_name):
    """
    绘制某个路径下，10轮比赛数据的曲线图
    :param data_path: 某种设置下，10轮独立运行的数据根目录，如：./al/constant[1]_al/
    :param figure_name:保存的图片名，通常为xxx.pdf
    :return:
    """
    num_list = range(0, 10)
    csv_path_list = [data_path + 'run' + str(num) + '/reward.csv' for num in num_list]
    class_name_list = ['run' + str(num) for num in num_list]
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

    plt.savefig(figure_name)
    plt.show()


def plot_fuse_pic():
    """绘制多种参属下，不同参数的对比曲线"""
    """constant_rand_choice"""
    # figure_name = "const_attack_rand_choice.pdf"
    # params = ['0', '1', '2', '3', '5', '-1', '-2', '-3', '-5']
    # path_list = ['./rand_choice/constant[{}]_rand_choice/'.format(p) for p in params]
    # line_name = ['constant[{}]'.format(p) for p in params]
    """constant_al"""
    # figure_name = "const_attack_al.pdf"
    # params = ['0', '1', '2', '3', '5', '-1', '-2', '-3', '-5']
    # path_list = ['./al/constant[{}]_al/'.format(p) for p in params]
    # line_name = ['constant[{}]'.format(p) for p in params]
    """uniform and gaussian rand choice"""
    # figure_name = "uniform_gaussian_rand_choice.pdf"
    # params = ['0, 1', '0, 2', '-1, 0', '-2, 0', '-2, 2']
    # path_list = ['./rand_choice/uniform[{}]_rand_choice/'.format(p) for p in params]
    # line_name = ['uniform[{}]'.format(p) for p in params]
    # path_list.append('./gaussian[]_rand_choice/')
    # line_name.append('gaussian')
    """uniform and gaussian al"""
    # figure_name = "uniform_gaussian_al.pdf"
    # params = ['0, 1', '0, 2', '-1, 0', '-2, 0', '-2, 2']
    # path_list = ['./al/uniform[{}]_al/'.format(p) for p in params]
    # line_name = ['uniform[{}]'.format(p) for p in params]
    # path_list.append('./al/gaussian[]_al/')
    # line_name.append('gaussian')
    """ diff scenes HalfCHeetah-v2"""
    # figure_name = "HalfCHeetah.pdf"
    # path_list = ['./HalfCHeetah-v2/constant1_al/', './HalfCHeetah-v2/gaussian_al/', './HalfCHeetah-v2/gaussian_all/',
    #              './HalfCHeetah-v2/gaussian_rand_choice/', './HalfCHeetah-v2/uniform[-1,1]_al/',
    #              './HalfCHeetah-v2/uniform[-1,1]_all/', './HalfCHeetah-v2/uniform[-1,1]_rand_choice/']
    # line_name = [p.split('/')[-2] for p in path_list]

    """ diff scenes LunarLander-v2"""
    figure_name = "LunarLander.pdf"
    path_list = ['./LunarLander-v2/constant1_al/', './LunarLander-v2/gaussian_al/', './LunarLander-v2/gaussian_all/',
                 './LunarLander-v2/gaussian_rand_choice/', './LunarLander-v2/uniform[-1,1]_al/',
                 './LunarLander-v2/uniform[-1,1]_all/', './LunarLander-v2/uniform[-1,1]_rand_choice/']
    line_name = [p.split('/')[-2] for p in path_list]

    colors = ["green", "red", "blue", "black", "pink", "orange", "purple", "brown", "yellow"]
    colors = colors[:len(path_list)]
    data = pd.DataFrame()
    for line_num, base_path in enumerate(path_list):
        num_list = range(0, 10)
        csv_path_list_1 = [base_path + 'run' + str(num) + '/reward.csv' for num in num_list]
        csv_path_list_1 = csv_path_list_1[0: min(10, len(os.listdir(base_path)))]
        if len(os.listdir(base_path)) < 10:
            print('warning {} has only {} test times'.format(base_path, len(os.listdir(base_path))))
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

def plot_whole_conditions(base_path, figure_name, line_count=5):
    """
    绘制某个目录下，所有参数设定的结果
    :param base_path:数据根目录，可以是 ./al/ 或./all/ 或./rand_choice/
    :param figure_name: 图表名
    :param line_count: 每幅图画的数据个数，这里由于颜色区分度问题，最大取10，即每10个设定画成一幅图，会生成多幅图
    :return:
    """
    conditions = os.listdir(base_path)
    print([c for c in conditions])
    print([len(os.listdir(os.path.join(base_path, c))) for c in conditions])
    all_path_list = [os.path.join(base_path, c) for c in conditions]
    i = 0
    while i < len(all_path_list):
        path_list = all_path_list[i: min(i+line_count, len(all_path_list))]
        line_name = conditions[i: min(i+line_count, len(all_path_list))]
        colors = ["green", "red", "blue", "black", "pink", "orange", "purple", "brown", "yellow"]
        colors = colors[:len(path_list)]
        data = pd.DataFrame()
        for line_num, base_path in enumerate(path_list):
            num_list = range(0, len(os.listdir(base_path)))
            print(base_path, [len(os.listdir(base_path + '/run' + str(num))) for num in num_list])
            csv_path_list_1 = [base_path + '/run' + str(num) + '/reward.csv' for num in num_list]
            if len(num_list) < 10:
                print('warning {} has only {} test times'.format(base_path, len(num_list)))
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

        plt.savefig("{}_".format(int(i/line_count))+figure_name)
        plt.show()
        i += line_count


if __name__ == '__main__':
    # 函数使用示例
    # plot_many_lines('./al/constant[1]_al/', 'constant[1]_al.pdf')
    plot_fuse_pic()
    # plot_whole_conditions("./al/", "total_al.pdf", 8)
    # plot_whole_conditions("./all/", "total_all.pdf", 8)
    # plot_whole_conditions("./rand_choice/", "total_rand_choice.pdf", 8)

"""
@Author: 
@Time: 2021/9/24
功能说明: 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_many_lines():
    base_path = './plot_reward/'
    num_list = range(0, 10)
    csv_path_list = [base_path + 'run' + str(num) + '/reward.csv' for num in num_list]
    class_name_list = ['test' + str(num) for num in num_list]
    data = pd.DataFrame()
    for index, csv_path in enumerate(csv_path_list):
        t_df = pd.read_csv(csv_path, delim_whitespace=False)
        # 增加一列种类
        t_df['Name'] = [class_name_list[index]] * t_df.shape[0]
        data = data.append(t_df)
    data['Epochs'] = data["Epochs"] * 100
    # 绘制最终结果
    sns.set(font_scale=3.5)

    # colors=["red","windows blue","grey","green"]
    # colors=["green","red","blue","black"]
    colors = ["green", "red", "blue", "black", "grey", "pink", "orange", "purple", "brown", "yellow"]

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
    base_path_1 = './plot_reward_al/'
    num_list_1 = range(0, 10)
    csv_path_list_1 = [base_path_1 + 'run' + str(num) + '/reward.csv' for num in num_list_1]
    data = pd.DataFrame()
    for index, csv_path in enumerate(csv_path_list_1):
        t_df = pd.read_csv(csv_path, delim_whitespace=False)
        # 增加一列种类
        t_df['Name'] = ['our_algorithm'] * t_df.shape[0]
        data = data.append(t_df)

    base_path_2 = './plot_reward_rand/'
    num_list_2 = range(0, 10)
    csv_path_list_2 = [base_path_2 + 'run' + str(num) + '/reward.csv' for num in num_list_2]
    for index, csv_path in enumerate(csv_path_list_2):
        t_df = pd.read_csv(csv_path, delim_whitespace=False)
        # 增加一列种类
        t_df['Name'] = ['rand_choice'] * t_df.shape[0]
        data = data.append(t_df)

    data['Epochs'] = data["Epochs"] * 100
    # 绘制最终结果
    sns.set(font_scale=3.5)

    # colors=["red","windows blue","grey","green"]
    # colors=["green","red","blue","black"]
    # colors = ["green", "red", "blue", "black", "grey", "pink", "orange", "purple", "brown", "yellow"]
    colors = ["red", "blue"]

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


if __name__ == '__main__':
    plot_fuse_pic()
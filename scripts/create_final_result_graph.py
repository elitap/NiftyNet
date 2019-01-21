import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def getColormap():
    return np.array([[0.4, 0.7607843137254902, 0.6470588235294118, 1.],
                     [0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.],
                     [0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.],
                     [0.9058823529411765, 0.5411764705882353, 0.7647058823529411, 1.],
                     [0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.],
                     [1.0, 0.8509803921568627, 0.1843137254901961, 1.],
                     [0.8980392156862745, 0.7686274509803922, 0.5803921568627451, 1.],
                     [0.6, 0.6, 0.6, 1.],
                     [0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.]])


   # np.array([[0.7019607843137254, 0.8862745098039215, 0.803921568627451],
   #           [0.9921568627450981, 0.803921568627451, 0.6745098039215687],
   #           [0.796078431372549, 0.8352941176470589, 0.9098039215686274],
   #           [0.9568627450980393, 0.792156862745098, 0.8941176470588236],
   #           [0.9019607843137255, 0.9607843137254902, 0.788235294117647],
   #           [1.0, 0.9490196078431372, 0.6823529411764706],
   #           [0.9450980392156862, 0.8862745098039215, 0.8],
   #           [0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.]])


def createbarplot(csvfile):
    df = pd.read_csv(csvfile)

    temp_df = df.drop(['std-95hd','mean-95hd', 'std-dice'], axis=1)
    df2plot_mean_dice = temp_df.pivot(index='organ', columns='Groups', values='mean-dice')

    cols = df2plot_mean_dice.columns.tolist()
    print(cols)
    cols.insert(len(cols)-1, cols.pop(cols.index('SJTU')))
    cols.insert(len(cols)-1, cols.pop(cols.index('SU')))
    cols.insert(len(cols), cols.pop(cols.index('UMIT')))


    temp_df = df.drop(['std-95hd', 'mean-95hd', 'mean-dice'], axis=1)
    df2plot_std_dice = temp_df.pivot(index='organ', columns='Groups', values='std-dice')

    temp_df = df.drop(['std-dice', 'mean-dice', 'std-95hd'], axis=1)
    df2plot_mean_95hd = temp_df.pivot(index='organ', columns='Groups', values='mean-95hd')


    temp_df = df.drop(['std-dice', 'mean-dice', 'mean-95hd'], axis=1)
    df2plot_std_95hd = temp_df.pivot(index='organ', columns='Groups', values='std-95hd')


    df2plot_mean_dice = df2plot_mean_dice.reindex(columns=cols)
    df2plot_std_dice = df2plot_std_dice.reindex(columns=cols)
    df2plot_mean_95hd = df2plot_mean_95hd.reindex(columns=cols)
    df2plot_std_95hd = df2plot_std_95hd.reindex(columns=cols)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = getColormap()
    df2plot_mean_dice.plot.bar(yerr=df2plot_std_dice, ax=ax, ylim=(0.1, 0.975), rot=0, width=0.7, color=colors)


    ax.grid(True, axis="y", linestyle='--', linewidth=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(cols))
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_label_text('Dice value')

    fig.savefig("../results/result_figures/dice_barplot_ea.eps", dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 5))
    df2plot_mean_95hd.plot.bar(yerr=df2plot_std_95hd, ax=ax, rot=0, width=0.7, ylim=(0, 15), color=colors)

    ax.grid(True, axis="y", linestyle='--', linewidth=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(cols))
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_label_text('95% HD [mm]')

    fig.savefig("../results/result_figures/95hd_barplot_ead.eps", dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--rescsv',
                        required=True,
                        help=("dataset path")
                        )

    args = parser.parse_args()
    createbarplot(args.rescsv)
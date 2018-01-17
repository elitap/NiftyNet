import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np


def createbarplot(csvfile):
    df = pd.read_csv(csvfile)

    temp_df = df.drop(['std-95hd','mean-95hd', 'std-dice'], axis=1)
    df2plot_mean_dice = temp_df.pivot(index='organ', columns='Groups', values='mean-dice')

    cols = df2plot_mean_dice.columns.tolist()
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
    df2plot_mean_dice.plot.bar(yerr=df2plot_std_dice, ax=ax, ylim=(0.1, 0.975), rot=0, width=0.7)

    ax.grid(True, axis="y", linestyle='--', linewidth=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(cols))
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_label_text('Dice value')

    fig.savefig("../../CARS2018/fig/dice_barplot.eps", dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 5))
    df2plot_mean_95hd.plot.bar(yerr=df2plot_std_95hd, ax=ax, rot=0, width=0.7)

    ax.grid(True, axis="y", linestyle='--', linewidth=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(cols))
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_label_text('95% HD [mm]')

    fig.savefig("../../CARS2018/fig/95hd_barplot.eps", dpi=150, bbox_inches='tight')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--rescsv',
                        required=True,
                        help=("dataset path")
                        )

    args = parser.parse_args()
    createbarplot(args.rescsv)
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

BOXPLOT_FILE = "%s/output/bb_%s.eps"
LINEPLOT_FILE = "%s/output/line_%s.eps"
COMBINED_LINEPLOT_FILE = "line_%d.eps"

def create_boxplots(result_file):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint', 'File'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]

    for model in models:
        df_to_plot = grouped_df[model].reset_index().drop('File', axis=1)
        df_to_plot.boxplot(by='Checkpoint', figsize=(14, 4.8))
        plt.title("organ mean dice")
        filename = BOXPLOT_FILE % (model, model)
        print filename
        plt.savefig(filename, dpi=100)
        plt.close()

def create_lineplots(result_file):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]

    for model in models:
        df_to_plot = grouped_df[model].reset_index()
        df_to_plot.plot(style='.-', x='Checkpoint', y='Dice', figsize=(14, 4.8))
        plt.title("sample and organ mean dice")
        filename = LINEPLOT_FILE % (model, model)
        print filename
        plt.savefig(filename, dpi=100)
        plt.close()

def combined_lineplot(models, grouped_df, id, ax):
    for model in models:
        df_to_plot = grouped_df[model].reset_index()
        df_to_plot.plot(style='.-', x='Checkpoint', y='Dice', ax=ax[id], label=model)
    ax[id].legend(loc='upper right', bbox_to_anchor=(0.95, 1.2), ncol=4)

def create_lineplot(result_file):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]
    idx_list = range(0,25,12)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    fig.suptitle("grouped by checkpoint - file and organ based dice mean")
    for cnt in range(0, len(idx_list)-1):
        combined_lineplot(models[idx_list[cnt]:idx_list[cnt+1]], grouped_df, cnt, ax)

    fig.savefig("combined_lineplot.eps", dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--result',
                        required=True
                        )

    args = parser.parse_args()
    #create_boxplots(args.result)
    #create_lineplots(args.result)
    create_lineplot(args.result)
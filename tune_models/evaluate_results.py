import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

BOXPLOT_FILE = "%s/output/bb_%s.eps"
LINEPLOT_FILE = "%s/output/line_%s.eps"
COMBINED_LINEPLOT_FILE = "line_%d.eps"

BOXPLOT_CHECKPOINT_FILE = "%s/output/%d/bb_%s.eps"
LINEPLOT_COMBINED = "combined_lineplot.png"

def create_boxplots_organ_avg(result_file):
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

def create_lineplots_organ_avg(result_file):
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

def create_lineplot_organ_samp_avg(result_file):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]
    idx_list = range(0,25,12)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    fig.suptitle("grouped by checkpoint - file and organ based dice mean")
    for cnt in range(0, len(idx_list)-1):
        combined_lineplot(models[idx_list[cnt]:idx_list[cnt+1]], grouped_df, cnt, ax)

    fig.savefig(LINEPLOT_COMBINED, dpi=100)

def create_boxplot(result_file, model, checkpoint):
    full_df = pd.read_csv(result_file)
    selected_df = full_df[(full_df['Model']==model) & (full_df['Checkpoint']==checkpoint)]
    selected_df.drop(['Model','Checkpoint','File'], axis=1, inplace=True)
    selected_df.boxplot(by='Organ', figsize=(14, 4.8))
    plt.savefig(BOXPLOT_CHECKPOINT_FILE % (model, checkpoint, model))
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--result',
                        required=True
                        )

    args = parser.parse_args()
    #create_boxplots(args.result)
    #create_lineplots(args.result)
    #create_lineplot_organ_samp_avg(args.result)
    create_boxplot(args.result, "half_e-3_48-8_dice_50k_1024s", 42000)
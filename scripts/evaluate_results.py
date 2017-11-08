import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

MODEL_BASE_PATH = "tune_models%s"

BOXPLOT_FILE = "../"+MODEL_BASE_PATH+"/%s/output/bb_%s_orig_size.png"
LINEPLOT_FILE = "../"+MODEL_BASE_PATH+"%s/output/line_%s_orig_size.png"
COMBINED_LINEPLOT_FILE = "../"+MODEL_BASE_PATH+"/line_%d_orig_size.png"

FILTERED_LINEPLOT_FILE = "../"+MODEL_BASE_PATH+"/%s.png"

BOXPLOT_CHECKPOINT_FILE = "../"+MODEL_BASE_PATH+"/%s/output/%d/bb_%s_orig_size.png"
LINEPLOT_COMBINED = "../"+MODEL_BASE_PATH+"/combined_lineplot_orig_size.png"

def create_boxplots_organ_avg(result_file, stage):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint', 'File'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]

    for model in models:
        df_to_plot = grouped_df[model].reset_index().drop('File', axis=1)
        df_to_plot.boxplot(by='Checkpoint', figsize=(14, 4.8))
        plt.title("organ mean dice")
        filename = BOXPLOT_FILE % (stage, model, model)
        print filename
        plt.savefig(filename, dpi=100)
        plt.close()

def create_lineplots_organ_avg(result_file, stage):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]

    for model in models:
        df_to_plot = grouped_df[model].reset_index()
        df_to_plot.plot(style='.-', x='Checkpoint', y='Dice', figsize=(14, 4.8))
        plt.title("sample and organ mean dice")
        filename = LINEPLOT_FILE % (stage, model, model)
        print filename
        plt.savefig(filename, dpi=100)
        plt.close()


def create_lineplot_organ_samp_avg_model_filtered(result_file, stage, filters, threshold=0.2):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])['Dice'].mean() #mean over organs

    grouped_df_to_thres = full_df.groupby(['Model'])['Dice'].mean()
    models = grouped_df_to_thres[grouped_df_to_thres > threshold].index.tolist()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.2), ncol=4)
    for model in models:
        if np.all([filter in model for filter in filters]):
            df_to_plot = grouped_df[model].reset_index()
            df_to_plot.plot(style='.-', x='Checkpoint', y='Dice', ax=ax, label=model)
    fig.suptitle("sample and organ mean dice with average dice vals over %f" % threshold)

    filename = FILTERED_LINEPLOT_FILE % (stage, '-'.join(filters))
    print filename
    fig.savefig(filename, dpi=100)


def combined_lineplot(models, grouped_df, id, ax):
    for model in models:
        df_to_plot = grouped_df[model].reset_index()
        df_to_plot.plot(style='.-', x='Checkpoint', y='Dice', ax=ax[id], label=model)
    ax[id].legend(loc='upper right', bbox_to_anchor=(0.95, 1.2), ncol=4)


def create_lineplot_organ_samp_avg(result_file, stage):
    full_df = pd.read_csv(result_file)
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])['Dice'].mean() #mean over organs
    models = grouped_df.index.levels[0]
    model_index = range(0,3,2)
    print model_index, models

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    fig.suptitle("grouped by checkpoint - file and organ based dice mean")
    for cnt in range(0, len(model_index)-1):
        combined_lineplot(models[model_index[cnt]:model_index[cnt+1]], grouped_df, cnt, ax)

    fig.savefig((LINEPLOT_COMBINED % stage), dpi=100)

def create_boxplot(result_file, model, checkpoint, stage):
    full_df = pd.read_csv(result_file)
    selected_df = full_df[(full_df['Model']==model) & (full_df['Checkpoint']==checkpoint)]
    selected_df.drop(['Model','Checkpoint','File'], axis=1, inplace=True)
    selected_df.boxplot(by='Organ', figsize=(14, 4.8))
    plt.savefig(BOXPLOT_CHECKPOINT_FILE % (stage, model, checkpoint, model))
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--result',
                        required=True
                        )
    parser.add_argument('--stage',
                        default=1,
                        type=int
                        )

    args = parser.parse_args()
    stage = "_2nd" if args.stage == 2 else ""
    result = (args.result % stage)
    #create_boxplots_organ_avg(result, stage)
    #create_lineplots(result)
    #create_lineplot_organ_samp_avg(result, stage)
    create_lineplot_organ_samp_avg_model_filtered(result, stage, ['dice','96-1'])
    #create_boxplot(result, "h_e-3_48-8_d_42k__full_e-3_24-24_dice_4096s", 222000, stage)
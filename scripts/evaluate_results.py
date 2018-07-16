import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import defs
from defs import LABELS

MODEL_BASE_PATH = "tune_models%s"

BOXPLOT_FILE = "../"+MODEL_BASE_PATH+"/%s/output/bb_%s_orig_size.png"
LINEPLOT_FILE = "../"+MODEL_BASE_PATH+"%s/output/line_%s_orig_size.png"
COMBINED_LINEPLOT_FILE = "../"+MODEL_BASE_PATH+"/line_%d_orig_size.png"
FILTERED_LINEPLOT_FILE = "../"+MODEL_BASE_PATH+"/figures/%s_%1.2f.png"
BOXPLOT_CHECKPOINT_FILE = "../"+MODEL_BASE_PATH+"/%s/output/%d/bb_%s_orig_size.png"
LINEPLOT_COMBINED = "../"+MODEL_BASE_PATH+"/combined_lineplot_orig_size.png"

BIG_ORGANS = ["BrainStem", "Mandible", "Parotid_L", "Parotid_R"]
SMALL_ORGANS = ["Chiasm", "OpticNerve_L", "OpticNerve_R"]



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


def create_lineplot_organ_samp_avg_model_filtered(result_file, stage, filters, value='Dice', threshold=0.4):

    full_df = pd.read_csv(result_file)
    full_df_big_organs = full_df[full_df['Organ'].isin(BIG_ORGANS)]
    full_df_small_organs = full_df[full_df['Organ'].isin(SMALL_ORGANS)]

    grouped_df_big_organs = full_df_big_organs.groupby(['Model', 'Checkpoint'])[value].mean()
    grouped_df_small_organs = full_df_small_organs.groupby(['Model', 'Checkpoint'])[value].mean()
    grouped_df = full_df.groupby(['Model', 'Checkpoint'])[value].mean()

    grouped_df_to_thres = full_df.groupby(['Model'])[value].mean()
    models = grouped_df_to_thres[grouped_df_to_thres > threshold].index.tolist()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 10))

    #fig_split, ax_split = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    #ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.2), ncol=4)
    df_data = []
    df_data_big = []
    df_data_small = []
    for cnt, model in enumerate(models):
        if np.all([filter in model for filter in filters]):
            grouped_df[model].reset_index().plot(style='.-', x='Checkpoint', y=value, ax=ax[0], label=model, ylim=(0, 0.85))
            grouped_df_big_organs[model].reset_index().plot(style='.-', x='Checkpoint', y=value, ax=ax[1], label=model+"_big_org", ylim=(0, 0.9))
            grouped_df_small_organs[model].reset_index().plot(style='.-', x='Checkpoint', y=value, ax=ax[2], label=model + "_small_org", ylim=(0, 0.9))
            #modelname = str(model).replace('_1024s', '').replace('half', 'h').replace('full', 'f').replace(
            #    'quarter', 'q').replace('e-', '')
            #df_data.append({'model': modelname, value: float(grouped_df[model].reset_index()[value])})
            #df_data_big.append({'model': modelname+"_b", value: float(grouped_df_big_organs[model].reset_index()[value])})
            #df_data_small.append({'model': modelname+"_s", value: float(grouped_df_small_organs[model].reset_index()[value])})



    #df_bar_plot = pd.DataFrame(df_data)
    #df_bar_plot = df_bar_plot.set_index('model')
    #df_bar_plot.plot(kind='bar', ax=ax[0])

    #df_bar_plot = pd.DataFrame(df_data_big)
    #df_bar_plot = df_bar_plot.set_index('model')
    #df_bar_plot.plot(kind='bar', ax=ax[1])

    #df_bar_plot = pd.DataFrame(df_data_small)
    #df_bar_plot = df_bar_plot.set_index('model')
    #df_bar_plot.plot(kind='bar', ax=ax[2])

    #for axis in ax:
    #    for tick in axis.get_xticklabels():
    #        tick.set_rotation(0)

    fig.suptitle("sample and organ mean %s with average dice val over %1.2f. Separated all/big/small organs" % (value, threshold))
    ax[0].xaxis.label.set_visible(False)
    ax[1].xaxis.label.set_visible(False)
    filename = FILTERED_LINEPLOT_FILE % (stage, value+'_'+'-'.join(filters), threshold)
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


def create_boxplot(result_file, model, checkpoint, value='Dice', stage=''):
    values = set(['HausDist', 'Dice'])
    full_df = pd.read_csv(result_file)
    selected_df = full_df[(full_df['Model']==model) & (full_df['Checkpoint']==checkpoint)]
    list_to_drop = ['Model', 'Checkpoint', 'File']
    list_to_drop.extend(list(values-set([value])))
    selected_df.drop(list_to_drop, axis=1, inplace=True)
    selected_df.boxplot(by='Organ', figsize=(14, 4.8))
    #plt.savefig(BOXPLOT_CHECKPOINT_FILE % (stage, model, checkpoint, model))
    plt.show()
    plt.close()


def evaluate_resultfile(result_file):
    full_df = pd.read_csv(result_file)
    small_organs = full_df[full_df['Organ'].isin(defs.SMALL_ORGANS.keys())].copy()
    big_organs = full_df[full_df['Organ'].isin(defs.BIG_ORGANS.keys())].copy()

    full_df.drop(["File", "Organ"], axis=1, inplace=True)
    small_organs.drop(["File", "Organ"], axis=1, inplace=True)
    big_organs.drop(["File", "Organ"], axis=1, inplace=True)

    grouped_df = full_df.groupby(['Model', 'Checkpoint'])
    grouped_small_organs = small_organs.groupby(['Model', 'Checkpoint'])
    grouped_big_organs = big_organs.groupby(['Model', 'Checkpoint'])

    filename_split = os.path.split(result_file)

    evaluated_result = os.path.join(filename_split[0],os.path.splitext(filename_split[1])[0] + "_evaluated.csv")
    grouped_df.describe(percentiles=[]).round(2).to_csv(evaluated_result)

    evaluated_result = os.path.join(filename_split[0],os.path.splitext(filename_split[1])[0] + "_evaluated_small.csv")
    grouped_small_organs.describe(percentiles=[]).round(2).to_csv(evaluated_result)

    evaluated_result = os.path.join(filename_split[0],os.path.splitext(filename_split[1])[0] + "_evaluated_big.csv")
    grouped_big_organs.describe(percentiles=[]).round(2).to_csv(evaluated_result)


def evaluate_resultfile_organwise(result_file, modelfilter = []):
    df = pd.read_csv(result_file)

    df_filtered = df
    if modelfilter:
        df_filtered = df[df['Model'].isin(modelfilter)]
    gd = df_filtered.groupby(['Model', 'Checkpoint', 'Organ'])

    filename_split = os.path.split(result_file)

    evaluated_result = os.path.join(filename_split[0],os.path.splitext(filename_split[1])[0] + "_evaluated_organwise.csv")
    result_df = gd.describe(percentiles=[]).round(2)
    result_df.to_csv(evaluated_result)

    return result_df

def evauluate_evaluated_result(organwise_result_df, result_file):
    ID_COL = "id"

    df = organwise_result_df
    df.reset_index(inplace=True)

    all_organs = [name.rstrip() for name in LABELS.keys()]
    best_config = pd.DataFrame()

    for organ in all_organs:
        ofilt = df[(df["Organ"] == organ)]
        hd95max = ofilt[ofilt[("95haus_dist", "mean")] == ofilt[("95haus_dist", "mean")].max()].copy()
        hd95max[ID_COL] = "hd95max"
        best_config = best_config.append(hd95max, ignore_index=True)
        hd95min = ofilt[ofilt[("95haus_dist", "mean")] == ofilt[("95haus_dist", "mean")].min()].copy()
        hd95min[ID_COL] = "hd95min"
        best_config = best_config.append(hd95min, ignore_index=True)

        dicemax = ofilt[ofilt[("dice", "mean")] == ofilt[("dice", "mean")].max()].copy()
        dicemax[ID_COL] = "dicemax"
        best_config = best_config.append(dicemax, ignore_index=True)
        dicemin = ofilt[ofilt[("dice", "mean")] == ofilt[("dice", "mean")].min()].copy()
        dicemin[ID_COL] = "dicemin"
        best_config = best_config.append(dicemin, ignore_index=True)

    filename_split = os.path.split(result_file)
    evaluated_result = os.path.join(filename_split[0],
                                    os.path.splitext(filename_split[1])[0] + "_evaluated_organwise_best.csv")
    best_config.to_csv(evaluated_result)

        #collect in one df with id for min max hd or dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resultfile',
                        required=True,
                        help='path to the result file'
                        )

    args = parser.parse_args()

    evaluate_resultfile(args.resultfile)
    organwise_rsult_df = evaluate_resultfile_organwise(args.resultfile)
    evauluate_evaluated_result(organwise_rsult_df, args.resultfile)

    #create_boxplots_organ_avg(result, stage)
    #create_lineplots(result)
    #create_lineplot_organ_samp_avg(result, stage)
    #create_lineplot_organ_samp_avg_model_filtered(result, stage, [''], value='Dice', threshold=0.3)
    #create_boxplot(result, "half_e-3_48-8_dice_1024s", 100000, value='Dice', stage=stage)
import argparse
import pandas
import os
import matplotlib.pyplot as plt

from defs import LABELS

PLOT_FILE = "../tune_models/figures/imbal_rat_%s.png"

def plot_imbal_statistics(resfile, organ_filter=[], window_size_filter=[], dataset_size_filter=[], test_train_filter=[]):

    df = pandas.read_csv(resfile)

    df = df[df['Datasize'].isin(dataset_size_filter) & df['Windowsize'].isin(window_size_filter) & df['Dataset'].isin(test_train_filter)]

    all_organs = [name.rstrip() for name in LABELS.keys()]
    df['Total'] = df[all_organs].sum(axis=1)
    all_organs.append('Total')

    grouped_df_organs = df.groupby(['File'])[all_organs].mean()
    grouped_df_organs.boxplot(all_organs, figsize=(8, 4))

    plt.title("imbalence ratio, mean over sample windows")
    samples = df['Sample'].max() + 1
    filename = PLOT_FILE % ('-'.join(test_train_filter)+"_"+'-'.join(dataset_size_filter)+"_"+'-'.join(str(size) for size in window_size_filter)+"_"+str(samples))
    print filename
    plt.savefig(filename, dpi=100)
    plt.close()

def plot_all_means(resfile, test_train_filter=[]):

    df = pandas.read_csv(resfile)
    df = df[df['Dataset'].isin(test_train_filter)]

    all_organs = [name for name in LABELS.keys()]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 8))
    ax_cnt = 0

    samples = df['Sample'].max() + 1

    for dataset_size in ['full', 'half', 'quarter']:
        df_filtered_size = df[df['Datasize'].isin([dataset_size])]
        window_sizes = [96, 48]
        if dataset_size == 'full':
            window_sizes = [96]
        if dataset_size == 'quarter':
            window_sizes = [48]
        for window_size in window_sizes:
            df_filtered_window = df_filtered_size[df_filtered_size['Windowsize'].isin([window_size])]

            dataname = dataset_size + "_" + str(window_size)
            df_filtered_window[dataname] = df_filtered_window[all_organs].sum(axis=1)
            df_filtered_window.drop(all_organs, axis=1, inplace=True)

            ax[ax_cnt].set_ylim([0, 0.0175])

            grouped_df = df_filtered_window.groupby(['File'])[dataname].mean()
            grouped_df.reset_index().boxplot([dataname], ax=ax[ax_cnt])
            ax_cnt += 1

    sampling_method = getSamplingMethod(resfile)

    fig.suptitle("imbalence ratio, mean over %d sampled %s windows" % (samples, sampling_method))

    filename = PLOT_FILE % ('-'.join(test_train_filter)+"_"+str(samples)+"_"+sampling_method)
    print filename
    fig.savefig(filename, dpi=75)


def getSamplingMethod(resfile):
    sampling_method = ''
    resfilename = os.path.split(resfile)[1]
    if 'uniform' in resfilename:
        sampling_method = 'uniform'
    if 'weighted' in resfilename:
        sampling_method = 'weighted'
    return sampling_method


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resfile',
                        required=True
                        )

    args = parser.parse_args()
    plot_all_means(args.resfile, test_train_filter=["Train"])
    #plot_imbal_statistics(args.resfile, organ_filter=[], window_size_filter=[96], dataset_size_filter=["full"], test_train_filter=["Train"])


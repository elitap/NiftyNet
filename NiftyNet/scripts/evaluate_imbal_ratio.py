import argparse
import pandas
import os
import matplotlib.pyplot as plt

from defs import LABELS

PLOT_FILE = "../tune_models_2nd/figures/imbal_rat_%s.png"
#SAMPLING_METHODS = ['weighted', 'uniform']
SAMPLING_METHODS = ['weighted']

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

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
    all_organs = [name for name in LABELS.keys()]
    sampling_cnt = 0

    for sampling in SAMPLING_METHODS:
        loadable_resfile = resfile % (sampling)


        df = pandas.read_csv(loadable_resfile)
        df = df[df['Dataset'].isin(test_train_filter)]
        ax_cnt = 0

        samples = df['Sample'].max() + 1

        #for dataset_size in ['full', 'half', 'quarter']:
        for dataset_size in ['full']:
            df_filtered_size = df[df['Datasize'].isin([dataset_size])]
            window_sizes = [48, 24]
            #window_sizes = [96, 48]

            #if dataset_size == 'full':
            #    window_sizes = [96]
            #if dataset_size == 'quarter':
            #    window_sizes = [48]
            for window_size in window_sizes:
                df_filtered_window = df_filtered_size[df_filtered_size['Windowsize'].isin([window_size])]

                dataname = dataset_size + "_" + str(window_size) + "_" + sampling
                df_filtered_window[dataname] = df_filtered_window[all_organs].sum(axis=1)
                df_filtered_window.drop(all_organs, axis=1, inplace=True)

                ax[sampling_cnt, ax_cnt].set_ylim([0, 0.65])

                grouped_df = df_filtered_window.groupby(['File'])[dataname].mean()
                grouped_df.reset_index().boxplot([dataname], ax=ax[sampling_cnt, ax_cnt])
                ax_cnt += 1
        sampling_cnt += 1

    fig.suptitle("imbalence ratio, mean over %d samples" % (samples))

    filename = PLOT_FILE % ('-'.join(test_train_filter)+"_"+str(samples))
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


def evaluate_resultfile(result_file):
    full_df = pandas.read_csv(result_file)
    all_organs = [name.rstrip() for name in LABELS.keys()]

    grouped_df_organs = full_df.groupby(['Dataset','Datasize','Windowsize','File'])[all_organs].mean()
    grouped_df_organs["file_imbalence"] = grouped_df_organs[all_organs].sum(axis=1)

    grouped_df_organs.drop(all_organs, axis=1, inplace=True)

    group = grouped_df_organs.groupby(['Dataset', 'Datasize', 'Windowsize'])

    filename_split = os.path.split(result_file)
    evaluated_result = os.path.join(filename_split[0], os.path.splitext(filename_split[1])[0] + "_evaluated.csv")
    group.describe(percentiles=[]).round(3).to_csv(evaluated_result)



def print_histogram(resfile, window_size_filter=[], dataset_size_filter=[], test_train_filter=[]):

    df = pandas.read_csv(resfile)

    df = df[df['Datasize'].isin(dataset_size_filter) & df['Windowsize'].isin(window_size_filter) & df['Dataset'].isin(test_train_filter)]

    all_organs = [name.rstrip() for name in LABELS.keys()]

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))

    # for dataset_size in ['full', 'half', 'quarter']:
    for cnt, organ in enumerate(all_organs):
        df.hist(column=organ, ax=ax[cnt/4, cnt % 4], bins=50)
        ax[cnt / 4, cnt % 4].set_ylim([0, 250])
        #ax[cnt / 4, cnt % 4].set_xlim([0, 1])

    name = '-'.join(test_train_filter)+"_"+'-'.join(str(e) for e in window_size_filter)+"_"+'-'.join(dataset_size_filter)
    fig.suptitle(name)
    plt.show();
    filename = ("../results/img/l2voxel_no_bg_hist_%s.eps") % name
    print filename

    fig.savefig(filename, dpi=75)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resultfile',
                        required=True,
                        help='path to the result file'
                        )

    args = parser.parse_args()
    print_histogram(args.resultfile, window_size_filter=[48], dataset_size_filter=["quarter"], test_train_filter=["Train"])
    #evaluate_resultfile(args.resultfile)
    #plot_all_means(args.resfile, test_train_filter=["Train"])
    #plot_imbal_statistics(args.resfile, organ_filter=[], window_size_filter=[96], dataset_size_filter=["full"], test_train_filter=["Train"])


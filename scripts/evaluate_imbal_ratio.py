import argparse
import pandas
import matplotlib.pyplot as plt

from defs import LABELS

PLOT_FILE = "../tune_models/figures/imbal_rat_%s.png"

def plot_imbal_statistics(resfile, organ_filter=[], window_size_filter=[], dataset_size_filter=[], test_train_filter=[]):

    df = pandas.read_csv(resfile)

    df = df[df['Datasize'].isin(dataset_size_filter) & df['Windowsize'].isin(window_size_filter) & df['Dataset'].isin(test_train_filter)]

    all_organs = [name for name in LABELS.keys()]
    df['Total'] = df[all_organs].sum(axis=1)
    all_organs.append('Total')

    grouped_df_organs = df.groupby(['File'])[all_organs].mean()
    grouped_df_organs.boxplot(all_organs, figsize=(8, 4))

    plt.title("imbalence ratio, mean over sample windows")
    filename = PLOT_FILE % ('-'.join(test_train_filter)+"_"+'-'.join(dataset_size_filter)+"_"+'-'.join(str(size) for size in window_size_filter))
    print filename
    plt.savefig(filename, dpi=100)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resfile',
                        required=True
                        )

    args = parser.parse_args()
    plot_imbal_statistics(args.resfile, organ_filter=[], window_size_filter=[48], dataset_size_filter=["quarter"], test_train_filter=["Train"])


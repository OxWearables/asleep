import argparse
import os
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

"""
How to run the script:

```bash
python src/asleep/visu_sleep.py outputs/test

```
This script will visualize the sleep prediction results from the sleepnet model.
The input data will be the subject-level prediction folder that contains
the following files:
- predictions.csv: the predicted sleep labels
- raw.csv: the raw accelerometer data
- sleep_blocks.csv: the sleep blocks

"""


def main():
    parser = argparse.ArgumentParser(
        description="A tool to visualize the sleep prediction results",
        add_help=True
    )
    parser.add_argument("filepath", help="Enter path to the subject-level prediction folder")
    args = parser.parse_args()

    # check all the files are there
    prediction_path = os.path.join(args.filepath, 'predictions.csv')
    sleep_blocks_path = os.path.join(args.filepath, 'sleep_block.csv')
    raw_x_path = os.path.join('outputs/test', 'data2model.npy')
    plot_path = os.path.join(args.filepath, 'sleep.png')

    if not os.path.exists(prediction_path):
        raise OSError(f"File {prediction_path} does not exist")
    if not os.path.exists(sleep_blocks_path):
        raise OSError(f"File {sleep_blocks_path} does not exist")
    if not os.path.exists(raw_x_path):
        raise OSError(f"File {raw_x_path} does not exist")

    y_df = pd.read_csv(prediction_path)
    sleep_block_df = pd.read_csv(sleep_blocks_path)
    raw_x = np.load(raw_x_path)

    # compute enmo
    raw_x_sq = np.square(raw_x)
    raw_x_sq = raw_x_sq[:, 0, :] + raw_x_sq[:, 1, :] + raw_x_sq[:, 2, :]
    raw_x_sq[np.isnan(raw_x_sq)] = 0
    raw_enmo = np.sqrt(raw_x_sq)
    raw_enmo = np.mean(raw_enmo, axis=1)

    # get the exact date of the sleep blocks
    exact_date_df = sleep_block_df.loc[sleep_block_df['is_longest_block']].copy()
    exact_date_df['interval_start'] = pd.to_datetime(exact_date_df['interval_start'])
    exact_date_df['interval_end'] = pd.to_datetime(exact_date_df['interval_end'])

    y_df['enmo'] = raw_enmo
    y_df['time'] = pd.to_datetime(y_df['time'])

    # visulise the sleep prediction results
    plt.box(False)

    nrows = len(exact_date_df)
    fig = plt.figure(None, figsize=(10, nrows), dpi=300)
    MAXRANGE = 2  # 2g (above this is very rare)

    # TODO: add feature to plot the SSL label
    # TODO: handle non_wear label
    # Package this to a script
    idx = 0

    for _, row in exact_date_df.iterrows():
        day_start = row['interval_start']
        day_end = row['interval_end']
        day_df = y_df[(y_df['time'] >= day_start) & (y_df['time'] <= day_end)].copy()
        day_df['is_nonwear'] = day_df['enmo'] == 0
        day_df.loc[day_df['is_nonwear'], 'sleep_wake'] = 'non_wear'

        is_sleep = np.array(day_df['sleep_wake'] == 'sleep').astype('f4')
        is_wake = np.array(day_df['sleep_wake'] == 'wake').astype('f4')
        is_nonwear = np.array(day_df['sleep_wake'] == 'non_wear').astype('f4')

        ax = fig.add_subplot(nrows, 1, 1 + idx)

        ax.plot(day_df['time'], day_df['enmo'], color='k', linewidth=1.5)
        ax.set_ylim(0, 2)
        ax.set_xlim(day_start, day_end)

        a = np.array([is_sleep, is_wake, is_nonwear]) * MAXRANGE

        ax.stackplot(day_df['time'], a,
                     colors=['#785EF0', '#FE6100', 'gray'],
                     edgecolor="none")

        ax.set_ylabel(day_start.strftime("%A\n%d %B"),
                      weight='bold',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='horizontal',
                      fontsize='medium',
                      color='k',
                      labelpad=5)

        plt.yticks([])
        plt.xticks([])
        ax.set_frame_on(False)  # remove frame

        if idx == 0:
            hrLabels = ['12:00', '16:00', '20:00', '00:00', '04:00', '08:00', '12:00']
            plt.xticks(pd.date_range(start=datetime.combine(day_start, time(12, 0, 0, 0)),
                                     end=datetime.combine(day_start + timedelta(days=1),
                                                          time(12, 0, 0, 0)),
                                     freq='4H'))
            ax.xaxis.tick_top()
            ax.set_xticklabels(hrLabels)
            ax.tick_params(labelbottom=False, labeltop=True, labelleft=False)

        idx += 1

    ax = fig.add_subplot(nrows, 1, idx)
    ax.axis('off')
    legend_patches = [mlines.Line2D([], [], color='k', label='acceleration')]

    labels = ['sleep', 'wake', 'non_wear']
    colors = ['#785EF0', '#FE6100', 'gray']

    for label, color in zip(labels, colors):
        legend_patches.append(mpatches.Patch(color=color, label=label))
    # create overall legend
    plt.legend(handles=legend_patches, bbox_to_anchor=(0., -1., 1., 1.),
               loc='center', ncol=4, mode="best",
               borderaxespad=0, framealpha=0.6, frameon=False, fancybox=True)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("File written to", plot_path)


if __name__ == '__main__':
    main()

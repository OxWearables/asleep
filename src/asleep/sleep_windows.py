import pandas as pd
from datetime import timedelta
import numpy as np
import itertools


IS_SLEEP_FLAG = 1
CLASS_POS = 0
LEN_POS = 1
IDX_POS = 2


def find_sleep_blocks(interval_counter):
    sleep_block_idxes = []
    i = 0

    while i < len(interval_counter):
        current_block = interval_counter[i]
        current_class = current_block[CLASS_POS]

        if current_class == IS_SLEEP_FLAG:
            sleep_block_idxes.append(i)
        i += 1

    return sleep_block_idxes


def find_block_duration(sleep_df):
    # Find sleep duration for each block
    # Return: Counter: n x 3: each row has block class, block length and the
    # block starting idx
    block_lengths = [
        (x[0], len(list(x[1]))) for x in itertools.groupby(sleep_df["label"])
    ]   # contains (class, block_len)
    block_lengths = np.array(block_lengths)

    counter = []  # [label, block_len, start_idx]
    i = 0
    freq_idx = 1
    for my_ele in block_lengths:
        my_ele = my_ele.tolist()
        my_ele.append(i)
        i += my_ele[freq_idx]
        counter.append(my_ele)
    counter = np.array(counter)
    return counter


def fill_sleep_block_gaps(sleep_block_idxes, counter):
    # fill all the eligible sleep blocks gaps with sleep class
    gap2fill = count_sleep_block_gap(sleep_block_idxes, counter)

    for gap in gap2fill:
        current_sleep_block_counter_id = sleep_block_idxes[gap]
        next_sleep_block_counter_id = sleep_block_idxes[gap + 1]

        k = current_sleep_block_counter_id + 1
        while k <= next_sleep_block_counter_id:
            counter[k][CLASS_POS] = IS_SLEEP_FLAG
            k += 1

    return counter


def count_sleep_block_gap(sleep_block_idxes, counter, epoch_length=30):
    # epoch_length sec
    gap2fill = []
    if len(sleep_block_idxes) > 1:
        first_block = counter[sleep_block_idxes[0]]
        end_of_block_idx = first_block[IDX_POS] + first_block[LEN_POS]
        min_one_hour_block_diff = 60 * 60 / epoch_length
        i = 1
        while i < len(sleep_block_idxes):
            current_block = counter[sleep_block_idxes[i]]
            dist2pre_block = current_block[IDX_POS] - end_of_block_idx
            if dist2pre_block <= min_one_hour_block_diff:
                gap2fill.append(i - 1)
            end_of_block_idx = current_block[IDX_POS] + current_block[LEN_POS]
            i += 1
    return gap2fill


def find_sleep_windows(sleep_blocks):
    current_start_master_idx = -1
    current_end_master_idx = -1
    pre_class = -1
    res_start_master_idx = -1
    res_end_master_idx = -1
    all_sleep_blocks = []

    for win in sleep_blocks:
        if win[CLASS_POS] == IS_SLEEP_FLAG:
            if pre_class != IS_SLEEP_FLAG:
                current_start_master_idx = win[IDX_POS]
            current_end_master_idx = win[IDX_POS] + win[LEN_POS]
        else:
            pre_class = win[CLASS_POS]
            if current_start_master_idx != -1:
                all_sleep_blocks.append(
                    [current_start_master_idx, current_end_master_idx])
                current_start_master_idx = -1
            continue
        pre_class = win[CLASS_POS]

        # compare current block len to ans
        if (current_end_master_idx - current_start_master_idx) > (
            res_end_master_idx - res_start_master_idx
        ):
            res_end_master_idx = current_end_master_idx
            res_start_master_idx = current_start_master_idx

    if pre_class == IS_SLEEP_FLAG and current_start_master_idx != -1:
        all_sleep_blocks.append(
            [current_start_master_idx, current_end_master_idx])

    return all_sleep_blocks, res_start_master_idx, res_end_master_idx


def get_sleep_blocks(interval_df):
    # merge sleep blocks that are less than 1 hour apart
    counter = find_block_duration(interval_df)  # [label, block_len, start_idx]
    sleep_block_idxes = find_sleep_blocks(counter)
    sleep_win_counter = fill_sleep_block_gaps(sleep_block_idxes, counter)

    # return the start_idx and end_idx for each sleep window
    all_sleep_idxes, long_start_idx, long_end_idx = find_sleep_windows(
        sleep_win_counter)

    sleep_wins = []
    for idx_pair in all_sleep_idxes:
        start_idx = idx_pair[0]
        end_idx = idx_pair[1]
        sleep_wins.append([interval_df.iloc[[start_idx]]["time"].item(),
                           interval_df.iloc[[end_idx - 1]]["time"].item()])

    start_sleep = interval_df.iloc[[long_start_idx]]["time"].item()
    end_sleep = interval_df.iloc[[long_end_idx - 1]]["time"].item()
    long_sleep_win = [start_sleep, end_sleep]
    return sleep_wins, long_sleep_win


def find_valid_sleep_blocks(counter, epoch_length, min_duration_min=30):
    SLEEP_LABEL = 1
    sleep_block_min_len = min_duration_min * 60  # .5 hour
    epoch_min_len = sleep_block_min_len / epoch_length

    valid_sleep_block_idxes = []
    for i in range(len(counter)):
        e = counter[i]
        len_idx = 1
        label_idx = 0
        if e[label_idx] == SLEEP_LABEL and e[len_idx] >= epoch_min_len:
            valid_sleep_block_idxes.append(i)
    return valid_sleep_block_idxes


def find_sleep_block_duration(sleep_df):
    # Find sleep duration for each block
    # Return: Counter: n x 3: each row has block class, block length and the
    # block starting idx
    block_lengths = [
        (x[0], len(list(x[1]))) for x in itertools.groupby(sleep_df["label"])
    ]
    block_lengths = np.array(block_lengths)

    counter = []  # [label, block_len, start_idx]
    i = 0
    freq_idx = 1
    for my_ele in block_lengths:
        my_ele = my_ele.tolist()
        my_ele.append(i)
        i += my_ele[freq_idx]
        counter.append(my_ele)
    counter = np.array(counter)
    return counter


def find_gaps2fill(valid_sleep_block_idxes, epoch_length, counter):
    """
    Identify gap idx that need filling.
    Take the eligible sleep block idx, output the idx for the gap that could be filled.
    The idx will be a pair of starting sleep block and ending sleep block.
    """
    max_non_wear_len = 60 * 60 / epoch_length  # one hour
    gap2fill = []
    for i in range(len(valid_sleep_block_idxes) - 1):

        current_block_idx = valid_sleep_block_idxes[i]
        next_block_idx = valid_sleep_block_idxes[i + 1]

        # compute the gap between these two blocks
        original_idx_pos = 2
        sleep_block_gap = (
            counter[next_block_idx][original_idx_pos] -
            counter[current_block_idx][original_idx_pos]
        )
        if sleep_block_gap <= max_non_wear_len:
            gap2fill.append([current_block_idx, next_block_idx])
    return gap2fill


def fill_gaps(my_df, counter, gap2fill):
    class_label = 1
    for gap in gap2fill:
        start_block_idx = gap[0]
        end_block_idx = gap[1]

        idx_pos = 2
        date_df_idx_start = counter[start_block_idx][idx_pos]
        date_df_idx_end = counter[end_block_idx][idx_pos]

        my_df.loc[date_df_idx_start:date_df_idx_end, "label"] = class_label
    return my_df


def get_sleep_blocks_per_day(my_df, my_intervals):
    """
    my_df: time-series df produced by bbaa
    epoch_len: sec
    date_format = '%Y-%m-%d %H:%M:%S.%f%z [Europe/London]'
    my_df['time'] = pd.to_datetime(my_df['time'], format=date_format)
    start_date = my_df['time'][0]
    end_date = my_df['time'][len(my_df)-1]

    """
    all_sleep_wins = []
    sleep_wins_long = []
    interval_starts = []
    interval_ends = []
    wear_times = []

    for interval in my_intervals:
        my_start_time = interval[0]
        my_end_time = interval[1]
        interval_df = my_df[
            (my_df["time"] >= my_start_time) & (my_df["time"] < my_end_time)
        ]
        wear_time = np.sum(interval_df["is_wear"]) / (2 * 60)

        sleep_wins, long_sleep_win = get_sleep_blocks(interval_df)
        sleep_wins_long.append(long_sleep_win)
        all_sleep_wins.extend(sleep_wins)

        interval_starts.extend([my_start_time] * len(sleep_wins))
        interval_ends.extend([my_end_time] * len(sleep_wins))
        wear_times.extend([wear_time] * len(sleep_wins))

    return all_sleep_wins, sleep_wins_long, interval_starts, interval_ends, wear_times


def time_series2sleep_blocks(
    my_df, date_format="%Y-%m-%d %H:%M:%S.%f"
):
    start_date = my_df["time"][0]
    end_date = my_df["time"][len(my_df) - 1]

    my_intervals = get_day_intervals(start_date, end_date, date_format)
    all_sleep_wins, \
        sleep_wins_long, \
        interval_start, \
        interval_end, \
        wear_time = get_sleep_blocks_per_day(my_df, my_intervals)
    return all_sleep_wins, sleep_wins_long, interval_start, interval_end, wear_time


def get_day_intervals(start_date, end_date, date_format):
    # 1. Get day intervals

    day_intervals = []
    day_end_str = "1990-01-01 09:47:50.439000"
    my_day_end = pd.to_datetime(day_end_str, format=date_format)
    my_day_start = start_date.replace(
        hour=12, minute=0, second=0, microsecond=0)
    my_day_start = my_day_start - timedelta(hours=24)

    while my_day_end < end_date:
        my_day_end = my_day_start + timedelta(hours=23, minutes=59, seconds=59)
        day_intervals.append([my_day_start, my_day_end])
        my_day_start = my_day_start + timedelta(hours=24)
    return day_intervals

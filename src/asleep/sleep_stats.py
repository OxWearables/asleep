# This file will compute all the necessary sleep parameters
import numpy as np
from datetime import timedelta
import pandas as pd
from collections import Counter
import re

WAKE_LABEL = 0


# Design a day class that stores all the information for a day
# This is to make it easier to compute the summary statistics
# The day class hold all the data for a day
# It also has a function to compute the summary statistics, get non-wear time, etc.
# The day class should be able to be serialized to json

def date_parser(t):
    """
    Parse date a date string of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    """
    tz = re.search(r"(?<=\[).+?(?=\])", t)
    if tz is not None:
        tz = tz.group()
    t = re.sub(r"\[(.*?)\]", "", t)
    return pd.to_datetime(t, utc=True).tz_convert(tz)


def psg_five2three(y, y_pred):
    my_y = np.copy(y)
    my_y_pred = np.copy(y_pred)
    my_y[my_y == 2] = 1
    my_y[my_y == 3] = 1
    my_y[my_y == 4] = 2

    my_y_pred[my_y_pred == 2] = 1
    my_y_pred[my_y_pred == 3] = 1
    my_y_pred[my_y_pred == 4] = 2
    return my_y, my_y_pred


# **** individual metric **** #
def get_sleep_onset_index(sleep_stages):
    i = 0
    k = 0
    threshold = 3  # epochs
    for stage in sleep_stages:
        if stage != WAKE_LABEL:
            k += 1
            if k >= threshold:
                break
        else:
            k = 0
        i += 1
    return i - (threshold - 1)


def get_sol(sleep_stages):
    # sleep onset latency
    # time in bed/light off till the first episode of sleep
    return get_sleep_onset_index(sleep_stages) * 0.5


def get_waso(sleep_stages):
    # wake after sleep onset, periods of wakefulness occurring after defined sleep onset
    # Sleep onset occurs after 1.5 mins of non-wake stages
    sleep_onset_index = get_sleep_onset_index(sleep_stages)
    stages_after_sleep_onset = sleep_stages[sleep_onset_index:]
    total_wake = np.sum(stages_after_sleep_onset == WAKE_LABEL) * 0.5
    return total_wake


def get_tst(sleep_stages):
    # Total sleep time
    # input numpy array
    # return in mins
    return np.sum(sleep_stages != 0) * 0.5


def get_se(sleep_stages):
    # Sleep efficiency: TST/(time in bed)
    return get_tst(sleep_stages) / (len(sleep_stages) * 0.5)


def get_reml(sleep_stages, rem_label=4):
    # Rapid eye movement latency is the time from the sleep onset to the first
    # epoch of REM sleep
    i = 0
    for stage in sleep_stages:
        if stage == rem_label:
            break
        i += 1

    return i * 0.5 - get_sol(sleep_stages)


def get_stage_lens(sleep_stages, rem_label=4):
    # Sleep efficiency: TST/(time in bed)
    wake = np.sum(sleep_stages == 0) * 0.5
    n1 = np.sum(sleep_stages == 1) * 0.5
    n2 = np.sum(sleep_stages == 2) * 0.5
    n3 = np.sum(sleep_stages == 3) * 0.5
    nrem = n1 + n2 + n3
    rem = np.sum(sleep_stages == rem_label) * 0.5
    return wake, n1, n2, n3, nrem, rem


def get_stage_portions(sleep_stages, rem_label=4):
    # Sleep efficiency: TST/(time in bed)
    tst = get_tst(sleep_stages)
    wake = np.sum(sleep_stages == 0) * 0.5 / tst
    n1 = np.sum(sleep_stages == 1) * 0.5 / tst
    n2 = np.sum(sleep_stages == 2) * 0.5 / tst
    n3 = np.sum(sleep_stages == 3) * 0.5 / tst
    nrem = n1 + n2 + n3
    rem = np.sum(sleep_stages == rem_label) * 0.5 / tst
    return wake, n1, n2, n3, nrem, rem


def get_all_sleep_paras(sleep_stages, rem_label=4):
    sol = get_sol(sleep_stages)
    tst = get_tst(sleep_stages)
    waso = get_waso(sleep_stages)
    reml = get_reml(sleep_stages, rem_label)
    se = get_se(sleep_stages)
    wake, n1, n2, n3, nrem, rem = get_stage_lens(sleep_stages, rem_label)
    num_transitions = get_num_stage_transition(sleep_stages)
    return sol, tst, waso, reml, se, wake, n1, n2, n3, nrem, rem, num_transitions


def get_AgeAndAHI(meta_df, subject_id):
    age = meta_df[meta_df["Study ID"] == subject_id]["Age at Study"].item()
    AHI = meta_df[meta_df["Study ID"] ==
                  subject_id]["#resp events/TST (hr)=AHI"].item()
    return age, AHI


# get sleep bounts
def get_sleep_bouts(sleep_stages):
    # if the next stage is back to the original stage, then keep the current
    # one
    new_stages = []
    i = 0
    while i < len(sleep_stages) - 2:
        if sleep_stages[i] == sleep_stages[i + 1]:
            new_stages.append(sleep_stages[i])
            i += 1
        elif sleep_stages[i] == sleep_stages[i + 2]:
            new_stages.append(sleep_stages[i])
            new_stages.append(sleep_stages[i])
            i += 2
        else:
            new_stages.append(sleep_stages[i])
            i += 1
    new_stages.append(sleep_stages[-2])
    new_stages.append(sleep_stages[-1])
    return new_stages


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


# get sleep bouts
def get_five_min_sleep_bouts(sleep_stages):
    # if the next stage is back to the original stage, then keep the current
    # one
    start_idx = 0  # sleep bout for every 5 minutes
    sleep_bouts = []
    bout_length = 10  # epochs
    while start_idx < len(sleep_stages) - bout_length:
        end_idx = start_idx + bout_length + 1
        current_stages = sleep_stages[start_idx:end_idx]
        sleep_bouts.append(most_common(current_stages))
        start_idx += bout_length

    return sleep_bouts


def get_num_stage_transition(sleep_stages):
    """
    Taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2982738/
    Utility of Sleep Stage Transitions in Assessing Sleep Continuity
    6 transitions will be counted: wake2nrem, nrem2wake, nrem2rem,
    rem2nrem, rem2wake, wake2rem

    Return #transition per hour
    """
    pre = -1
    num_transitions = 0

    # convert from five stage to three stage
    three_stages, _ = psg_five2three(sleep_stages, sleep_stages)
    # three_stages_bouts = get_five_min_sleep_bouts(sleep_stages)

    for my_stage in three_stages:
        if pre != -1:
            if pre != my_stage:
                num_transitions += 1
        pre = my_stage
    num_hour = len(sleep_stages) / (2 * 60)
    return num_transitions / num_hour


# **** Overall metric estimates **** #
def get_sleep_paras(subject_id, y_df, y_pred, pid):
    # This extracts both ground truth and prediction (mostly for model evaluation)
    # INPUT df are actually NP arrays
    subject_filter = pid == subject_id
    y_df = np.array(y_df)

    y_df = y_df[subject_filter]
    y_pred_df = y_pred[subject_filter]

    paras = []
    (
        sol,
        tst,
        waso,
        reml,
        se,
        wake,
        n1,
        n2,
        n3,
        nrem,
        rem,
        transitions,
    ) = get_all_sleep_paras(y_df)
    paras.extend([sol, tst, waso, reml, se, wake,
                  n1, n2, n3, nrem, rem, transitions])
    (
        sol,
        tst,
        waso,
        reml,
        se,
        wake,
        n1,
        n2,
        n3,
        nrem,
        rem,
        transitions,
    ) = get_all_sleep_paras(y_pred_df)
    paras.extend([sol, tst, waso, reml, se, wake,
                  n1, n2, n3, nrem, rem, transitions])
    return paras


def get_sleep_block_day(first_time):
    # get hour of a timestamp first_time
    start_hour = 12
    if first_time.hour >= 12:
        block_start = first_time.replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )  # after noon current day
    else:
        # before noon the day before
        block_start = first_time - timedelta(days=1)
        block_start = block_start.replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )
    return block_start.date()


def add_mean_and_median(
        data_df,
        data_dict,
        metric_list,
        name_prefix="overall",
        get_mean=True,
        get_median=True,
):
    var_means = data_df.mean(axis=0, numeric_only=True)
    var_median = data_df.median(axis=0, numeric_only=True)

    for my_metric in metric_list:
        mean_prefix = name_prefix + "_" + "mean"
        median_prefix = name_prefix + "_" + "median"

        mean_name = mean_prefix + "_" + my_metric
        median_name = median_prefix + "_" + my_metric

        if get_mean:
            data_dict[mean_name] = var_means[my_metric]
        if get_median:
            data_dict[median_name] = var_median[my_metric]

    return data_dict


def compute_stats(
        summary_df,
        y_pred,
        times,
        weekday_y_pred,
        weekday_times,
        weekend_y_pred,
        weekend_times,
        metric_list,
):
    summary_dict = {}
    summary_dict = add_mean_and_median(summary_df, summary_dict, metric_list)

    # 0. Mon-Sun
    week_days = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    num_days_in_week = 7
    for i in range(num_days_in_week):
        ith_day_df = summary_df[summary_df["day"] == i]
        add_mean_and_median(
            ith_day_df,
            summary_dict,
            metric_list,
            name_prefix=week_days[i],
            get_median=False,
        )

    # 1. weekday and weekend
    weekday_df = summary_df[(
        (summary_df["day"] == 1) |
        (summary_df["day"] == 2) |
        (summary_df["day"] == 3) |
        (summary_df["day"] == 0) |
        (summary_df["day"] == 6)
    )]
    weekend_df = summary_df[(summary_df["day"] == 4) |
                            (summary_df["day"] == 5)]
    add_mean_and_median(
        weekday_df,
        summary_dict,
        metric_list,
        name_prefix="weekday")
    add_mean_and_median(
        weekend_df,
        summary_dict,
        metric_list,
        name_prefix="weekend")

    # 2. Include summary statistics
    # 2.1 day count
    weekday_count = 0
    weekend_count = 0
    for i in range(num_days_in_week):
        day_name = week_days[i] + "_num"
        day_count = len(summary_df[summary_df["day"] == i])
        summary_dict[day_name] = day_count
        if i == 4 or i == 5:
            weekend_count += day_count
        else:
            weekday_count += day_count
    summary_dict["weekend_day_num"] = weekend_count
    summary_dict["weekday_day_num"] = weekday_count

    # 2.2 hourly stage duration
    # TODO: add the divison of number of days
    times_min = np.array([x.hour * 60 + x.minute for x in times])
    weekday_times_min = np.array(
        [x.hour * 60 + x.minute for x in weekday_times])
    weekend_times_min = np.array(
        [x.hour * 60 + x.minute for x in weekend_times])
    # convert all times to hourly
    num_hour_in_day = 24
    for i in range(num_hour_in_day):
        lb = i * 60
        ub = (i + 1) * 60

        hour_filter = (times_min >= lb) & (times_min < ub)
        hourly_y = y_pred[hour_filter]
        get_stage_durations(summary_dict, hourly_y, stats_name=str(i) + "hour")

        weekday_hour_filter = (
            weekday_times_min >= lb) & (
            weekday_times_min < ub)
        weekday_hourly_y = weekday_y_pred[weekday_hour_filter]
        get_stage_durations(
            summary_dict, weekday_hourly_y, stats_name=str(i) + "-weekday-hour"
        )

        weekend_hour_filter = (
            weekend_times_min >= lb) & (
            weekend_times_min < ub)
        weekend_hourly_y = weekend_y_pred[weekend_hour_filter]
        get_stage_durations(
            summary_dict, weekend_hourly_y, stats_name=str(i) + "-weekend-hour"
        )

    return summary_dict


def get_stage_durations(summary_dict, hourly_y, stats_name):
    wake, n1, n2, n3, nrem, rem = get_stage_lens(hourly_y)
    tst = get_tst(hourly_y)
    summary_dict[stats_name + "_" + "wake_min"] = wake
    summary_dict[stats_name + "_" + "n1_min"] = n1
    summary_dict[stats_name + "_" + "n2_min"] = n2
    summary_dict[stats_name + "_" + "n3_min"] = n3
    summary_dict[stats_name + "_" + "nrem_min"] = nrem
    summary_dict[stats_name + "_" + "rem_min"] = rem
    summary_dict[stats_name + "_" + "tst_min"] = tst


def obtain_non_wear_df(sleep_block_path):
    all_time_df = pd.read_csv(
        sleep_block_path, parse_dates=["start", "end"], date_parser=date_parser
    )
    time_df = all_time_df[all_time_df["is_longest_block"]]
    # TODO: include nonwear processing placeholder

    time_df['imputed_duration-H'] = 0

    # Extract dates of the first days
    start_dates = []
    non_wear_threshold_hrs = 2
    ok_wear_time = []
    for _, row in time_df.iterrows():
        first_date = get_sleep_block_day(row["start"])
        start_dates.append(first_date)
        if row["imputed_duration-H"] > non_wear_threshold_hrs:
            ok_wear_time.append(False)
        else:
            ok_wear_time.append(True)

    time_df["1st_date"] = start_dates
    time_df["ok_wear_time"] = ok_wear_time
    return time_df

# This file will compute all the necessary sleep parameters
import json
import numpy as np
from datetime import timedelta
import pandas as pd
from collections import Counter
import re

WAKE_LABEL = 0


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
    # Rapid eye movement latency is the time from the sleep onset to the first epoch of REM sleep
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
    AHI = meta_df[meta_df["Study ID"] == subject_id]["#resp events/TST (hr)=AHI"].item()
    return age, AHI


# get sleep bounts
def get_sleep_bouts(sleep_stages):
    # if the next stage is back to the original stage, then keep the current one
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
    # if the next stage is back to the original stage, then keep the current one
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
    paras.extend([sol, tst, waso, reml, se, wake, n1, n2, n3, nrem, rem, transitions])
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
    paras.extend([sol, tst, waso, reml, se, wake, n1, n2, n3, nrem, rem, transitions])
    return paras


def get_sleep_block_day(first_time):
    # get hour of a timestamp first_time
    start_hour = 12
    if first_time.hour >= 12:
        block_start = first_time.replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )  # after noon current day
    else:
        block_start = first_time - timedelta(days=1)  # before noon the day before
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
    weekday_df = summary_df[
        (summary_df["day"] == 1)
        | (summary_df["day"] == 2)
        | (summary_df["day"] == 3)
        | (summary_df["day"] == 0)
        | (summary_df["day"] == 6)
    ]
    weekend_df = summary_df[(summary_df["day"] == 4) | (summary_df["day"] == 5)]
    add_mean_and_median(weekday_df, summary_dict, metric_list, name_prefix="weekday")
    add_mean_and_median(weekend_df, summary_dict, metric_list, name_prefix="weekend")

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
    weekday_times_min = np.array([x.hour * 60 + x.minute for x in weekday_times])
    weekend_times_min = np.array([x.hour * 60 + x.minute for x in weekend_times])
    # convert all times to hourly
    num_hour_in_day = 24
    for i in range(num_hour_in_day):
        lower_bound = i * 60
        upper_bound = (i + 1) * 60

        hour_filter = (times_min >= lower_bound) & (times_min < upper_bound)
        hourly_y = y_pred[hour_filter]
        get_stage_durations(summary_dict, hourly_y, stats_name=str(i) + "hour")

        weekday_hour_filter = (weekday_times_min >= lower_bound) & (
            weekday_times_min < upper_bound
        )
        weekday_hourly_y = weekday_y_pred[weekday_hour_filter]
        get_stage_durations(
            summary_dict, weekday_hourly_y, stats_name=str(i) + "-weekday-hour"
        )

        weekend_hour_filter = (weekend_times_min >= lower_bound) & (
            weekend_times_min < upper_bound
        )
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


def check_inclusion(time_df, current_date):
    selected_day = time_df[time_df["1st_date"] == current_date]
    assert len(selected_day) >= 1

    if selected_day["ok_wear_time"].item() is False:
        return False
    else:
        return True


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


def summarize_sleep_stages(y_pred, times, out_file, sleep_block_path=""):
    """
    It first computes the daily avg metrics for each night. Then compute the summary statistics
    across days adn weeks. The output is stored in a json format that can used be further analysis.

    y_pred: sleepnet output
    times: timestamp array
    sleep_block_file: longest sleep_block generated by get_sleep_windows.py that contains non-wear information
    out_file: json file name to store
    """
    metric_list = [
        "sol_min",
        "tst_min",
        "waso_min",
        "reml_min",
        "se_perc",
        "wake_min",
        "n1_min",
        "n2_min",
        "n3_min",
        "nrem_min",
        "rem_min",
        "transitions",
    ]
    summary_df = pd.DataFrame(columns=["day"] + metric_list)

    # 0. filter out the sleep blocks based on non-wear time for that block
    if len(sleep_block_path) != 0:
        time_df = obtain_non_wear_df(sleep_block_path)

        # nid2include = []
        # for my_pid in np.unique(npid):
        #     pid_filter = npid == my_pid
        #     current_times = times[pid_filter]
        #     start_time = current_times[0]
        #     current_first_date = get_sleep_block_day(start_time)
        #
        #     if check_inclusion(time_df, current_first_date):
        #         nid2include.append(my_pid)
        #
        # inclusion_filter = [k in nid2include for k in npid]
        #
        # # when no days are eligible, we don't filter anymore
        # if np.sum(inclusion_filter) != 0:
        #     y_pred = y_pred[inclusion_filter]
        #     npid = npid[inclusion_filter]
        #     times = times[inclusion_filter]

    # 1. Compute daily average. We will use the first day to annotate
    i = 0
    weekday_y_pred = []
    weekday_times = []
    weekend_y_pred = []
    weekend_times = []
    times = pd.to_datetime(times)
    for index, row in time_df.iterrows():
        start_time = row['start']
        end_time = row['end']
        print(start_time, end_time)
        print(times)
        # set trace here
        #import pdb; pdb.set_trace()

        time_filter = (times >= start_time) & (times < end_time)

        current_y = y_pred[time_filter]
        current_times = times[time_filter]
        block_start_time = current_times[0]
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
        ) = get_all_sleep_paras(current_y)

        current_first_date = get_sleep_block_day(block_start_time)
        summary_df.loc[i] = [
            current_first_date.weekday(),
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
        ]
        i += 1
        # store y_pred for weekend and weekday separately
        if current_first_date.weekday() == 4 or current_first_date.weekday() == 5:
            weekend_y_pred.extend(current_y)
            weekend_times.extend(current_times)
        else:
            weekday_y_pred.extend(current_y)
            weekday_times.extend(current_times)

    weekday_y_pred = np.array(weekday_y_pred)
    weekday_times = np.array(weekday_times)
    weekend_y_pred = np.array(weekend_y_pred)
    weekend_times = np.array(weekend_times)

    # 2. Calculate day-level and summary-level stats
    summary_dict = compute_stats(
        summary_df,
        y_pred,
        times,
        weekday_y_pred,
        weekday_times,
        weekend_y_pred,
        weekend_times,
        metric_list,
    )
    summary_dict["num_valid_days"] = len(time_df)

    # 3. Saving results
    with open(out_file, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print("Summary file written to %s!" % out_file)

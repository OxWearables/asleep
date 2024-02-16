from asleep.sleep_stats import get_all_sleep_paras, get_stage_durations
import numpy as np
import pandas as pd
import json


def summarize_df(my_df, prefix, compute_median=False):
    """
        Convert a dataframe to a json object with a pre-defined prefix
    """

    if compute_median:
        my_df = my_df.median(axis=0, numeric_only=True)
        computed_stats = my_df.add_prefix(prefix + "_median_")

    else:
        my_df = my_df.mean(axis=0, numeric_only=True)
        computed_stats = my_df.add_prefix(prefix + "_mean_")
    summary_dict = computed_stats.to_dict()
    return summary_dict


def summarize_daily_sleep(day_summary_df, output_json_path, min_wear_time_h):
    # 1. overall
    day_summary_df = day_summary_df[day_summary_df["wear_duration_H"] >=
                                    min_wear_time_h]
    overall_means_dict = summarize_df(day_summary_df, "overall")
    overall_median_dict = summarize_df(
        day_summary_df, "overall", compute_median=True)
    summary_dict = {**overall_means_dict, **overall_median_dict}

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
    # 2. weekday-specific
    for i in range(num_days_in_week):
        ith_day_df = day_summary_df[day_summary_df["day_of_week"] == i]
        day_dict = summarize_df(ith_day_df, week_days[i])
        summary_dict = {**summary_dict, **day_dict}

    # 3. weekday-weekend
    weekday_df = day_summary_df[
        (day_summary_df["day_of_week"] == 1) |
        (day_summary_df["day_of_week"] == 2) |
        (day_summary_df["day_of_week"] == 3) |
        (day_summary_df["day_of_week"] == 0) |
        (day_summary_df["day_of_week"] == 6)
    ]
    weekend_df = day_summary_df[(day_summary_df["day_of_week"] == 4) | (
        day_summary_df["day_of_week"] == 5)]

    # merge dictionaries into one dictionary
    weekday_dict = summarize_df(weekday_df, "weekday")
    weekend_dict = summarize_df(weekend_df, "weekend")
    summary_dict = {**summary_dict, **weekday_dict}
    summary_dict = {
        **summary_dict,
        **weekend_dict,
        'num_valid_weekday': sum(
            weekday_df['wear_duration_H'] >= min_wear_time_h),
        'num_valid_weekend': sum(
            weekend_df['wear_duration_H'] >= min_wear_time_h),
        'num_valid_days': sum(
            day_summary_df['wear_duration_H'] >= min_wear_time_h)}

    # save dictionary to json
    with open(output_json_path, 'w') as fp:
        json.dump(summary_dict, fp)
    print("Summary saved to: {}".format(output_json_path))


def generate_sleep_parameters(
        all_sleep_wins_df,
        times,
        predictions_df,
        day_summary_path):
    my_days = []
    longest_sleep_block = all_sleep_wins_df[all_sleep_wins_df['is_longest_block']]

    for index, row in longest_sleep_block.iterrows():
        start_t = row['start']
        end_t = row['end']

        time_filter = (times >= start_t) & (times <= end_t)
        current_day_y_pred = predictions_df['raw_label'][time_filter].values
        current_day_times = times[time_filter]

        current_day_obj = Day(
            current_day_y_pred,
            current_day_times,
            row['interval_start'],
            wear_duration=row['wear_duration_H'])
        my_days.append(current_day_obj)

    # Convert all the day summary jsons to dataframe
    day_summary_df = pd.DataFrame([day.summary for day in my_days])

    # save day level df to csv
    day_summary_df.to_csv(day_summary_path, index=False)
    return day_summary_df


def is_weekend(current_time):
    """
    Return True if the times are from a weekend, False otherwise.
    """
    if current_time.weekday() == 4 or current_time.weekday() == 5:
        return True
    else:
        return False


def get_day_stats(y_pred, wear_duration, is_my_weekend, interval_start):
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
        _,
    ) = get_all_sleep_paras(y_pred)

    # convert interval_start to string
    return dict({
        "start_day": interval_start.strftime("%Y-%m-%d"),
        "day_of_week": interval_start.weekday(),
        "wear_duration_H": wear_duration,
        "is_weekend": is_my_weekend,
        "sol_min": sol,
        "tst_min": tst,
        "waso_min": waso,
        "reml_min": reml,
        "se_perc": se,
        "wake_min": wake,
        "n1_min": n1,
        "n2_min": n2,
        "n3_min": n3,
        "nrem_min": nrem,
        "rem_min": rem,
    })


def get_day_hourly_stats(times, y_pred, summary):
    times_min = np.array([x.hour * 60 + x.minute for x in times])
    # convert all times to hourly
    num_hour_in_day = 24

    for i in range(num_hour_in_day):
        lower_bound = i * 60
        upper_bound = (i + 1) * 60

        hour_filter = (times_min >= lower_bound) & (times_min < upper_bound)
        hourly_y = y_pred[hour_filter]
        get_stage_durations(summary, hourly_y, stats_name=str(i) + "_hour")

    return summary


class Day:
    def __init__(self, y_pred, times, interval_start, wear_duration=0):
        # TODO: add support for TST by using all the sleep blocks
        self.interval_start = interval_start

        if isinstance(times[0], np.datetime64):
            times = pd.to_datetime(times)

        self.times = times
        self.y_pred = y_pred
        # the exact day string in the format of YYYY-MM-DD
        self.day = interval_start.strftime("%Y-%m-%d")
        self.is_weekend = is_weekend(interval_start)

        self.summary = get_day_stats(
            y_pred,
            wear_duration,
            self.is_weekend,
            interval_start)
        self.summary = get_day_hourly_stats(times, y_pred, self.summary)

    def get_day(self):
        return self.day

    def get_y_pred(self):
        return self.y_pred

    def get_times(self):
        return self.times

    def get_summary(self):
        return self.summary

    def day2json(self):
        # should include day id when printing to json
        return {
            "day": self.day,
            "y_pred": self.y_pred,
            "times": self.times,
            "summary": self.summary,
        }

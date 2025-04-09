import argparse
import json
from collections import OrderedDict

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def collate_outputs(
    results_dir,
    collated_results_dir="collated_outputs/",
):
    """Collate all results files in <results_dir>.
    :param str results_dir: Root directory in which to search for result files.
    :param str collated_results_dir: Directory to write the collated files to.
    :return: Collated files written to <collated_results_dir>
    :rtype: void
    """

    print("Searching files...")

    info_files = []

    # Iterate through the files and append to the appropriate list based on the suffix
    for file in Path(results_dir).rglob('*'):
        if file.is_file():
            if file.name.endswith("info.json"):
                info_files.append(file)

    collated_results_dir = Path(collated_results_dir)
    collated_results_dir.mkdir(parents=True, exist_ok=True)

    # Collate Info.json files
    print(f"Collating {len(info_files)} Info files...")
    outfile = collated_results_dir / "info.csv.gz"
    collate_jsons(info_files, outfile)
    print('Collated info CSV written to', outfile)

    return


def collate_jsons(file_list, outfile, overwrite=True):
    """ Collate a list of JSON files into a single CSV file."""

    if overwrite and outfile.exists():
        print(f"Overwriting existing file: {outfile}")
        outfile.unlink()  # remove existing file

    df = []
    for file in tqdm(file_list):
        with open(file, 'r') as f:
            df.append(json.load(f, object_pairs_hook=OrderedDict))
    df = pd.DataFrame.from_dict(df)  # merge to a dataframe
    df = df.applymap(convert_ordereddict)  # convert any OrderedDict cell values to regular dict
    df.to_csv(outfile, index=False)

    return


def collate_csvs(file_list, outfile, overwrite=True):
    """ Collate a list of CSV files into a single CSV file."""

    if overwrite and outfile.exists():
        print(f"Overwriting existing file: {outfile}")
        outfile.unlink()  # remove existing file

    header_written = False
    for file in tqdm(file_list):
        df = pd.read_csv(file)
        df.to_csv(outfile, mode='a', index=False, header=not header_written)
        header_written = True

    return


def convert_ordereddict(value):
    """ Convert OrderedDict to regular dict """
    if isinstance(value, OrderedDict):
        return dict(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir',
                        help="Root directory in which to search for result files")
    parser.add_argument('--output', '-o',
                        default="collated-outputs/",
                        help="Directory to write the collated files to")
    args = parser.parse_args()

    return collate_outputs(
        results_dir=args.results_dir,
        collated_results_dir=args.output,
    )


if __name__ == '__main__':
    main()

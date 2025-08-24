import argparse
import json
from collections import OrderedDict

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def collate_outputs(
    results_dir,
    collated_results_dir="collated_outputs/",
    include_files=None,
):
    """Collate selected results files in <results_dir>.
    :param str results_dir: Root directory in which to search for result files.
    :param str collated_results_dir: Directory to write the collated files to.
    :param list include_files: List of file types to collate. Options: 'info', 'summary', 'predictions', 'sleep_block', 'day_summary'
    :return: Collated files written to <collated_results_dir>
    :rtype: void
    """
    
    # Default to only JSON files if none specified
    if include_files is None:
        include_files = ['info', 'summary']
    
    # Convert to set for faster lookups
    include_set = set(include_files)

    print("Searching files...")

    info_files = []
    summary_files = []
    predictions_files = []
    sleep_block_files = []
    day_summary_files = []

    # Iterate through the files and append to the appropriate list based on the suffix
    for file in Path(results_dir).rglob('*'):
        if file.is_file():
            if file.name.endswith("info.json") and 'info' in include_set:
                info_files.append(file)
            elif file.name.endswith("summary.json") and 'summary' in include_set:
                summary_files.append(file)
            elif file.name == "predictions.csv" and 'predictions' in include_set:
                predictions_files.append(file)
            elif file.name == "sleep_block.csv" and 'sleep_block' in include_set:
                sleep_block_files.append(file)
            elif file.name == "day_summary.csv" and 'day_summary' in include_set:
                day_summary_files.append(file)

    collated_results_dir = Path(collated_results_dir)
    collated_results_dir.mkdir(parents=True, exist_ok=True)

    # Collate files based on what was requested
    if info_files:
        print(f"Collating {len(info_files)} info files...")
        outfile = collated_results_dir / "info.csv.gz"
        collate_jsons(info_files, outfile)
        print('Collated info CSV written to', outfile)
    
    if summary_files:
        print(f"Collating {len(summary_files)} summary files...")
        outfile = collated_results_dir / "summary.csv.gz"
        collate_jsons(summary_files, outfile)
        print('Collated summary CSV written to', outfile)
    
    if predictions_files:
        print(f"Collating {len(predictions_files)} predictions files...")
        outfile = collated_results_dir / "predictions.csv.gz"
        collate_csvs_with_filepath(predictions_files, outfile)
        print('Collated predictions CSV written to', outfile)
    
    if sleep_block_files:
        print(f"Collating {len(sleep_block_files)} sleep_block files...")
        outfile = collated_results_dir / "sleep_block.csv.gz"
        collate_csvs_with_filepath(sleep_block_files, outfile)
        print('Collated sleep_block CSV written to', outfile)
    
    if day_summary_files:
        print(f"Collating {len(day_summary_files)} day_summary files...")
        outfile = collated_results_dir / "day_summary.csv.gz"
        collate_csvs_with_filepath(day_summary_files, outfile)
        print('Collated day_summary CSV written to', outfile)
    
    # Print summary of what was processed
    file_counts = {
        'info': len(info_files),
        'summary': len(summary_files), 
        'predictions': len(predictions_files),
        'sleep_block': len(sleep_block_files),
        'day_summary': len(day_summary_files)
    }
    processed_types = [k for k, v in file_counts.items() if v > 0 and k in include_set]
    if processed_types:
        print(f"\nSummary: Processed {processed_types} file types")
    else:
        print("\nWarning: No matching files found for the requested file types")

    return


def collate_jsons(file_list, outfile, overwrite=True):
    """ Collate a list of JSON files into a single CSV file."""

    if overwrite and outfile.exists():
        print(f"Overwriting existing file: {outfile}")
        outfile.unlink()  # remove existing file

    df = []
    for file in tqdm(file_list):
        with open(file, 'r') as f:
            j = json.load(f, object_pairs_hook=OrderedDict)
            j['filepath'] = file
            df.append(j)
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


def collate_csvs_with_filepath(file_list, outfile, overwrite=True):
    """ Collate a list of CSV files into a single CSV file, adding filepath column."""

    if overwrite and outfile.exists():
        print(f"Overwriting existing file: {outfile}")
        outfile.unlink()  # remove existing file

    header_written = False
    for file in tqdm(file_list):
        df = pd.read_csv(file)
        df['filepath'] = str(file)
        df.to_csv(outfile, mode='a', index=False, header=not header_written)
        header_written = True

    return


def convert_ordereddict(value):
    """ Convert OrderedDict to regular dict """
    if isinstance(value, OrderedDict):
        return dict(value)
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Collate asleep output files from multiple runs into single CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available file types:
  info        - Device and recording metadata (info.json)
  summary     - Aggregated sleep statistics (summary.json)
  predictions - Epoch-by-epoch sleep predictions (predictions.csv)
  sleep_block - Sleep onset/wake times (sleep_block.csv)
  day_summary - Daily sleep statistics (day_summary.csv)

Examples:
  # Collate info and summary files (default)
  collate_sleep results/
  
  # Collate only summary and sleep blocks
  collate_sleep results/ --include summary sleep_block
  
  # Collate all file types
  collate_sleep results/ --include info summary predictions sleep_block day_summary"""
    )
    parser.add_argument('results_dir',
                        help="Root directory in which to search for result files")
    parser.add_argument('--output', '-o',
                        default="collated-outputs/",
                        help="Directory to write the collated files to")
    parser.add_argument('--include', '-i',
                        nargs='+',
                        choices=['info', 'summary', 'predictions', 'sleep_block', 'day_summary'],
                        help="Specify which file types to collate. Default: info summary")
    args = parser.parse_args()

    return collate_outputs(
        results_dir=args.results_dir,
        collated_results_dir=args.output,
        include_files=args.include,
    )


if __name__ == '__main__':
    main()

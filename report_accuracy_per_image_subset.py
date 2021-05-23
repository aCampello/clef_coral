import argparse
import os
import json

import pandas as pd

from report_participants_accuracy import calculate_score_all_runs


if __name__ == "__main__":
    #gt_file = '../yam/plugins/test_set_2020_400_images_annotations_clef_v3/' \
    #          'annotations_test_task_2.csv'

    #runs_folder = '../yam/plugins/participant_runs_2020/coral-pixelwise'

    gt_file = '../yam/plugins/test_set_2020_400_images_annotations_clef_v3/' \
              'annotations_test_task_1.csv'
    runs_folder = '../yam/plugins/participant_runs_2020/coral-annotation'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int)
    task = parser.parse_args().task

    with open(os.path.join(runs_folder, 'submissions.json'), 'r') as f:
        runs_json = json.load(f)

    subset_files_folder = '../yam/plugins/'
    list_of_files = [
        'imageCLEFcoral2020_GT_similarLocation.csv',
        'imageCLEFcoral2020_GT_geographicallyDistinct.csv',
        'imageCLEFcoral2020_GT_sameLocation.csv',
        'imageCLEFcoral2020_GT_geographicallySimilar.csv'
    ]

    subsets = {}
    for file in list_of_files:
        with open(os.path.join(subset_files_folder, file)) as f:
            subsets[file[:-4]] = [line.split(' ')[0] for line in f]


    results = []
    for name, subset in subsets.items():
        current_results = calculate_score_all_runs(
            runs_json,  runs_folder, gt_file, subset=subset, task=task
        )

        current_results = [
            {**{'subset': name}, **result} for result in current_results
        ]

        results += current_results

    pd.DataFrame(results).to_csv(f'data/pixel_evaluation_per_location_'
                                  f'task_{task}.csv',
                                  index=False)










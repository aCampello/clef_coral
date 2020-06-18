import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from pixel_accuracy import run_pixel_evaluation


def calculate_score_all_runs(runs_json, runs_folder, gt_file, task=2,
                             subset=None):
    results = []
    for i, run in tqdm(enumerate(runs_json), total=len(runs_json)):
        participant_id = run['participant_id']
        participant_afiliation = run['participant_affiliation']
        run_id = run['id']
        file_name = run['submission_files'][0]['file_name']

        iou_per_substrate, iou_average = run_pixel_evaluation(
            gt_file=gt_file,
            run_file=os.path.join(runs_folder, 'submission_files', file_name),
            subset=subset,
            task=task
        )

        print(iou_average)
        current_result = {"participant_id": participant_id,
                          "participant_afiliation": participant_afiliation,
                          "run_id": run_id,
                          "iou_average": iou_average}
        current_result = {**current_result, **iou_per_substrate}

        results.append(current_result)

    return results


if __name__ == "__main__":
    gt_file = 'path_gt_file.csv'

    runs_folder = 'folder_with_multiple_csv_runs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int)
    task = parser.parse_args().task

    with open(os.path.join(runs_folder, 'submissions.json'), 'r') as f:
        runs_json = json.load(f)

    results = calculate_score_all_runs(runs_json, runs_folder, gt_file,
                                       task=task)

    pd.DataFrame(results).to_csv(f'data/pixel_evaluation_task_{task}.csv',
                                 index=False)

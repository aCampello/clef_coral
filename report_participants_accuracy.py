import os
import json

from pixel_accuracy import run_pixel_evaluation


def calculate_score_all_runs(runs_json, runs_folder, gt_file, task=2):
    results = []
    for run in runs_json:
        participant_id = run['participant_id']
        participant_afiliation = run['participant_affiliation']
        run_id = run['id']
        file_name = run['submission_files'][0]['file_name']

        iou_per_substrate, iou_average = run_pixel_evaluation(
            gt_file=gt_file,
            run_file=os.path.join(runs_folder, 'submission_files', file_name),
            task=task
        )
        current_result = {"participant_id": participant_id,
                          "participant_afiliation": participant_afiliation,
                          "run_id": run_id,
                          "iou_average": iou_average}
        current_result = {**current_result, **iou_per_substrate}

        results.append(current_result)

    return results


if __name__ == "__main__":
    gt_file = '../yam/plugins/test_set_2020_400_images_annotations_clef_v3/' \
              'annotations_test_task_2.csv'

    runs_folder = '../yam/plugins/participant_runs_2020/coral-pixelwise'

    with open(os.path.join(runs_folder, 'submissions.json'), 'r') as f:
        runs_json = json.load(f)[:2]

    calculate_score_all_runs(runs_json, runs_folder, gt_file, task=2)

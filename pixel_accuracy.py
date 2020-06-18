"""Calculates the pixel IoU per substrate of a ClefCoral run"""
from collections import defaultdict

import cv2
import numpy as np

SUBSTRATE_LIST = [
    'c_algae_macro_or_leaves',
    'c_fire_coral_millepora',
    'c_hard_coral_boulder',
    'c_hard_coral_branching',
    'c_hard_coral_encrusting',
    'c_hard_coral_foliose',
    'c_hard_coral_mushroom',
    'c_hard_coral_submassive',
    'c_hard_coral_table',
    'c_soft_coral',
    'c_soft_coral_gorgonian',
    'c_sponge',
    'c_sponge_barrel'
]

SUBSTRATE_TO_IDX = {
    substrate: idx+1 for idx, substrate in enumerate(SUBSTRATE_LIST)
}


def run_pixel_evaluation(gt_file, run_file, task=2, subset=None):
    """
    Wraps all functionalities for pixel evaluation

    Args:
        gt_file: A path for the gt file
        run_file: A path for a file with predictions
        task: the clef coral task (1 or 2)
        subset: whether to run evaluation only on a subset of images. If unset
        uses all images

    Returns:
        iou_per_substrate(dict), iou_average(float)

    """

    gt_task_2 = read_annotations_gt(gt_file, task=task, subset=subset)
    participant_task_2 = read_participant_run(run_file, task=task,
                                              subset=subset)

    pixel_annotations_gt = convert_pixel_images(gt_task_2)
    pixel_annotations_participant = convert_pixel_images(participant_task_2)

    return calculate_iou_metrics_run(pixel_annotations_gt,
                                     pixel_annotations_participant)


def read_annotations_gt(file, task=2, subset=None):
    """
    Given a file in the clef coral format, reads the polygons and puts into a
    multilevel dictionary

    Args:
        file: path to annotations
        task: 1 or 2 (the two clef coral tasks)

    Returns:
        Multi-level dictionary where each entry,
        list_of_polygons[polygon_file][substrate], contains a list of all
        polygons of that file and substrate
    """
    annotation_dict = defaultdict(lambda: defaultdict(list))

    with open(file, 'r') as f:
        for row in f:
            row_vec = row[:-1].split(' ')
            # row_vec[0] is the id in the image
            if subset and row_vec[0] not in subset:
                continue
            if task == 1:
                dimensions = [int(x) for x in row_vec[4:]]
                x_min = dimensions[2]
                y_min = dimensions[3]
                x_max = dimensions[0] + dimensions[2]
                y_max = dimensions[1] + dimensions[3]
                polygon = [(x_min, y_min),
                           (x_min, y_max),
                           (x_max, y_max),
                           (x_max, y_min)]
                annotation_dict[row_vec[0]][row_vec[2]] += [polygon]
            elif task == 2:
                annotation_dict[row_vec[0]][row_vec[2]] += [
                    [(int(x), int(y))
                     for x, y in zip(row_vec[4:][::2], row_vec[4:][1::2])]
                ]

    return annotation_dict


def read_participant_run(file, task=2, subset=None):
    """
    Given a participant submission file in the clef format, reads the
    polygons and puts into a multilevel dictionary

    Args:
        file: path to a participant run
        task: 1 or 2 (the two clef coral tasks)

    Returns:
        Multi-level dictionary where each entry,
        list_of_polygons[polygon_file][substrate], contains a list of all
        polygons of that file and substrate
    """
    annotation_dict = defaultdict(lambda: defaultdict(list))

    with open(file, 'r') as f:
        for row in f:
            row_unrolled = row[:-1].split(';')
            image_id = row_unrolled[0]
            if subset and image_id not in subset:
                continue
            predictions = row_unrolled[1:]

            for prediction in predictions:
                substrate, polygons = prediction.split(' ')
                for polygon in polygons.split(','):
                    confidence, values = polygon.split(':')
                    if task == 1:
                        width_height, x_min, y_min = values.split('+')
                        width, height = width_height.split('x')
                        width, height = int(width), int(height)
                        x_min, y_min = int(x_min), int(y_min)
                        x_max = x_min + width
                        y_max = y_min + height

                        processed_polygon = [(x_min, y_min), (x_min, y_max),
                                             (x_max, y_max), (x_max, y_min)]
                    elif task == 2:
                        values_list = values.split('+')
                        processed_polygon = [
                            (int(x), int(y))
                            for x, y in zip(values_list[::2], values_list[1::2])
                        ]

                    annotation_dict[image_id][substrate] += [processed_polygon]

    return annotation_dict


def convert_pixel_images(annotation_dict, shape=(4032, 3024)):
    """
    Given an annotation dictionary,
    generates numpy arrays filling the polygons with integers representing
    the ID of a substrate (as in the global variable SUBSTRATE_TO_IDX)

    Args:
        annotation_dict: An annotation dictionary
        (as returned for instance by .read_annotations)
        shape: the shape of the image file

    Returns:
        Dictionary of numpy arrays replacing pixels by its substrate class,
        and 0 for background
    """

    pixel_images = {}
    for image_name, annotations in annotation_dict.items():
        pixel_annotations = np.zeros(shape=shape, dtype=np.int32)
        for substrate, idx in SUBSTRATE_TO_IDX.items():
            if annotations.get(substrate):
                # Open CV only works with np arrays, so need to convert input
                points = [np.array(pts) for pts in annotations[substrate]]
                pixel_annotations = cv2.fillPoly(
                    pixel_annotations, points, idx
                )
        pixel_images[image_name] = pixel_annotations

    return pixel_images


def calculate_agreement(pixel_image_1, pixel_image_2, substrate_idx=None):
    """
    Given two images containing pixel annotations, calculates agreement.
    If a substrate_idx is provided, calculates IoU over that substrate,
    otherwise calculates pixel accuracy
    
    Args:
        pixel_image_1: a numpy array  
        pixel_image_2: a numpy array
        substrate_idx: a valid substrate index

    Returns:
        Agreement metric (accuracy or IoU)
    """

    intersection = ((pixel_image_1 == substrate_idx) *
                    (pixel_image_2 == substrate_idx)).sum()

    union = ((pixel_image_1 == substrate_idx) +
             (pixel_image_2 == substrate_idx)).sum()

    # If there is no pixels with that class, IoU is undefined
    return intersection, union


def calculate_iou_metrics_run(pixel_annotations_gt,
                              pixel_annotations_participant):
    """Given datasets (predicted and gt) for pixel annotations, returns IoU"""
    intersection_per_substrate = defaultdict(int)
    union_per_substrate = defaultdict(int)

    for image in pixel_annotations_gt.keys():
        for substrate_name, substrate_idx in SUBSTRATE_TO_IDX.items():
            intersection, union = \
                calculate_agreement(pixel_annotations_gt[image],
                                    pixel_annotations_participant.get(image, []),
                                    substrate_idx=substrate_idx)
            if union:
                intersection_per_substrate[substrate_name] += intersection
                union_per_substrate[substrate_name] += union

    eps = 10e-16
    # Ads epsilon to the denominator to avoid division by zero
    iou_per_substrate = {
        substrate_name: intersection_per_substrate[substrate_name] /
                        (union_per_substrate[substrate_name] + eps)
        for substrate_name in SUBSTRATE_LIST
    }

    iou_average = sum(intersection_per_substrate.values()) / \
                  sum(union_per_substrate.values())

    return iou_per_substrate, iou_average


if __name__ == "__main__":
    run_file = "path_to_run.csv"
    gt_file = 'path_to_groundtruth_file.csv'

    iou_per_substrate, iou_average = run_pixel_evaluation(gt_file, run_file)

    pretty_output = "\n".join(f"{substrate}: {iou}"
                              for substrate, iou in iou_per_substrate.items())
    print("IoU:")
    print(pretty_output)

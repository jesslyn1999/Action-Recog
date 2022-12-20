import os
from pathlib import Path as _Path
from test_mod.system.finalization import image_folder_bbox, process_image_folder_filter_bbox
import time
import torch
import threading


def main():
    DATASET = 'jhmdb'  # ucf24 or jhmdb or random
    DATASET_FOLDER = {
        'jhmdb': 'jhmdb21',
        'ucf24': 'ucf24'
    }

    BASE_FOLDER = "/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/" + DATASET_FOLDER[DATASET]

    # image_folder = "catch/Frisbee_catch_f_cm_np1_ri_med_0"

    image_folder = ""


    yolo_label_folder = os.path.join(BASE_FOLDER, "yolo_det_labels", image_folder)
    yowo_label_folder = os.path.join(BASE_FOLDER, "det_labels", image_folder)
    gt_folder = os.path.join(BASE_FOLDER, "groundtruths")

    final_label_folder = os.path.join(BASE_FOLDER, "filtered_det_labels", image_folder)
    final_video_path = os.path.join(BASE_FOLDER, "output_yowo_yolo_video", image_folder)
    final_vid_yowo_path = os.path.join(BASE_FOLDER, "output_yowo_video", image_folder)

    input_image_folder = os.path.join(BASE_FOLDER, "rgb-images", image_folder)

    test_list_video = "/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/" + DATASET_FOLDER[DATASET] + "/splitfiles/testlist_video.txt"

    # process_image_folder_filter_bbox(final_label_folder, yolo_label_folder, yowo_label_folder, isGT=False)
    # image_folder_bbox(input_image_folder, final_video_path, det_folder=final_label_folder, gt_folder=gt_folder)
    image_folder_bbox(
        input_image_folder, final_video_path, det_folder=final_label_folder, 
        gt_folder=gt_folder, test_list_video_file=test_list_video, dataset=DATASET
    )
    # image_folder_bbox(
    #     input_image_folder, final_vid_yowo_path, det_folder=yowo_label_folder, 
    #     gt_folder=gt_folder, test_list_video_file=test_list_video, dataset=DATASET
    # )

    pass


if __name__ == "__main__":
    main()
import glob
from PIL import Image
from typing import List
import numpy as np
import test_mod.system.bbox as bb_util

from core.utils import bbox_iou, mkdir

import cv2 as _cv
import os
from pathlib import Path as _Path


ALL_ACTION_LABELS = {
    "random": [
        "falling_down", "chest_pain", "pushing", "touch_pocket",
        "hit_with_object", "wield_knife", "shoot_with_gun", 
        "support_somebody", "attacked_by_gun", "run"
    ],
    "ucf24": [
        "Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling",
        "Diving", "Fencing", "FloorGymnastics", "GolfSwing", "HorseRiding",
        "IceDancing", "LongJump", "PoleVault", "RopeClimbing", "SalsaSpin",
        "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling", "Surfing",
        "TennisSwing", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog"
    ],
    "jhmdb": [
        "brush_hair", "catch", "clap", "climb_stairs", "golf",
        "jump", "kick_ball", "pick", "pour", "pullup", "push",
        "run", "shoot_ball", "shoot_bow", "shoot_gun", "sit",
        "stand", "swing_baseball", "throw", "walk", "wave"
    ]
}



def filter_bbox(yolo_bboxes, yowo_bboxes, out_label_path: str, iou_thres=0.3):
    num_gts = len(yolo_bboxes)
    selected_final_bboxes = []

    if not os.path.exists(os.path.dirname(out_label_path)):
        os.makedirs(os.path.dirname(out_label_path), exist_ok=True)

    for i in range(num_gts):
        # x1, y1, x2, y2, localization conf
        box_gt = yolo_bboxes[i][:5]
        best_iou = 0
        best_j = -1

        for j in range(len(yowo_bboxes)):
            dboxes = yowo_bboxes[j]

            # print(box_gt)
            # print(dboxes)
            iou = bbox_iou(box_gt, dboxes, x1y1x2y2=True)  # iou > 0,5 = TP, iou < 0.5 = FP
            # print(iou)

            if iou > best_iou:
                best_j = j
                best_iou = iou
            elif iou == best_iou and best_iou != 0:
                print(
                    "OMG UNBELIEVABLE! NEED TO CHANGE CODE with IOU={} : {} and {}".format(iou, yowo_bboxes[best_j][:4],
                                                                                           yowo_bboxes[j][:4]))

        if best_j != -1 and best_iou > iou_thres:
            selected_final_bboxes.append(box_gt[:5] + yowo_bboxes[best_j][5:])
            # selected_final_bboxes.append(box_gt[:4] + [1] + yowo_bboxes[best_j][5:])

        # if best_iou > iou_thresh:
        #     total_detected += 1
        #     if int(boxes[best_j][6]) == box_gt[6]:
        #         correct_classification += 1
        #
        # if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
        #     correct = correct + 1

    l_bboxes = len(selected_final_bboxes)

    with open(out_label_path, 'w+') as writer:
        for idx, bbox in enumerate(selected_final_bboxes):
            if idx == l_bboxes - 1:
                writer.write("{}".format(" ".join(["{:g}".format(elmt) for elmt in bbox])))
            else:
                writer.write("{}\n".format(" ".join(["{:g}".format(elmt) for elmt in bbox])))


def process_label_video(video_path: str, out_label_folder: str, yolo_label_folder: str, yowo_label_folder: str):
    cap = _cv.VideoCapture(video_path)
    mkdir(out_label_folder)

    n_frames = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
    n_digits = len(str(n_frames))

    cur_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cur_frame += 1

        targeted_filename = "{}.txt".format(str(cur_frame).zfill(n_digits))
        yolo_label_path = os.path.join(yolo_label_folder, targeted_filename)
        yowo_label_path = os.path.join(yowo_label_folder, targeted_filename)
        final_label_path = os.path.join(out_label_folder, targeted_filename)

        yolo_bboxes = read_file_lines(yolo_label_path)
        yowo_bboxes = read_file_lines(yowo_label_path)

        filter_bbox(yolo_bboxes, yowo_bboxes, final_label_path)

    cap.release()

def process_image_folder_filter_bbox(out_label_folder: str, yolo_label_folder: str, yowo_label_folder: str, isGT: bool=False):
    os.chdir(yowo_label_folder)
    sub_listed_files = os.listdir(yowo_label_folder)
    sub_folders_p = list(filter(os.path.isdir, sub_listed_files))
    sub_folders = []

    print(sub_folders_p)

    for sub_folder in sub_folders_p:
        sub_listed_files = os.listdir(os.path.join(yowo_label_folder, sub_folder))
        sub_listed_files = [os.path.join(sub_folder, f) for f in sub_listed_files]
        sub_folders += list(filter(os.path.isdir, sub_listed_files))

    if len(sub_folders) == 0: # only files
        sub_folders = ['']

    
    for sub_folder in sub_folders:
        label_files = []
        label_files = glob.glob(os.path.join(sub_folder, "*.txt"))
        label_files.sort()

        check_dir = os.path.join(yolo_label_folder, sub_folder)

        if not isGT and not os.path.exists(check_dir):
            print("Folder yolo is not exist: ", check_dir)
            continue

        for label_file in label_files:
            targeted_filename = label_file

            yolo_label_path = os.path.join(yolo_label_folder, targeted_filename)

            if isGT:
                yolo_label_path = os.path.join(yolo_label_folder, targeted_filename.replace("/", "_"))
                
            yowo_label_path = os.path.join(yowo_label_folder, targeted_filename)
            final_label_path = os.path.join(out_label_folder, targeted_filename)

            yolo_bboxes = read_file_lines(yolo_label_path, isGT)
            yowo_bboxes = read_file_lines(yowo_label_path)

            filter_bbox(yolo_bboxes, yowo_bboxes, final_label_path)


def process_image(
        img: Image.Image, gt_label_path: str = None, det_label_path: str = None,
        is_usual: bool = True, is_demo: bool = False, num_labels=3, dataset: str=None) -> Image.Image:
    gt_bboxes = []
    if gt_label_path:
        gt_bboxes = read_file_lines(gt_label_path)

    det_bboxes = []
    if det_label_path:
        det_bboxes = read_file_lines(det_label_path)

    if is_usual:
        img = image_bbox(img, gt_bboxes, det_bboxes, num_labels, dataset=dataset)

    if is_demo:
        img = image_bbox(img, gt_bboxes, det_bboxes, 10)

    return img


def image_bbox(img: Image.Image, gt_bboxes: List[List[float]], det_bboxes: List[List[float]], num_show_label: int, dataset:str='random'):
    """
    image bbox for system output
    """
    color_list: List[str] = list(bb_util.COLOR_NAME_TO_RGB.keys())
    np_img = np.array(img)

    for idx, gt_box in enumerate(gt_bboxes):
        x1, y1, x2, y2 = gt_box[1:5]
        label = ALL_ACTION_LABELS[dataset][int(gt_box[0] - 1)]
        bb_util.add(np_img, x1, y1, x2, y2, label, "lime", place_label="bottom")

    for idx, d_box in enumerate(det_bboxes):
        det_labels = np.zeros(len(ALL_ACTION_LABELS[dataset]))
        itr = iter(d_box[5:])
        for cls_conf, cls_label in zip(itr, itr):
            det_labels[int(cls_label)] = cls_conf * 100

        x1, y1, x2, y2 = d_box[:4]
        desc_sort_idxs = det_labels.argsort()[-1:-1 * (num_show_label + 1):-1]
        label = "\n".join(["{} ... {:g}".format(ALL_ACTION_LABELS[dataset][tmp_idx], det_labels[tmp_idx])
                           for tmp_idx in desc_sort_idxs])
        bb_util.add(np_img, x1, y1, x2, y2, label, color_list[desc_sort_idxs[0]],
                    place_label="top")

    return Image.fromarray(np_img)


def video_bbox(video_path: str, out_video_path: str, gt_folder: str = None, det_folder: str = None):
    """
    video bbox for system output
    """
    cap = _cv.VideoCapture(video_path)
    width = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(_cv.CAP_PROP_FPS))
    n_frames = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
    n_digits = len(str(n_frames))

    cv_writer = _cv.VideoWriter(out_video_path, _cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

    cur_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cur_frame += 1

        frame = _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)

        targeted_filename = "{}.txt".format(str(cur_frame).zfill(n_digits))
        gt_label_path = None
        if gt_folder:
            # print("HEYYY: ", str([path for path in _Path(gt_folder).rglob("*{}.txt".format(cur_frame))][0]))

            gt_label_path = str([path for path in _Path(gt_folder).rglob("*{}.txt".format(cur_frame))
                                 if int(str(path.stem)) == cur_frame][0])
        det_label_path = os.path.join(det_folder, targeted_filename)

        img = process_image(frame, gt_label_path, det_label_path, is_usual=True)

        img = np.array(img)[:, :, ::-1]
        cv_writer.write(img)

    cap.release()


def image_folder_bbox(
    p_img_folder_path: str, out_video_folder: str, 
    gt_folder: str = None, det_folder: str = None, 
    test_list_video_file: str=None, dataset: str = None
    ):
    """
    video bbox for system output
    """
    os.chdir(p_img_folder_path)
    sub_listed_files = os.listdir(p_img_folder_path)

    sub_folders_p = list(filter(os.path.isdir, sub_listed_files))
    sub_folders = []

    print(sub_folders_p)

    if not test_list_video_file:
        for sub_folder in sub_folders_p:
            sub_listed_files = os.listdir(os.path.join(p_img_folder_path, sub_folder))
            sub_listed_files = [os.path.join(sub_folder, f) for f in sub_listed_files]
            sub_folders += list(filter(os.path.isdir, sub_listed_files))
    else:
        with open(test_list_video_file, 'r') as file:
            video_lines = file.read().splitlines() 
            sub_folders += video_lines


    if len(sub_folders) == 0: # only files
        sub_folders = ['']

    for sub_folder in sub_folders:
        img_folder_path = sub_folder
        image_files = glob.glob(os.path.join(img_folder_path, "*.png")) + glob.glob(os.path.join(img_folder_path, '*.jpg'))
        image_files.sort()

        frame = Image.open(image_files[0])

        width, height = frame.size
        fps = 20

        out_video_path = os.path.join(out_video_folder, sub_folder+".mp4")

        if not os.path.exists(os.path.dirname(out_video_path)):
            os.makedirs(os.path.dirname(out_video_path), exist_ok=True)


        # cv_writer = _cv.VideoWriter(out_video_path, _cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        cv_writer = _cv.VideoWriter(out_video_path, _cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for image_file in image_files:
            frame = Image.open(image_file)
            base_name = image_file.rsplit('.', 1)[0]

            targeted_filename = "{}.txt".format(base_name)
            gt_label_path = None
            if gt_folder:
                # print("HEYYY: ", str([path for path in _Path(gt_folder).rglob("*{}.txt".format(cur_frame))][0]))
                gt_label_path = os.path.join(gt_folder, targeted_filename.replace("/", "_"))
            det_label_path = os.path.join(det_folder, targeted_filename)

            img = process_image(frame, gt_label_path, det_label_path, is_usual=True, dataset=dataset)

            img = np.array(img)[:, :, ::-1]
            # img = np.array(img)[:, :, :]
            cv_writer.write(img)

        cv_writer.release()



def read_file_lines(file_path, isGT=False):
    if not file_path or not os.path.exists(file_path):
        return []
    f = open(file_path)
    lines = [line for line in f.read().splitlines() if line]
    lines = [line for line in lines if len(line) != 0]
    unique_lines = list(dict.fromkeys(lines))
    split_lines = []
    for line in unique_lines:
        if not isGT:
            split_lines.append([float(element) for element in line.split(" ") if element])
        else:
            elmts = [float(element) for element in line.split(" ") if element]
            split_lines.append([elmts[1], elmts[2], elmts[3], elmts[4], 1])
    f.close()
    return split_lines


def main():
    # gt_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_yolo"
    # det_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_complete"
    # final_det_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_final"
    # frame_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/frames"
    #
    # manage_det(gt_folder, det_folder, frame_folder, final_det_folder)

    img_path = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/frames/" \
               "complete_chest_pain/S004C001P007R001A045_rgb/00001.png"
    gt_file = ""
    det_file = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_final/" \
               "complete_chest_pain/S004C001P007R001A045_rgb/00001.txt"
    process_image(Image.open(img_path), gt_file, det_file, is_demo=True)
    pass


if __name__ == '__main__':
    main()

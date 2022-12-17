import cv2
from pathlib import Path as _Path
import os


def vid_to_frames(input_vid, output_folder):
    vid_title = input_vid.split('/')[-1]
    video_path_stem = _Path(vid_title).stem
    vidcap = cv2.VideoCapture(input_vid)
    success,image = vidcap.read()
    count = 1

    output_vid_folder = os.path.join(output_folder, video_path_stem)

    if not os.path.exists(output_vid_folder):
        os.makedirs(output_vid_folder, exist_ok=True)

    while success:
        cv2.imwrite(os.path.join(output_vid_folder, '{:05d}.jpg'.format(count)), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1


if __name__ == '__main__':
    vid_to_frames(
        '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/random/video/fencing-fps10.mp4',
        '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/random/rgb-images'
    )


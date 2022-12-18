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
    # vid_to_frames(
    #     '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/random/video/fencing-fps10.mp4',
    #     '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/random/rgb-images'
    # )
    img = cv2.imread("/Users/jesslyn1999/Downloads/Rename_Images/clap/103_years_old_japanese_woman__Nao_is_clapping_with_piano_music_by_beethoven_clap_u_cm_np1_fr_med_1/00007.png")
    dimensions = img.shape
    print(dimensions)


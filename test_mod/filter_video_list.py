import glob
import os
import shutil

def filter_video_to_process():
    # suffixes = [
    #     "_g01_c01",
    #     "_g02_c05",
    #     "_g03_c03",
    #     "_g04_c04",
    #     "_g05_c05",
    #     "_g06_c02",
    #     "_g07_c01",
    #     "_g07_c02"
    # ]
    suffixes = [
        "_g02_c05",
        "_g05_c05",
        "_g06_c02",
        "_g07_c01"
    ]
    with open('test_mod/list_vid.txt', 'r') as file:
        video_lines = file.read().splitlines() 
        result_lines = []

        splitted_video_lines = [video_line[:-8] for video_line in video_lines]

        categories = list(set(splitted_video_lines))

        for category in categories:
            for suffix in suffixes:
                video_dir = category + suffix
                if video_dir in video_lines:
                    result_lines.append(video_dir)
                else:
                    video_dir = video_dir[:-1] + '1'
                    if video_dir in video_lines:
                        result_lines.append(video_dir)
                    else:
                        print("check: ", video_dir)
    
    result_lines = sorted(result_lines)
            
    with open('test_mod/your_file.txt', 'w') as f:
        for line in result_lines:
            f.write(f"{line}\n")


def generate_filtered_gt_folder(testlist_video_path, gt_folder, output_gt_folder):
    os.chdir(gt_folder)
    gt_files = glob.glob("*.txt")
    gt_files.sort()
    gt_file_prefixes = [file.rsplit("_", 1)[0] for file in gt_files]
    with open(testlist_video_path, 'r') as file:
        video_lines = file.read().splitlines()
        video_lines = sorted([line.replace("/", "_") for line in video_lines])

    filtered_gt_files = []
    
    for gt_file, gt_file_prefix in zip(gt_files, gt_file_prefixes):
        if gt_file_prefix in video_lines:
            filtered_gt_files.append(gt_file)

    for gt_file in filtered_gt_files:
        src = os.path.join(gt_folder, gt_file)
        dst = output_gt_folder
        shutil.copy(src, dst)


if __name__ == '__main__':
    # filter_video_to_process()
    generate_filtered_gt_folder(
        '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/ucf24/splitfiles/testlist_video.txt',
        '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/ucf24/groundtruths_complete',
        '/Users/jesslyn1999/Documents/master/lectures/人工智能技术前沿与产业应用/big_project/data/ucf24/groundtruths'
    )
    pass



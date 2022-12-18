import glob
import os

def main():
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

if __name__ == '__main__':
    main()


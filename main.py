from __future__ import print_function
import os
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets import list_dataset
from datasets.ava_dataset import Ava 
from core.optimization import *
from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss, RegionLoss_Ava
from core.model import YOWO, get_fine_tuning_parameters

from test_mod.test_video import test_random



def main():

    ####### Load configuration arguments
    # ---------------------------------------------------------------
    args  = parser.parse_args()
    cfg   = parser.load_config(args)


    ####### Check backup directory, create if necessary
    # ---------------------------------------------------------------
    if not os.path.exists(cfg.BACKUP_DIR):
        os.makedirs(cfg.BACKUP_DIR)


    # Create model
    # ---------------------------------------------------------------
    mps_device = torch.device("mps")
    cpu_device = torch.device("cpu")
    model = YOWO(cfg)
    # model = model.cuda()
    model = model.to(mps_device)
    model = nn.DataParallel(model, device_ids=None) # in multi-gpu case
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging('Total number of trainable parameters: {}'.format(pytorch_total_params))


    # Load resume path 

    if cfg.TRAIN.RESUME_PATH:
        print("===================================================================")
        print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
        checkpoint = torch.load(cfg.TRAIN.RESUME_PATH, map_location=cpu_device)
        model.load_state_dict(checkpoint['state_dict'])
        # model.eval()
        print("Model loaded!")
        print("===================================================================")
        del checkpoint


    seed = int(time.time())
    torch.manual_seed(seed)
    use_cuda = False
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: add to config e.g. 0,1,2,3
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


    ####### Create optimizer
    # ---------------------------------------------------------------
    # parameters = get_fine_tuning_parameters(model, cfg)
    # optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    best_score   = 0 # initialize best score
    # optimizer = optim.SGD(parameters, lr=cfg.TRAIN.LEARNING_RATE/batch_size, momentum=cfg.SOLVER.MOMENTUM, dampening=0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)



    dataset = cfg.TRAIN.DATASET
    assert dataset == 'ucf24' or dataset == 'jhmdb21' or dataset == 'ava' or dataset == 'random', 'invalid dataset'


    video_path = cfg.LISTDATA.BASE_PTH
    gt_label_dir = cfg.LISTDATA.BASE_PTH


    test_dataset  = list_dataset.Random_UCF_JHMDB_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, dataset=dataset,
                       shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                       transform=transforms.Compose([transforms.ToTensor()]), 
                       train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)

    
    # test_dataset  = list_dataset.SystemDataset(
    #     video_path, gt_label_dir, shape=(224, 224),
    #     frame_transform=transforms.Compose([transforms.ToTensor()]),
    #     clip_dur=16
    # )

    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)

    # test  = getattr(sys.modules[__name__], 'test_random')

    score = test_random(cfg, 0, model, test_loader)


if __name__ == "__main__":
    main()
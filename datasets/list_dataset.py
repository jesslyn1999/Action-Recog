#!/usr/bin/python
# encoding: utf-8

import os
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image

from datasets.clip import *

from core.utils import count_frames_path
from pathlib import Path as _Path

from datasets.clip import load_data_detection, load_system_data

from torch.utils.data.dataset import T_co


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class UCF_JHMDB_Dataset(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset='ucf24', shape=None,
                 transform=None, target_transform=None, 
                 train=False, clip_duration=16, sampling_rate=1):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base  # folder path which contains label folder
        self.dataset = dataset
        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.clip_duration = clip_duration
        self.sampling_rate = sampling_rate

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train: # For Training
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            clip, label = load_data_detection(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.shape, self.dataset, jitter, hue, saturation, exposure)

        else: # For Testing
            frame_idx, clip, label = load_data_detection(self.base_path, imgpath, False, self.clip_duration, self.sampling_rate, self.shape, self.dataset)
            clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (clip, label)
        else:
            return (frame_idx, clip, label)


class Random_Dataset(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset='random', shape=None,
                 transform=None, target_transform=None, 
                 train=False, clip_duration=16, sampling_rate=1):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base
        self.dataset = dataset
        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.clip_duration = clip_duration
        self.sampling_rate = sampling_rate

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train: # For Training
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            clip, label = load_data_detection(self.base_path, imgpath,  self.train, self.clip_duration, self.sampling_rate, self.shape, self.dataset, jitter, hue, saturation, exposure)

        else: # For Testing
            frame_idx, clip, label = load_data_detection(self.base_path, imgpath, False, self.clip_duration, self.sampling_rate, self.shape, self.dataset)
            clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (clip, label)
        else:
            return (frame_idx, clip, label)


class SystemDataset(Dataset):
    def __init__(self, video_path: str = None, label_folder: str = None,
                 shape: tuple = (224, 224), frame_transform: Compose = None, label_transform=None,
                 clip_dur: int = 16, sampling_rate=1) -> None:
        self.video_path = video_path
        self.label_folder = label_folder
        self.shape = shape
        self.frame_transform = frame_transform
        self.label_transform = label_transform
        self.clip_dur = clip_dur
        self.sampling_rate = sampling_rate
        self._num_samples = count_frames_path(video_path)
        super().__init__()

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index) -> T_co:
        """
        :return:
            index start from zero, indicate the index of the frames in video
        """
        assert index <= len(self), 'SystemDataset index range error'

        label_path = ""
        if self.label_folder:
            label_path = str([path for path in _Path(self.label_folder).rglob("*{}.txt".format(index + 1))
                              if int(str(path)) == index + 1][0])

        t0 = time.time()

        clip, label, key_frame = load_system_data(self.video_path, index, self.clip_dur, self.sampling_rate, label_path)
        t1 = time_synchronized()

        clip = [img.resize(self.shape) for img in clip]

        t2 = time_synchronized()

        if self.frame_transform is not None:
            clip = [self.frame_transform(img) for img in clip]

        if clip:
            clip = torch.cat(clip, 0).view((-1, 3) + self.shape).permute(1, 0, 2, 3)
            len_clip = clip.size(1)
            if len_clip != self.clip_dur:
                ret_clip = torch.empty(3, self.clip_dur, *self.shape)
                ret_clip[:, :len_clip, :, :] = clip
                clip = ret_clip
        else:
            clip = torch.empty(3, self.clip_dur, *self.shape)
            len_clip = 0

        t3 = time_synchronized()

        if self.label_transform is not None:
            label = self.label_transform(label)

        print(f'{index}.all_time: ({t3 - t0:.3f}s), break down: 1.({t1 - t0:.3f}s). 2.({t2 - t1:.3f}s). '
              f'3.({t3 - t2:.3f}s).')

        return index, self._num_samples, clip, label, len_clip, key_frame
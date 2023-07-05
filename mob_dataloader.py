import os
import cv2
import random
import numpy as np
import skvideo.io
import pandas as pd
import os.path as osp
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvideotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, \
    ColorJitter, Normalize, RandomHorizontalFlip, CenterCrop
from torchvideotransforms.volume_transforms import ClipToTensor

class MOBDataset(Dataset):
    """MOB dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """

    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=1, train_file=None,
                 test_file=None, max_sr=1):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        self.train_file = train_file
        self.test_file = test_file

        self.max_sr = max_sr

        if self.train:
            train_split_path =  self.train_file
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = self.test_file
            self.test_split = pd.read_csv(test_split_path, header=None, sep=' ')[0]


    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        #print('_'*100)
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        #print('parent_file',videoname)
        class_idx = videoname.split('/')[-1]
        vid_name = videoname.split('/')[1]# + '_' + videoname.split('_')[2]
        folder_name = videoname.split('/')[0]
        #print('folder_name',folder_name)
        
        # exit()
        #print(vid_name,class_idx)
        #exit()
        class_idx = int(class_idx) - 1
        sample_rate = self.max_sr
        filename = os.path.join(self.root_dir,folder_name,vid_name)
        #print('filename',filename)
        length = len(os.listdir(filename))
        #print('length',length)
        #exit()
        try:
            clip_start = random.randint(1, length - (self.clip_len * sample_rate))
        except:
            try:
                clip_start = random.randint(1, length + 1 - (self.clip_len * sample_rate))
            except:
                clip_start = 1
        clip = self.loop_load_rgb(filename, clip_start, sample_rate, self.clip_len, length)

        #print('clip',clip.shape)
        #exit()
        sample = {}
        # random select a clip for train
        if self.train:
            # clip_start = random.randint(0, length - self.clip_len)
            # clip = videodata[clip_start: clip_start + self.clip_len]  # uint8
            # print(clip.shape, clip.dtype)
            # exit()

            if self.transforms_:
                trans_clip_1 = []
                trans_clip_2 = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    framex = self.toPIL(frame)  # PIL image
                    # framex = self.transforms_(framex) # tensor [C x H x W]
                    trans_clip_1.append(framex)
                # (T x C X H x W) to (C X T x H x W)
                # clip_1 = torch.stack(trans_clip_1).permute([1, 0, 2, 3])
                clip_1 = self.transforms_(trans_clip_1)

            else:
                clip_1 = torch.tensor(clip)

            #print(clip.shape, class_idx)
            #exit()
            
            
            
            sample['data_1'] = clip_1
            sample['class_id'] = torch.tensor(class_idx)
     
            return sample

        # sample several clips for test
        else:
            # print(filename)
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len / 2, length - self.clip_len, self.test_sample_num):
                # clip_start = int(i - self.clip_len/2)
                # clip = videodata[clip_start: clip_start + self.clip_len]
                clip = list()
                clip_start = int(i - self.clip_len / 2) + 1
                clip_end = clip_start + self.clip_len
                # print(clip_start, clip_end)

                for ind_frame in range(clip_start, clip_end):
                    #print(osp.join(filename, str(ind_frame).zfill(4) + '.jpg'))
                    #exit()
                    # print(filename, str(ind_frame).zfill(5) + '.png')
                    # exit()
                    # cur_img_path = os.path.join(
                    #     video_dir,
                    #     # str(ind_frame).zfill(5) + '.png'
                    #     "{:04}.jpg".format(start_frame + idx * sample_rate))
                    frm = cv2.cvtColor(cv2.imread(osp.join(filename, str(ind_frame).zfill(4) + '.jpg')),
                                       cv2.COLOR_BGR2RGB)
                    # frm = cv2.resize(frm, (112, 112))
                    clip.append(frm)
                    # print(frm.shape)
                clip = np.array(clip)

                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame)  # PIL image
                        # frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    # clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                    clip = self.transforms_(trans_clip)
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(class_idx))

            # print(all_idx, class_idx)
            # sample['data'] = all_clips[0]
            sample['data'] = all_clips
            # sample['pid'] = all_idx
            sample['class_id'] = torch.tensor(class_idx)
            # print(sample['data'].shape, sample['pid'])
            return sample

    def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames):

        video_clip = []
        idx = 0

        for i in range(clip_len):
            # cur_img_path = os.path.join(
            #     video_dir,
            #     # str(ind_frame).zfill(5) + '.png'
            #     "image_{:05}.png".format(start_frame + idx * sample_rate))
            cur_img_path = os.path.join(
                video_dir,
                # str(ind_frame).zfill(5) + '.png'
                "{:04}.jpg".format(start_frame + idx * sample_rate))
            #print('cur_img_path',cur_img_path)
            #exit()
            img = cv2.cvtColor(cv2.imread(cur_img_path), cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (112, 112))
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1

        video_clip = np.array(video_clip)

        return video_clip

def build_dataloader():
    train_transforms = Compose([
        # RandomRotation(15),
        Resize((256, 256)),
        RandomCrop((224, 224)),
        ColorJitter(0.5, 0.5, 0.5, 0.25),
        RandomHorizontalFlip(),
        ClipToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transforms = Compose([
        # RandomRotation(15),
        Resize((224, 224)),
        ClipToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    user_root = '/home/syedhammadahmed/Datasets'    #todo: directory where /mob directory resides
    mob_frames_path = '/mob/MOB_RUN/frames/video'
    mob_frames_abs_path = user_root + mob_frames_path

    train_path = './mob_dataloader/train_list.txt'
    train_dataset = MOBDataset(mob_frames_abs_path,
        8, train=True, transforms_=train_transforms, train_file=train_path)

    test_path = './mob_dataloader/test_list.txt'
    test_dataset = MOBDataset(mob_frames_abs_path,
        8, train=False, transforms_=test_transforms, test_file=test_path)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8, num_workers=4,
                                  pin_memory=True, drop_last=True, shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1, num_workers=4,
                                 pin_memory=True, drop_last=True, shuffle=False)

    return train_dataloader, test_dataloader, test_dataset


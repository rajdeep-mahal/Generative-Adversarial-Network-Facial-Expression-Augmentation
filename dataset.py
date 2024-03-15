import os
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image, ImageSequence

from torchvision.transforms import v2


def get_transform():
    transforms = v2.Compose([
                             v2.ToImage(),
                             v2.ToDtype(torch.uint8, scale=True),
                             v2.Resize([256, 256], antialias=True),
                             v2.RandomHorizontalFlip(0.5),
                             v2.ColorJitter(hue=0.5),
                             v2.ToDtype(torch.float32, scale=True),])
                             #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms


def read_video(path:str):
    """
    Reads the video from the given path.
    The path could contain any of the following:
        - .mp4 file or any video file format
        - folder of images
        - .jpg or any image format which contains concatenated frames
    """

    if path.endswith('.gif'):       
        image = Image.open(path)
        frames = []
        for frame in ImageSequence.Iterator(image):
            frame = frame.convert('RGB')  
            frames.append(np.asarray(frame))
        video_arr =  np.stack(frames, axis=0)

    return video_arr


class VideoDataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        super().__init__()
        self.data_path = data_path
        self.videos = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx:int):
        video_name = self.videos[idx]
        video_arr = read_video(f"{self.data_path}/{video_name}")

        num_frames = video_arr.shape[0]
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        video_arr = video_arr[frame_idx]

        source, driving = video_arr[0], video_arr[1]

        if self.transform is not None:
            source = self.transform(source)
            driving = self.transform(driving)

        output = {"source": source,
                  "driving": driving}
        
        return output

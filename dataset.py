import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import v2


def get_transform():
    transforms = v2.Compose([v2.Resize([256, 256], antialias=True),
                             v2.RandomHorizontalFlip(0.5),
                             v2.ColorJitter(brightness=0.1, saturation=0.1, contrast=0.1, hue=0.1),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms


class VoxCelebaDataset(Dataset):
    def __init__(self, root_dir:str, transform=None):
        super().__init__()
        self.videos = glob.glob(root_dir+'/'+'*.mp4')
        self.transform = transform

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx:int):
        video_path = self.videos[idx]
        vframes = read_video(video_path, pts_unit="sec", output_format="TCHW")[0]

        num_frames = vframes.shape[0]
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        vframes = vframes[frame_idx]

        source, driving = vframes[0], vframes[1]

        if self.transform is not None:
            source = self.transform(source)
            driving = self.transform(driving)

        output = {"source": source,
                  "driving": driving}
        
        return output

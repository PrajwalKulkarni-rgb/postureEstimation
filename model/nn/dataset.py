import torch
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, x_path, y_path, seq_length=50, augment=False):
        self.augment = augment
        self.seq_length = seq_length
        self.X, self.y = self._process_data(x_path, y_path)

    def _process_data(self, x_path, y_path):
        try:
            X = np.load(x_path)
            y = np.load(y_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data files not found at {x_path} or {y_path}")

        # X is already chunked to (N, 50, 102), reshape to (N, 50, 17, 6)
        X_seq = X.reshape(-1, self.seq_length, 17, 6)
        return X_seq, torch.tensor(y, dtype=torch.long)

    def _augment_physics(self, clip):
        """Apply random rotation and jitter."""
        # Rotation around gravity (Y-axis)
        theta = np.random.uniform(-0.3, 0.3) 
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        # Separate Pos and Vel
        pos = clip[:, :, 0:3].reshape(-1, 3)
        vel = clip[:, :, 3:6].reshape(-1, 3)
        
        # Rotate both
        pos = np.dot(pos, rot_mat.T).reshape(self.seq_length, 17, 3)
        vel = np.dot(vel, rot_mat.T).reshape(self.seq_length, 17, 3)
        
        # Add Jitter
        pos += np.random.normal(0, 0.002, pos.shape)

        # Write rotated + jittered data back to clip
        clip[:, :, 0:3] = pos
        clip[:, :, 3:6] = vel

        if np.random.rand() < 0.2:
            # Freeze legs to their first frame position, zero velocity
            clip[:, 11:17, 0:3] = clip[0, 11:17, 0:3] 
            clip[:, 11:17, 3:6] = 0 
            
        # 10% chance to freeze one arm (Simulate side view blockage)
        if np.random.rand() < 0.1:
            if np.random.rand() < 0.5:
                clip[:, 5:11:2, 0:3] = clip[0, 5:11:2, 0:3] # Freeze Left Arm pos
                clip[:, 5:11:2, 3:6] = 0 # Zero Left Arm vel
            else:
                clip[:, 6:12:2, 0:3] = clip[0, 6:12:2, 0:3] # Freeze Right Arm pos
                clip[:, 6:12:2, 3:6] = 0 # Zero Right Arm vel
                
        return clip

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        clip = self.X[idx].copy()
        if self.augment:
            clip = self._augment_physics(clip)
        
        # To Tensor: (Time, Vertices, Channels) -> (Channels, Time, Vertices)
        clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(2, 0, 1)
        return clip_tensor, self.y[idx]
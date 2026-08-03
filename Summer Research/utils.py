import pandas as pd
from astropy.io import fits
from pathlib import Path
from PIL import Image
import tqdm

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock



def fits_df(folder_dir):
    """
    Reads the headers of all fits files in folder_dir, and puts
    them into a DataFrame
    Parameters:
        folder_dir: directory with fits images
    Returns:
        pandas DataFrame with fits image file paths and header values
    """
    imgs = list(folder_dir.glob('*.fits'))
    headers = []

    for img_path in imgs:
        header = dict(fits.getheader(img_path))
        header["FILEPATH"] = img_path
        header["FILENAME"] = img_path.name
        headers.append(header)
        
    return pd.DataFrame(headers)


# make a class to transform, store, and organize data
class GalaxyDataset(Dataset):
    """Dataset of preprocessed galaxy images"""
    def __init__(self, galaxy_paths, input_size=100, transform=None, cache_path=None):
        self.transform = transform

        if cache_path and Path(cache_path).exists():
            # caching logic to avoid preprocessing on consecutive runs
            cache = torch.load(cache_path, weights_only=False)
            self.images = cache['images']
            self.galaxy_ids = cache['galaxy_ids']
            self.labels = cache['labels']
            
        
        else:
            # preproccessing transforms
            pre = transforms.Compose([
                transforms.Resize(input_size)]) # resize
            
            # store ids, images, and class probabilities
            ids = []
            imgs = []
            lbls = []
            for gal_path in galaxy_paths:
                headers = fits_df(gal_path).sort_values(by="BAND")
                ids.append(headers.at[0, 'RA'])
                lbls.append(headers.at[0, 'FLAG'])

                gal_imgs = []
                for _, row in headers.iterrows():
                    hdul = fits.open(gal_path / f"{row['BAND']}.fits")
                    data = hdul[0].data.astype('float32')
                    gal_imgs.append(torch.from_numpy(data))
                    hdul.close()

                imgs.append(pre(torch.stack(gal_imgs, dim=0))) # shape = C, H, W ?

            self.galaxy_ids = torch.tensor(ids)
            self.images = torch.stack(imgs, dim=0)
            self.labels = torch.tensor(lbls)

            # caching logic - saves the transformed data
            if cache_path:
                torch.save({'images': self.images,
                            'galaxy_ids': self.galaxy_ids,
                            'labels': self.labels},
                            cache_path)

    # function to return the image and class probabilities for a galaxy
    # index based, not based on galaxy id
    # the passed transform method is applied here to allow testing different transforms
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]
    
    # returns to number of stored galaxies
    def __len__(self):
        return len(self.images)

    
class ConvBlock(nn.Module):
    """Two CNN layers with batch normalization and ReLU activation. Followed by one MaxPooling layer."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.block(x)


class ConvTokenizer(nn.Module):
    """Projects feature map into a sequence of tokens for the transformer."""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x) # project features into patches
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # reshape to be able to pass into CvT
        return x


class CvTBlock(nn.Module):
    """Uses PyTorch's EncoderBlock for the convolutional vision transformer."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        
        # calculates the multilayer perceptron dimension using mlp_dim
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # combines LayerNorm, MultiheadAttention, and the MLP
        self.transformer_layer = EncoderBlock(
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=dropout,
            norm_layer=nn.LayerNorm)

    def forward(self, x):
        return self.transformer_layer(x)


class FCBlock(nn.Module):
    def __init__(self, embed_dim, fc_dim, dropout, num_outputs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fc_dim, fc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fc_dim // 2, num_outputs))

    def forward(self, x):
        return self.block(x)
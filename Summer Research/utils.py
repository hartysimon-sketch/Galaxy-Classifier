import pandas as pd
from astropy.io import fits
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm
from torchvision.transforms import InterpolationMode

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
                transforms.Resize(input_size, InterpolationMode.BILINEAR)]) # resize
            
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
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fc_dim, fc_dim // 2),
            nn.BatchNorm1d(fc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fc_dim // 2, num_outputs))

    def forward(self, x):
        return self.block(x)


def mask_other_sources(data, box_size=15, fwhm=3.0, nsigma=5, npixels=10, seed=None):
    from astropy.convolution import convolve
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import make_2dgaussian_kernel, SourceFinder
    """Detect sources, keep only the segment at the image center, and
    replace all other detected sources with background noise.

    Parameters
    ----------
    data : 2D ndarray
        The cutout image.
    box_size : int
        Background2D mesh size (pixels). Should be smaller than the cutout.
    fwhm : float
        FWHM (pixels) of the Gaussian smoothing kernel used for detection.
    nsigma : float
        Detection threshold in units of background RMS.
    npixels : int
        Minimum number of connected pixels for a detection.

    Returns
    -------
    cleaned : 2D ndarray
        Image with all non-central sources replaced by background noise.
    segment_map : SegmentationImage or None
        Final segmentation map (None if no sources were detected).
    central_label : int or None
        Label of the segment identified as the central/target galaxy.
    """

    # 1. Estimate background and background RMS
    box_size = min(box_size, min(data.shape) // 3)
    bkg = Background2D(data, box_size, filter_size=(3, 3),
                        sigma_clip=SigmaClip(sigma=3.0),
                        bkg_estimator=MedianBackground())
    data_sub = data - bkg.background

    # 2. Convolve for detection
    kernel = make_2dgaussian_kernel(fwhm, size=5)
    convolved = convolve(data_sub, kernel)

    # 3. Detect+deblend sources
    threshold = nsigma * bkg.background_rms
    finder = SourceFinder(n_pixels=npixels, progress_bar=False)
    segment_map = finder(convolved, threshold)

    cleaned = data.copy()
    central_label = None

    if segment_map is not None:
        # 4. Identify the segment covering the center target
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        central_label = segment_map.data[cy, cx]
        # if the exact center is background (0), pick the segment closest to the center
        if central_label == 0 and segment_map.nlabels > 0:
            from photutils.segmentation import SourceCatalog
            cat = SourceCatalog(data_sub, segment_map, convolved_data=convolved)
            dist = np.hypot(cat.xcentroid - cx, cat.ycentroid - cy)
            central_label = cat.labels[np.argmin(dist)]
        # 5. Build noise
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=bkg.background, scale=bkg.background_rms)
        # 6. Replace every pixel that is not the central target
        other_mask = (segment_map.data != 0) & (segment_map.data != central_label)
        cleaned[other_mask] = noise[other_mask]

    return cleaned, segment_map, central_label


class RunLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump([], f)

    def log_run(self, logged_params):
        # Load existing runs
        with open(self.filepath, "r") as f:
            data = json.load(f)
        
        # Add to log
        data.append(logged_params)
        
        # Save back to disk
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)
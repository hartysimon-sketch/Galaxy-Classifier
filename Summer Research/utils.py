import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits


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
        header['FILEPATH'] = img_path
        header['FILENAME'] = img_path.name
        headers.append(header)

    return pd.DataFrame(headers)


# make a class to transform, store, and organize data
class GalaxyDataset(torch.utils.data.Dataset):
    """Dataset of preprocessed galaxy images"""

    def __init__(self, images, ids, labels, transform=None, cache_path=None):
        self.transform = transform

        if cache_path and Path(cache_path).exists():
            # caching logic to avoid preprocessing on consecutive runs
            cache = torch.load(cache_path, weights_only=False)
            self.images = cache['images']
            self.galaxy_ids = cache['galaxy_ids']
            self.labels = cache['labels']

        else:
            self.galaxy_ids = ids
            self.images = images
            self.labels = labels

            # caching logic - saves the data
            if cache_path:
                torch.save(
                    {
                        'images': self.images,
                        'galaxy_ids': self.galaxy_ids,
                        'labels': self.labels,
                    },
                    cache_path,
                )

    # function to return the image and class probabilities for a galaxy.
    # index based, not based on galaxy id
    # the passed transform method is applied here
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

    # returns to number of stored galaxies
    def __len__(self):
        return len(self.images)


def mask_other_sources(
    data, box_size=15, fwhm=3.0, nsigma=5, npixels=10, seed=None
):
    from astropy.convolution import convolve
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import SourceFinder, make_2dgaussian_kernel

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
    bkg = Background2D(
        data,
        box_size,
        filter_size=(3, 3),
        sigma_clip=SigmaClip(sigma=3.0),
        bkg_estimator=MedianBackground(),
    )
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
        other_mask = (segment_map.data != 0) & (
            segment_map.data != central_label
        )
        cleaned[other_mask] = noise[other_mask]

    return cleaned, segment_map, central_label


class RunLogger:
    def __init__(self, filepath: Path):
        """_summary_

        Args:
            filepath (Path): _description_
        """
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def log_run(self, logged_params):
        # load existing log
        with open(self.filepath) as f:
            data = json.load(f)

        # update and save
        data.append(logged_params)  # add to log
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def set_seeds(SEED: int) -> None:
    """set random seeds for pytorch training

    Args:
        SEED (int): the random seed to be set
    """
    random.seed(SEED)
    np.random.seed(SEED)

    # pytorch seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # deterministic pytorch cuda operations -- slows training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def do_epoch(model, loader, loss_function, device, train=True, optimizer=None):
    running_loss = 0
    correct = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)

        # zero gradients in training
        if train:
            optimizer.zero_grad()

        # get outputs and loss
        outputs = model(data)
        labels = labels.long()  # FOR CLASSIFICATION
        loss = loss_function(outputs, labels)

        # compute gradients and update weights in training
        # otherwise, compute loss and num correct
        if train:
            loss.backward()
            optimizer.step()
        else:
            running_loss += loss.item() * data.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).float().sum()

    if not train:
        return running_loss, correct

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from astropy.io import fits
from PIL import Image


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

    def __init__(
        self,
        images,
        labels,
        transform=None,
        save_path=None,
        load_path=None,
        multiplier=None,
        transform_on_get=False,
    ):

        self.transform = transform
        self.multiplier = multiplier
        self.transform_on_get = transform_on_get

        # will load dataset saved at given path
        if load_path and Path(load_path).exists():
            cache = torch.load(load_path, weights_only=False)
            self.images = cache['images']
            self.labels = cache['labels']
        elif load_path is None:
            all_images = [images]
            all_labels = [labels]
            if multiplier:
                for _ in range(multiplier - 1):
                    batch_images = (
                        torch.stack([transform(img) for img in images])
                        if not transform_on_get
                        else [images]
                    )
                    all_images.append(batch_images)
                    all_labels.append(labels)
            self.images = torch.cat(all_images, dim=0)
            self.labels = torch.cat(all_labels, dim=0)
        else:
            raise FileNotFoundError(f'no file named: {load_path} found')

        # saves dataset to given path
        if save_path:
            torch.save(
                {
                    'images': self.images,
                    'labels': self.labels,
                },
                save_path,
            )

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform and self.transform_on_get:
            img = self.transform(img)

        return img, self.labels[idx]

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
    if train:
        model.train()
    else:
        model.eval()

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

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).float().sum()
        running_loss += loss.item() * data.size(0)

    return running_loss, correct


def imshow_tensor(tensor, scale: int = 4):
    img = F.to_pil_image(tensor.detach().cpu())
    if scale != 1:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, resample=Image.Resampling.NEAREST)
    return img


def masked_ratio(img1, img2):
    from astropy.stats import sigma_clipped_stats

    # Divide two images, keeping only pixels where both bands exceed a flux threshold.
    _, _, thresh1 = sigma_clipped_stats(img1)
    _, _, thresh2 = sigma_clipped_stats(img2)
    good = (img1 > thresh1) & (img2 > thresh2)
    ratio = np.full_like(img1, 0)
    np.divide(img1, img2, out=ratio, where=good)
    return ratio


def clip_extremes(image, sigma=5):
    from astropy.stats import sigma_clip

    # Sigma-clip residual outliers
    clipped = sigma_clip(image, sigma=sigma, maxiters=5, masked=True)
    result = image.copy()
    result[clipped.mask] = 0
    return result


def reset_weights(m):
    # source: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def analyze_data(data, axis=None, name=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if name is not None:
        print(name)

    print('Mean: ', np.mean(data, axis=axis))
    print('Median: ', np.median(data, axis=axis))
    print('St. Dev: ', np.std(data, axis=axis))

    print('Max: ', np.max(data, axis=axis))
    print('Min: ', np.min(data, axis=axis))

    print('Length: ', len(data))


def get_hyperparameters(trial, model, device):

    # optimizer and learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical(
        'optimizer', ['Adam', 'SGD', 'AdamW']
    )

    kwargs = {'lr': lr}

    if optimizer_name == 'AdamW':
        kwargs['weight_decay'] = trial.suggest_float(
            'adamw_weight_decay', 1e-5, 1e-1, log=True
        )
    elif optimizer_name == 'SGD':
        kwargs['momentum'] = trial.suggest_float('sgd_momentum', 0.0, 0.99)

    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), **kwargs
    )

    # weights and loss function
    w1 = trial.suggest_float('w1', 1, 3)
    w2 = trial.suggest_float('w2', 1, 3)
    weights = torch.tensor([w1, w2]).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)

    # batch size and loaders
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    return optimizer, loss_func, batch_size

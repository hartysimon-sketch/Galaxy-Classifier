import pandas as pd
import ccdproc as ccdp
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
        header["FILEPATH"] = img_path
        header["FILENAME"] = img_path.name
        headers.append(header)
        
    return pd.DataFrame(headers)
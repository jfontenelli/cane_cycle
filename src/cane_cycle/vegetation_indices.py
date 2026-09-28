"""Vegetation indices computed from Sentinel-2 band reflectances.

To add a new index: write a function that takes the DataFrame and returns a
Series, then register it in INDEX_FUNCTIONS and INDEX_GROUPS.
"""

import pandas as pd

# Bands used to compute the indices (checked for reflectance scale).
SPECTRAL_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
]

# If the median band value is above this, bands are assumed to be in 0-10000.
SCALE_THRESHOLD = 1.5

# Divisor used to convert 0-10000 digital numbers to 0-1 reflectance.
REFLECTANCE_SCALE = 10000


def ensure_reflectance_scale(df):
    """Return a copy of df with bands in 0-1 reflectance.

    The scale is detected from the median of all band values (the median is
    robust to bright cloudy pixels that can exceed 1 even in 0-1 data).
    """
    df = df.copy()
    bands = [b for b in SPECTRAL_BANDS if b in df.columns]
    median_value = df[bands].stack().median()
    if median_value > SCALE_THRESHOLD:
        df[bands] = df[bands] / REFLECTANCE_SCALE
    return df


def add_peak_band(df, bands=None):
    """Return a copy of df with the highest band of each observation.

    The band that holds the spectral peak describes the *shape* of the
    spectrum, not its brightness, so it is robust to how bright or dark the
    target is. In the field spectral library a live sugarcane canopy peaks in
    the narrow NIR (B8A), straw and bare soil peak in SWIR1 (B11), and thick
    cloud peaks in the blue / coastal aerosol bands (B1, B2).

    Parameters
    ----------
    df : pandas.DataFrame
        Table with Sentinel-2 band columns.
    bands : list of str, optional
        Bands to search for the peak. If None, all bands of SPECTRAL_BANDS
        present in df are used.

    Returns
    -------
    pandas.DataFrame
        Copy of df with two new columns: `peak_band` (name of the highest
        band) and `peak_value` (its reflectance).
    """
    df = df.copy()
    if bands is None:
        bands = [b for b in SPECTRAL_BANDS if b in df.columns]

    df["peak_band"] = df[bands].idxmax(axis=1)
    df["peak_value"] = df[bands].max(axis=1)
    return df


def normalized_difference(a, b):
    """Generic normalized difference: (a - b) / (a + b)."""
    return (a - b) / (a + b)


def ndvi(df):
    """Normalized Difference Vegetation Index: green biomass / chlorophyll."""
    return normalized_difference(df["B8"], df["B4"])


def evi(df):
    """Enhanced Vegetation Index: biomass, less saturation than NDVI.

    Requires reflectance in 0-1 (the constant 1 in the denominator assumes it).
    """
    return 2.5 * (df["B8"] - df["B4"]) / (df["B8"] + 6 * df["B4"] - 7.5 * df["B2"] + 1)


def ndre(df):
    """Normalized Difference Red Edge: chlorophyll / nitrogen status.

    Uses B8A (narrow NIR); falls back to B8 if B8A is not available.
    """
    nir = df["B8A"] if "B8A" in df.columns else df["B8"]
    return normalized_difference(nir, df["B5"])


def ndmi(df):
    """Normalized Difference Moisture Index: canopy water content."""
    return normalized_difference(df["B8"], df["B11"])


def ndti(df):
    """Normalized Difference Tillage Index: crop residue (straw) vs bare soil."""
    return normalized_difference(df["B11"], df["B12"])


# Catalog of available indices: name -> function.
INDEX_FUNCTIONS = {
    "NDVI": ndvi,
    "EVI": evi,
    "NDRE": ndre,
    "NDMI": ndmi,
    "NDTI": ndti,
}

# Group of each index: indices in the same group share a y-axis in plots.
INDEX_GROUPS = {
    "NDVI": "vegetation",
    "EVI": "vegetation",
    "NDRE": "vegetation",
    "NDMI": "moisture",
    "NDTI": "tillage",
}


def add_indices(df, indices=None):
    """Return a copy of df with one new column per vegetation index.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with Sentinel-2 band columns (B2, B4, B5, B8, B8A, B11, B12...).
    indices : list of str, optional
        Index names to compute (e.g. ["NDVI", "NDTI"]). If None, all indices
        in INDEX_FUNCTIONS are computed.

    Returns
    -------
    pandas.DataFrame
        Copy of df (bands converted to 0-1 if needed) plus the index columns.
    """
    if indices is None:
        indices = list(INDEX_FUNCTIONS)

    unknown = [name for name in indices if name not in INDEX_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown indices: {unknown}. Available: {list(INDEX_FUNCTIONS)}"
        )

    df = ensure_reflectance_scale(df)
    for name in indices:
        df[name] = INDEX_FUNCTIONS[name](df)
    return df

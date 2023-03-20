import lacosmic
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats


def linear(x, m, c):
    return m * x + c


def clean_cosmic_rays(spectrum2D,
                      mask=None,
                      contrast=3,
                      cr_threshold=5,
                      neighbor_threshold=0.3,
                      effective_gain=1,
                      readnoise=1
                      ):
    if mask is None:
        mask = (spectrum2D == 0)

    cleaned_spectrum2D, _ = lacosmic.lacosmic(spectrum2D,
                                              contrast=contrast,
                                              cr_threshold=cr_threshold,
                                              neighbor_threshold=neighbor_threshold,
                                              mask=mask,
                                              effective_gain=effective_gain,
                                              readnoise=readnoise
                                              )
    return cleaned_spectrum2D


def plot_spec2d(spec2D, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    mean, median, std = sigma_clipped_stats(spec2D)
    ax.imshow(spec2D, origin='lower', vmin=median - 3 * std,
               vmax=median + 3 * std, aspect=0.4 * spec2D.shape[1] / spec2D.shape[0])
    return ax

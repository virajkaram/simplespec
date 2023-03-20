import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from simplespec.utils import linear, plot_spec2d


class SkySubtractor:
    """
    Module to subtract sky lines from spectrum.
    First, the user needs to specify a relatively broad but not too broad spatial range,
    preferably around the trace, around a bright sky line. The code fits a linear
    function to trace the shape of the sky line in the spectrum (which is why the
    spatial range cannot be too broad if the lines are curved).
    Then, the code uses a smaller region around trace (specified by neartrace) to
    estimate the flux at the trace using linear interpolation.
    Attributes:
        skyline_spectral_min: min pixel number around a bright skyline
        skyline_spectral_max: max pixel number around the bright skyline
        neartrace_spatial_min: narrower region around trace, used for sky estimates
        neartrace_spatial_max: narrower region around trace, used for sky estimates
        trace_spatial_min: min pixel number around the trace
        trace_spatial_max: max pixel number around trace
    """
    def __init__(self,
                 spectrum2D,
                 skyline_spectral_min: int,
                 skyline_spectral_max: int,
                 trace_spatial_min: int,
                 trace_spatial_max: int,
                 neartrace_spatial_min: int,
                 neartrace_spatial_max: int,
                 skyline_spatial_min: int = None,
                 skyline_spatial_max: int = None,
                 ):
        self.spectrum2D = spectrum2D

        self.skyline_spectral_min = skyline_spectral_min
        self.skyline_spectral_max = skyline_spectral_max
        self.skyline_spatial_min = skyline_spatial_min
        self.skyline_spatial_max = skyline_spatial_max
        self.neartrace_spatial_min = neartrace_spatial_min
        self.neartrace_spatial_max = neartrace_spatial_max

        self.trace_spatial_min = trace_spatial_min
        self.trace_spatial_max = trace_spatial_max

        if self.skyline_spatial_min is None:
            self.skyline_spatial_min = 0

        if self.skyline_spatial_max is None:
            self.skyline_spatial_max = spectrum2D.shape[0]

        self.skyline_cutout = self.spectrum2D[self.skyline_spatial_min:
                                              self.skyline_spatial_max,
                                              self.skyline_spectral_min:
                                              self.skyline_spectral_max]

        self.cutout = self.spectrum2D[self.skyline_spatial_min:
                                      self.skyline_spatial_max,
                                      :]
        self.sky_slope = None
        self.sky_intercept = None

    def plot_2dspec(self,
                    spec2D: np.ndarray = None,
                    ax=None):
        if spec2D is None:
            spec2D = self.spectrum2D

        if ax is None:
            ax = plt.figure().gca()

        ax = plot_spec2d(spec2D, ax=ax)
        return ax

    def plot_trace(self, x: float = None, ax=None):
        if ax is None:
            ax = plt.figure().gca()
        self.plot_2dspec(self.skyline_cutout, ax=ax)
        plotys = range(self.skyline_cutout.shape[0])
        if x is None:
            x = int(self.skyline_cutout.shape[1] / 2)
        ax.plot(np.array(linear(plotys, self.sky_slope, x), dtype=int),
                plotys, color='blue')
        return ax

    def fit_trace(self):
        maxids = np.argmax(self.skyline_cutout, axis=1)
        popt, pcov = curve_fit(xdata=range(len(maxids)), ydata=maxids, f=linear)
        self.sky_slope, self.sky_intercept = popt
        return popt

    def get_interpolated_sky_values(self):
        neartracecut = self.cutout
        sky_interp_vals = []
        ys = np.array(range(self.cutout.shape[0]))
        for ind in range(neartracecut.shape[1]):
            lineinds = np.array(linear(ys, self.sky_slope, ind), dtype=int)

            tracemask = (ys > self.trace_spatial_min) & (ys < self.trace_spatial_max)
            neartracemask = (ys > self.neartrace_spatial_min) & \
                            (ys < self.neartrace_spatial_max)
            nontracemask = np.invert(tracemask)

            nonneartmask = neartracemask & nontracemask
            interp_vals = np.interp(x=ys[tracemask], xp=ys[nonneartmask],
                                    fp=neartracecut[ys, lineinds][nonneartmask])

            sky_interp_vals.append(interp_vals)

        sky_interp_vals = np.array(sky_interp_vals).T
        return sky_interp_vals

    def linearize_slant_spatial_profiles(self):
        neartracecut = self.cutout
        ys = np.array(range(self.cutout.shape[0]))
        trace_nlin_values = []
        for ind in range(neartracecut.shape[1]):
            lineinds = np.array(linear(ys, self.sky_slope, ind), dtype=int)
            tracevals = neartracecut[ys, lineinds]
            trace_nlin_values.append(tracevals)

        trace_nlin_values = np.array(trace_nlin_values).T
        return trace_nlin_values

    def subtract_sky(self):
        ys = np.array(range(self.cutout.shape[0]))
        tracemask = (ys > self.trace_spatial_min) & (ys < self.trace_spatial_max)
        cutout_lin_values = self.linearize_slant_spatial_profiles()
        trace_vals = cutout_lin_values[tracemask]

        sky_interp_vals = self.get_interpolated_sky_values()
        sky_sub_trace = trace_vals - sky_interp_vals

        return sky_sub_trace

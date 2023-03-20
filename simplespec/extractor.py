import numpy as np


class Extractor:
    def __init__(self,
                 sky_sub_spec2D: np.ndarray,
                 trace_extract_spatial_min: int,
                 trace_extract_spatial_max: int
                 ):
        self.sky_sub_spec2D = sky_sub_spec2D
        self.trace_extract_spatial_min = trace_extract_spatial_min
        self.trace_extract_spatial_max = trace_extract_spatial_max

    def extract_trace(self):
        trace_extract = self.sky_sub_spec2D[self.trace_extract_spatial_min:
                                            self.trace_extract_spatial_max, :]
        extract_vals = np.sum(trace_extract, axis=0)
        pixels = range(len(extract_vals))
        return pixels, extract_vals

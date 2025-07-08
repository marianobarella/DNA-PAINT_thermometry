"""
Created on Tuesday Novemeber 17 2021

@author: Mariano Barella

This script contains the auxiliary functions that process_picasso_data
main script uses.

"""

# ================ IMPORT LIBRARIES ================
import os
import numpy as np
import scipy.signal as sig
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from itertools import groupby
import scipy.stats as sta
import matplotlib.pyplot as plt
import pickle


# ================ GLOBAL CONSTANTS ================
# time resolution at 100 ms
R = 0.07 # resolution width, in s
R = 0.00 # resolution width, in s


# ================ IMAGE PROCESSING FUNCTIONS ================
# 2D peak detection algorithm
# taken from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value 
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    # local_max is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image==0)

    # a little technicality: we must erode the background in order to 
    # successfully subtract it form local_max, otherwise a line will 
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def detect_peaks_improved(x_positions, y_positions, hist_bounds, expected_peaks=3, min_distance_nm=10):
    """
    Improved peak detection with better resolution and minimum distance constraint.
    
    Args:
        x_positions, y_positions: coordinate arrays
        hist_bounds: histogram boundaries
        expected_peaks: expected number of peaks
        min_distance_nm: minimum distance between peaks in nm
    
    Returns:
        peak_coords: list of (x, y) peak coordinates
    """
    from scipy.ndimage import gaussian_filter
    
    # Use finer binning for better resolution
    bins_fine = 50  # Much finer than the original 20 bins
    
    # Create fine 2D histogram
    z_hist, x_edges, y_edges = np.histogram2d(
        x_positions, y_positions, 
        bins=bins_fine, range=hist_bounds, density=True
    )
    z_hist = z_hist.T
    
    # Apply light gaussian smoothing to reduce noise
    z_hist_smooth = gaussian_filter(z_hist, sigma=0.8)
    
    # Find local maxima
    detected_peaks = detect_peaks(z_hist_smooth)
    peak_indices = np.where(detected_peaks)
    
    if len(peak_indices[0]) == 0:
        return []
    
    # Convert indices to coordinates
    x_centers = x_edges[:-1] + np.diff(x_edges)/2
    y_centers = y_edges[:-1] + np.diff(y_edges)/2
    
    peak_coords = [(x_centers[peak_indices[1][i]], y_centers[peak_indices[0][i]]) 
                   for i in range(len(peak_indices[0]))]
    
    # Get peak intensities for ranking
    peak_intensities = [z_hist_smooth[peak_indices[0][i], peak_indices[1][i]] 
                       for i in range(len(peak_indices[0]))]
    
    # Sort peaks by intensity (strongest first)
    sorted_indices = np.argsort(peak_intensities)[::-1]
    peak_coords = [peak_coords[i] for i in sorted_indices]
    peak_intensities = [peak_intensities[i] for i in sorted_indices]
    
    # Enforce minimum distance between peaks
    min_distance_um = min_distance_nm / 1000  # Convert nm to um
    filtered_peaks = []
    
    for i, (x_peak, y_peak) in enumerate(peak_coords):
        # Check distance to already selected peaks
        too_close = False
        for x_sel, y_sel in filtered_peaks:
            distance = np.sqrt((x_peak - x_sel)**2 + (y_peak - y_sel)**2)
            if distance < min_distance_um:
                too_close = True
                break
        
        # Add peak if it's far enough from others
        if not too_close:
            filtered_peaks.append((x_peak, y_peak))
            
        # Stop when we have enough peaks
        if len(filtered_peaks) >= expected_peaks:
            break
    
    return filtered_peaks


def get_peak_detection_histogram(x_positions, y_positions, hist_bounds, bins_fine=50, sigma=0.8):
    """
    Get the histogram data used for peak detection visualization.
    
    Args:
        x_positions, y_positions: coordinate arrays
        hist_bounds: histogram boundaries
        bins_fine: number of bins for histogram (default: 50)
        sigma: gaussian filter sigma (default: 0.8)
    
    Returns:
        z_hist_smooth: smoothed histogram data
        x_centers: x bin centers
        y_centers: y bin centers
    """
    from scipy.ndimage import gaussian_filter
    
    # Create fine 2D histogram
    z_hist, x_edges, y_edges = np.histogram2d(
        x_positions, y_positions, 
        bins=bins_fine, range=hist_bounds, density=True
    )
    z_hist = z_hist.T
    
    # Apply gaussian smoothing
    z_hist_smooth = gaussian_filter(z_hist, sigma=sigma)
    
    # Get bin centers
    x_centers = x_edges[:-1] + np.diff(x_edges)/2
    y_centers = y_edges[:-1] + np.diff(y_edges)/2
    
    return z_hist_smooth, x_centers, y_centers


# ================ GEOMETRIC CALCULATION FUNCTIONS ================
# distance calculation circle
def distance(x, y, xc, yc):
    d = ((x - xc)**2 + (y - yc)**2)**0.5
    return d

# ================ GAUSSIAN FUNCTIONS ================
def gaussian_2D_angle(xy_tuple, amplitude, x0, y0, a, b, c, offset):
    (x, y) = xy_tuple
    g = offset + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
    return g.ravel()

# def N_gaussians_2D(xy_tuple, amplitude, x0, y0, a, b, c, offset):
#     (x, y) = xy_tuple
#     g1 = offset1 + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
#     g2 = offset2 + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
#     g3 = offset3 + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
#     g = g1 + g2 + g3
#     return g.ravel()

def abc_to_sxsytheta(a, b, c):
    # ================ CALCULATE GAUSSIAN PARAMETERS FROM COEFFICIENTS ================
    theta_rad = 0.5*np.arctan(2*b/(a-c))
    theta_deg = 360*theta_rad/(2*np.pi)
    aux_sx = a*(np.cos(theta_rad))**2 + \
             2*b*np.cos(theta_rad)*np.sin(theta_rad) + \
             c*(np.sin(theta_rad))**2
    sx = np.sqrt(0.5/aux_sx)
    aux_sy = a*(np.sin(theta_rad))**2 - \
             2*b*np.cos(theta_rad)*np.sin(theta_rad) + \
             c*(np.cos(theta_rad))**2
    sy = np.sqrt(0.5/aux_sy)
    return theta_deg, sx, sy

# ================ STATISTICAL FUNCTIONS ================
# Calculate coefficient of determination
def calc_r2(observed, fitted):
    avg_y = observed.mean()
    # sum of squares of residuals
    ssres = ((observed - fitted)**2).sum()
    # total sum of squares
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

# linear fit without weights
def fit_linear(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    p, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    x_fitted = np.array(x)
    y_fitted = np.polyval(p, x_fitted)
    Rsquared = calc_r2(y, y_fitted)
    # p[0] is the slope
    # p[1] is the intercept
    slope = p[0]
    intercept = p[1]
    return x_fitted, y_fitted, slope, intercept, Rsquared

def perpendicular_distance(slope, intercept, x_point, y_point):
    # source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    numerator = np.abs(slope*x_point + (-1)*y_point + intercept)
    denominator = distance(slope, (-1), 0, 0)
    d = numerator/denominator
    return d

# ================ FILE AND DIRECTORY UTILITIES ================
def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    # Normalize path to handle mixed separators
    path = os.path.normpath(path)
    new_folder_path = os.path.join(path, new_folder_name)
    new_folder_path = os.path.normpath(new_folder_path)
    
    if not os.path.exists(new_folder_path):
        try:
            os.makedirs(new_folder_path)
        except OSError as e:
            print(f"Error creating directory {new_folder_path}: {e}")
            raise
    return new_folder_path

# ================ DATA BINNING FUNCTIONS ================
def classification(value, totalbins, rango):
    # Bin the data. Classify a value into a bin.
    # totalbins = number of bins to divide rango (range)
    bin_max = totalbins - 1
    numbin = 0
    inf = rango[0]
    sup = rango[1]
    if value > sup:
        print('Value higher than max')
        return bin_max
    if value < inf:
        print('Value lower than min')
        return 0
    step = (sup - inf)/totalbins
    # tiene longitud totalbins + 1
    # pero en total son totalbins "cajitas" (bines)
    binned_range = np.arange(inf, sup + step, step)
    while numbin < bin_max:
        if (value >= binned_range[numbin] and value < binned_range[numbin+1]):
            break
        numbin += 1
        if numbin > bin_max:
            break
    return numbin


# ================ SIGNAL PROCESSING FUNCTIONS ================
def mask(number_of_dips=1):
    # Handle special cases first
    if number_of_dips == -1:
        mask_array = [0, -1, 1, 0]
    elif number_of_dips == -99:
        mask_array = [0, 1, 1]
    elif number_of_dips < 0:
        # Assuming the "otherwise" case is for any negative number not handled above
        mask_array = [0, 0]
    else:
        # Handle the general case for positive number_of_dips or 0
        mask_array = [0, 1] + [0] * (number_of_dips - 1) + [-1, 0] if number_of_dips > 0 else [0, 0]

    # Normalize the mask array
    mask_array = 0.5 * np.array(mask_array)
    return mask_array


# ================ BINARY TRACE ANALYSIS FUNCTIONS ================
def find_consecutive_ones(binary_trace):
    sequence_lengths = []
    count = 0

    for bit in binary_trace:
        if bit == 1:
            count += 1
        else:
            if count > 0:
                sequence_lengths.append(count)
                count = 0

    # Check if there's an ongoing sequence of 1s at the end of the trace
    if count > 0:
        sequence_lengths.append(count)

    return sequence_lengths





# ================ BINDING TIME CALCULATION FUNCTIONS ================
def calculate_tau_on_times(trace, threshold, bkg, exposure_time, mask_level, mask_singles, verbose_flag, index):
    """
    Updated binding event analysis using improved logic from position averaging method.
    
    Improvements over original version:
    - Cleaner binary trace creation and masking logic
    - More robust partial frame handling for accurate timing
    - Simplified and more reliable segment detection
    - Better edge case handling
    - Cleaner code structure for maintainability
    
    Args:
        trace: Photon trace array
        threshold: Photon threshold for ON/OFF detection  
        bkg: Background level
        exposure_time: Frame exposure time
        mask_level: Number of consecutive dips to mask (0 = no masking)
        mask_singles: Whether to mask single blips
        verbose_flag: Print debug info
        index: Pick index for debugging
    
    Returns:
        Tuple: (t_on, t_off, binary_trace, start_time, SNR, SBR, sum_photons, 
                avg_photons, photon_intensity, std_photons, start_time_avg_photons, double_events_counts)
    """
    
    # ================ STEP 1: CREATE BINARY TRACE ================
    # Improved: cleaner boolean logic instead of complex np.where operations
    binary_trace = (trace >= threshold).astype(int)
    photons_trace = np.where(trace >= threshold, trace, 0)
    
    if len(np.where(binary_trace > 0)[0]) == 0:
        return np.array([False] * 12)  # Return False array if no events found
    
    # ================ STEP 2: APPLY IMPROVED MASKING ================
    # Improved: direct gap filling instead of complex convolution operations
    if mask_level > 0:
        if verbose_flag:
            print(f'Applying improved masking for {mask_level} frame gaps...')
        
        # Simple dip masking: fill gaps of mask_level frames or less between ON segments
        binary_masked = binary_trace.copy()
        changes = np.diff(np.concatenate(([0], binary_trace, [0])))
        off_starts = np.where(changes == -1)[0]
        off_ends = np.where(changes == 1)[0]
        
        # Fill short gaps between ON segments
        for start, end in zip(off_starts, off_ends):
            if end - start <= mask_level:
                binary_masked[start:end] = 1
                # Interpolate photon values for masked frames
                if start > 0 and end < len(photons_trace):
                    gap_length = end - start
                    start_val = photons_trace[start - 1]
                    end_val = photons_trace[end] if photons_trace[end] > 0 else start_val
                    for i in range(gap_length):
                        photons_trace[start + i] = start_val + (end_val - start_val) * (i + 1) / (gap_length + 1)
        
        binary_trace = binary_masked
    elif verbose_flag:
        print('No masking applied.')
    
    # ================ STEP 3: APPLY SINGLE BLIP MASKING ================
    if mask_singles:
        if verbose_flag:
            print('Removing single-frame blips...')
        # Remove single-frame blips
        for i in range(1, len(binary_trace) - 1):
            if binary_trace[i] == 1 and binary_trace[i-1] == 0 and binary_trace[i+1] == 0:
                binary_trace[i] = 0
                photons_trace[i] = 0
    
    # ================ STEP 4: FIND ON/OFF SEGMENTS ================
    # Improved: cleaner segment detection using diff operations
    changes = np.diff(np.concatenate(([0], binary_trace, [0])))
    on_starts = np.where(changes == 1)[0]
    on_ends = np.where(changes == -1)[0]
    
    if len(on_starts) == 0:
        return np.array([False] * 12)
    
    # ================ STEP 5: CALCULATE SEGMENT PROPERTIES ================
    t_on = []
    t_off = []
    sum_photons = []
    avg_photons = []
    std_photons = []
    start_times = []
    photon_intensity = []
    double_events_counts = []
    start_times_avg_photons = []
    
    # Process each ON segment with improved partial frame handling
    for i, (start, end) in enumerate(zip(on_starts, on_ends)):
        segment = photons_trace[start:end]
        
        # ================ IMPROVED PARTIAL FRAME HANDLING ================
        # More accurate timing calculations accounting for partial frames
        if len(segment) > 2:
            # Use middle frames for statistics, but include partial contributions for timing
            middle_segment = segment[1:-1]
            first_partial = min(segment[0] / np.mean(middle_segment) if len(middle_segment) > 0 else 1, 1)
            last_partial = min(segment[-1] / np.mean(middle_segment) if len(middle_segment) > 0 else 1, 1)
            
            on_time = len(middle_segment) + first_partial + last_partial
            avg_photon = np.mean(middle_segment) if len(middle_segment) > 0 else np.mean(segment)
            std_photon = np.std(middle_segment, ddof=1) if len(middle_segment) > 1 else 0
            
            # Collect photon intensity from middle frames
            photon_intensity.extend(middle_segment)
        else:
            on_time = len(segment)
            avg_photon = np.mean(segment)
            std_photon = np.std(segment, ddof=1) if len(segment) > 1 else 0
            photon_intensity.extend(segment)
        
        # Store results
        t_on.append(on_time)
        sum_photons.append(np.sum(segment))
        avg_photons.append(avg_photon)
        std_photons.append(std_photon)
        start_times.append(start)
        
        # Store start times for segments with sufficient data for averaging
        if len(segment) > 4:
            start_times_avg_photons.append(start)
        
        # ================ SIMPLIFIED DOUBLE EVENT DETECTION ================
        # Improved: simple duration threshold instead of complex rolling window
        double_events_counts.append(1 if len(segment) > 7 else 0)
        
        # Calculate OFF time to next segment (if not last segment)
        if i < len(on_starts) - 1:
            off_time = on_starts[i + 1] - end
            t_off.append(off_time)
    
    # ================ STEP 6: CALCULATE DERIVED METRICS ================
    # Convert to numpy arrays
    t_on = np.array(t_on)
    t_off = np.array(t_off)
    sum_photons = np.array(sum_photons)
    avg_photons = np.array(avg_photons)
    std_photons = np.array(std_photons)
    start_times = np.array(start_times)
    start_times_avg_photons = np.array(start_times_avg_photons)
    photon_intensity = np.array(photon_intensity)
    double_events_counts = np.array(double_events_counts)
    
    # Calculate signal metrics with improved error handling
    SNR = np.divide(avg_photons, std_photons, out=np.zeros_like(avg_photons), where=std_photons!=0)
    SBR = avg_photons / bkg if bkg > 0 else np.zeros_like(avg_photons)
    
    if verbose_flag:
        print(f'Found {len(t_on)} binding events with improved detection')
        if len(t_on) > 0:
            print(f'Average ON time: {np.mean(t_on) * exposure_time:.2f} s')
            print(f'Average photons per event: {np.mean(avg_photons):.1f}')
        print('---------------------------')
    
    # Debug visualization (disabled by default)
    if False and index in range(9):
        plt.scatter(start_times*exposure_time, t_on*exposure_time, s=0.8)
        plt.show()
    
    # ================ STEP 7: RETURN RESULTS ================
    # Return results in same format as original function to maintain compatibility
    return (t_on*exposure_time, t_off*exposure_time, binary_trace, start_times*exposure_time, 
            SNR, SBR, sum_photons, avg_photons, photon_intensity, std_photons, 
            start_times_avg_photons*exposure_time, double_events_counts)


def detect_double_events_rolling(events, window_size=2, threshold=1.5):
    # Calculate rolling averages using a convolution approach
    window_means = np.convolve(events, np.ones(window_size) / window_size, mode='valid')

    # Find indices where event counts exceed the rolling mean threshold
    # We can simplify by directly comparing the relevant slice of events with window_means
    return np.where(events[window_size-1:window_size-1+len(window_means)] > window_means * threshold)[0] + window_size - 1


# ================ PROBABILITY DENSITY FUNCTIONS ================
# definition of hyperexponential p.d.f.
def hyperexp_func(time, real_binding_time, short_on_time, ratio):
    # Prevent overflow by limiting extreme values
    real_binding_time = np.clip(real_binding_time, 1e-6, 1e6)
    short_on_time = np.clip(short_on_time, 1e-6, 1e6)
    
    beta_binding_time = 1/real_binding_time
    beta_short_time = 1/short_on_time
    A = ratio/(ratio + 1)
    B = 1/(ratio + 1)
    
    # Clip exponential arguments to prevent overflow
    exp_arg_binding = np.clip(-time*beta_binding_time, -700, 700)
    exp_arg_short = np.clip(-time*beta_short_time, -700, 700)
    
    f_binding = beta_binding_time*np.exp(exp_arg_binding)
    f_short = beta_short_time*np.exp(exp_arg_short)
    f = A*f_binding + B*f_short
    
    # Replace any inf/nan values with very small positive numbers
    f = np.where(np.isfinite(f) & (f > 0), f, 1e-30)
    return f

# definition of monoexponential p.d.f.
def monoexp_func(time, real_binding_time, short_on_time, amplitude):
    # short on time is not used
    # neither the amplitude
    beta_binding_time = 1/real_binding_time
    f = beta_binding_time*np.exp(-time*beta_binding_time)
    return f

# ================ ERROR-ADJUSTED PROBABILITY FUNCTIONS ================
# definition of hyperexponential p.d.f. including instrumental error
def hyperexp_func_with_error(time, real_binding_time, short_on_time, ratio):
    # Prevent overflow by limiting extreme values
    real_binding_time = np.clip(real_binding_time, 1e-6, 1e6)
    short_on_time = np.clip(short_on_time, 1e-6, 1e6)
    
    beta_binding_time = 1/real_binding_time
    beta_short_time = 1/short_on_time
    A = ratio/(ratio + 1)
    B = 1/(ratio + 1)
    
    # Clip exponential arguments to prevent overflow
    exp_arg_binding = np.clip(-time*beta_binding_time, -700, 700)
    exp_arg_short = np.clip(-time*beta_short_time, -700, 700)
    
    f_binding = beta_binding_time*np.exp(exp_arg_binding)
    f_short = beta_short_time*np.exp(exp_arg_short)
    
    argument_binding = time/R - beta_binding_time*R
    argument_short = time/R - beta_short_time*R
    std_norm_distro = sta.norm(loc=0, scale=1)
    G_binding = std_norm_distro.cdf(argument_binding)
    G_short = std_norm_distro.cdf(argument_short)
    
    # Clip additional exponential arguments
    exp_arg_binding_new = np.clip(0.5*(beta_binding_time*R)**2, -700, 700)
    exp_arg_short_new = np.clip(0.5*(beta_short_time*R)**2, -700, 700)
    
    f_binding_new = np.exp(exp_arg_binding_new)*G_binding*f_binding
    f_short_new = np.exp(exp_arg_short_new)*G_short*f_short
    f = A*f_binding_new + B*f_short_new
    
    # Replace any inf/nan values with very small positive numbers
    f = np.where(np.isfinite(f) & (f > 0), f, 1e-30)
    return f


# definition of monoexponential p.d.f.
def monoexp_func_with_error(time, real_binding_time, short_on_time, amplitude):
    # short on time is not used
    beta = 1/real_binding_time
    f_mono = amplitude*beta*np.exp(-time*beta)
    argument_binding = time/R - beta*R
    std_norm_distro = sta.norm(loc=0, scale=1)
    G = std_norm_distro.cdf(argument_binding)
    f_mono_new = np.exp(0.5*(beta*R)**2)*G*f_mono
    return f_mono_new

# ================ LOG LIKELIHOOD FUNCTIONS ================
# definition of hyperlikelihood function
def log_likelihood_hyper(theta_param, data):
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = hyperexp_func(data, real_binding_time, short_on_time, ratio)
    log_likelihood = -np.sum(np.log(pdf_data))
    # print(log_likelihood)
    return log_likelihood

# definition of hyperlikelihood function
def log_likelihood_hyper_with_error(theta_param, data):
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = hyperexp_func_with_error(data, real_binding_time, short_on_time, ratio)
    
    # Filter out invalid values before taking log to prevent warnings
    pdf_data = pdf_data[pdf_data > 0]  # Remove zeros and negative values
    if len(pdf_data) == 0:
        return np.inf  # Return infinity if no valid data points
        
    log_pdf = np.log(pdf_data)
    log_pdf = log_pdf[~np.isinf(log_pdf)]
    log_pdf = log_pdf[~np.isnan(log_pdf)]
    log_likelihood = -np.sum(log_pdf)
    # print(log_likelihood)
    return log_likelihood

# definition of hyperlikelihood function
def log_likelihood_mono_with_error(theta_param, data):
    # no error actually
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = monoexp_func(data, real_binding_time, short_on_time, ratio)
    
    # Filter out invalid values before taking log to prevent warnings
    pdf_data = pdf_data[pdf_data > 0]  # Remove zeros and negative values
    if len(pdf_data) == 0:
        return np.inf  # Return infinity if no valid data points
        
    log_pdf = np.log(pdf_data)
    log_pdf = log_pdf[~np.isinf(log_pdf)]
    log_pdf = log_pdf[~np.isnan(log_pdf)]
    log_likelihood = -np.sum(log_pdf)
    # print(log_likelihood)

    return log_likelihood


# ================ ALTERNATIVE LOG LIKELIHOOD FUNCTIONS ================
def log_likelihood_mono_with_error_alt(theta_param, data):
    # Unpack the parameters
    loc = theta_param[0]
    real_binding_time = theta_param[1]
    short_on_time = theta_param[2]
    ratio = theta_param[3]

    # Adjust the data by subtracting the loc parameter
    adjusted_data = data - loc
    adjusted_data = adjusted_data[adjusted_data > 0]

    pdf_data = monoexp_func(adjusted_data, real_binding_time, short_on_time, ratio)
    log_pdf = np.log(pdf_data)
    log_pdf = log_pdf[~np.isinf(log_pdf)]
    log_pdf = log_pdf[~np.isnan(log_pdf)]
    log_likelihood = -np.sum(log_pdf)

    return log_likelihood


def log_likelihood_mono_with_error_one_param(theta_param, data):
    # no error actually
    real_binding_time = theta_param
    short_on_time = 0
    ratio = 0
    pdf_data = monoexp_func(data, real_binding_time, short_on_time, ratio)
    log_pdf = np.log(pdf_data)
    log_pdf = log_pdf[~np.isinf(log_pdf)]
    log_pdf = log_pdf[~np.isnan(log_pdf)]
    log_likelihood = -np.sum(log_pdf)
    # print(log_likelihood)

    return log_likelihood


# ================ PLOTTING FUNCTIONS ================
def plot_vs_time_with_hist(data, time, order = 3, fit_line = False):
    dict = {}
    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))  # Set figure size

    # Create a gridspec for 1 row and 2 columns with a ratio of 4:1 between the main and marginal plots.
    # Adjust subplot parameters for optimal layout.
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1],
                          left=0.15, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05)  # `hspace` is not needed for 1 row

    # Create the main plot area.
    ax = fig.add_subplot(gs[0, 0])

    # ================ DATA AGGREGATION FOR PLOTTING ================
    for x, y in zip(time, data):
        if x in dict:
            dict[x] = np.append(dict[x], y)
        else:
            dict[x] = np.array([y])

    for x in dict.keys():
        dict[x] = np.mean(dict[x], axis=None)

    unique_time_values = np.array(list(dict.keys()))
    summed_data = np.array(list(dict.values()))
    sorted_indices = np.argsort(unique_time_values)
    unique_time_values = unique_time_values[sorted_indices]
    summed_data = summed_data[sorted_indices]

    # ================ DATA FILTERING AND VISUALIZATION ================
    filtered_data = sig.savgol_filter(summed_data, window_length=int(len(summed_data)/20), polyorder=1)
    if fit_line:
        x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(unique_time_values, filtered_data)
    ax.scatter(unique_time_values, summed_data, s=0.8)
    ax.plot(unique_time_values, filtered_data, 'r--', linewidth=3, alpha = 0.8)

    # ================ CREATE MARGINAL HISTOGRAM ================
    bin_edges = np.histogram_bin_edges(data, 'fd')
    # Create the marginal plot on the right of the main plot, sharing the y-axis with the main plot.
    ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_histy.hist(data, bins=bin_edges, orientation='horizontal')

    # ================ FINALIZE PLOT LAYOUT ================
    # Make sure the marginal plot's y-axis ticks don't overlap with the main plot.
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    x_limit = [0, unique_time_values[-1]]
    ax.set_xlim(x_limit)
    # y_limit_photons = [0, round(np.max(data, axis=None)+10**order, -order)]
    # ax.set_ylim(y_limit_photons)
    ax.set_ylim(0, np.mean(data)+2*np.std(data))

    if fit_line:
        return ax, slope, intercept

    return ax


# ================ SERIALIZATION UTILITIES ================
def update_pkl(file_path, key, value):
    # Check if the file exists
    file_path_pkl = os.path.join(file_path, 'parameters.pkl')
    if os.path.exists(file_path_pkl):
        # Load the existing dictionary from the file
        with open(file_path_pkl, 'rb') as file:
            data = pickle.load(file)
    else:
        # If the file does not exist, create an empty dictionary
        data = {}

    # Update the dictionary with the new key and value
    data[key] = value

    # Save the updated dictionary back to the file
    with open(file_path_pkl, 'wb') as file:
        pickle.dump(data, file)



# ================ BINDING TIME CALCULATION FUNCTIONS ================
def calculate_tau_on_times_average(trace, threshold, bkg, exposure_time, mask_level, mask_singles, verbose_flag, index, x_positions=None, y_positions=None, frame_numbers=None):
    """
    Simplified binding event analysis with position averaging.
    
    Args:
        trace: Photon trace array
        threshold: Photon threshold for ON/OFF detection
        bkg: Background level
        exposure_time: Frame exposure time
        mask_level: Number of consecutive dips to mask (0 = no masking)
        mask_singles: Whether to mask single blips
        verbose_flag: Print debug info
        index: Pick index for debugging
        x_positions, y_positions, frame_numbers: Optional position data for averaging
    
    Returns:
        Tuple of results including on-times, off-times, positions, etc.
    """
    
    # ================ STEP 1: CREATE BINARY TRACE ================
    binary_trace = (trace >= threshold).astype(int)
    photons_trace = np.where(trace >= threshold, trace, 0)
    
    if len(np.where(binary_trace > 0)[0]) == 0:
        return [False] * 14  # Return False array if no events found
    
    # ================ STEP 2: APPLY SIMPLE MASKING ================
    if mask_level > 0:
        # Simple dip masking: fill gaps of mask_level frames or less between ON segments
        binary_masked = binary_trace.copy()
        changes = np.diff(np.concatenate(([0], binary_trace, [0])))
        off_starts = np.where(changes == -1)[0]
        off_ends = np.where(changes == 1)[0]
        
        # Fill short gaps
        for start, end in zip(off_starts, off_ends):
            if end - start <= mask_level:
                binary_masked[start:end] = 1
                # Interpolate photon values for masked frames
                if start > 0 and end < len(photons_trace):
                    gap_length = end - start
                    start_val = photons_trace[start - 1]
                    end_val = photons_trace[end] if photons_trace[end] > 0 else start_val
                    for i in range(gap_length):
                        photons_trace[start + i] = start_val + (end_val - start_val) * (i + 1) / (gap_length + 1)
        
        binary_trace = binary_masked
    
    if mask_singles:
        # Remove single-frame blips
        for i in range(1, len(binary_trace) - 1):
            if binary_trace[i] == 1 and binary_trace[i-1] == 0 and binary_trace[i+1] == 0:
                binary_trace[i] = 0
                photons_trace[i] = 0
    
    # ================ STEP 3: FIND ON/OFF SEGMENTS ================
    changes = np.diff(np.concatenate(([0], binary_trace, [0])))
    on_starts = np.where(changes == 1)[0]
    on_ends = np.where(changes == -1)[0]
    
    if len(on_starts) == 0:
        return [False] * 14
    
    # ================ STEP 4: CALCULATE SEGMENT PROPERTIES ================
    t_on = []
    t_off = []
    sum_photons = []
    avg_photons = []
    std_photons = []
    start_times = []
    average_x_positions = []
    average_y_positions = []
    
    # Process each ON segment
    for i, (start, end) in enumerate(zip(on_starts, on_ends)):
        segment = photons_trace[start:end]
        segment_frames = np.arange(start, end)
        
        # ================ HANDLE PARTIAL FRAMES ================
        # Calculate partial frame contributions at segment boundaries
        if len(segment) > 2:
            # Use middle frames for statistics, but include partial contributions for timing
            middle_segment = segment[1:-1]
            first_partial = min(segment[0] / np.mean(middle_segment) if len(middle_segment) > 0 else 1, 1)
            last_partial = min(segment[-1] / np.mean(middle_segment) if len(middle_segment) > 0 else 1, 1)
            
            on_time = len(middle_segment) + first_partial + last_partial
            avg_photon = np.mean(middle_segment) if len(middle_segment) > 0 else np.mean(segment)
            std_photon = np.std(middle_segment, ddof=1) if len(middle_segment) > 1 else 0
        else:
            on_time = len(segment)
            avg_photon = np.mean(segment)
            std_photon = np.std(segment, ddof=1) if len(segment) > 1 else 0
        
        # Store results
        t_on.append(on_time)
        sum_photons.append(np.sum(segment))
        avg_photons.append(avg_photon)
        std_photons.append(std_photon)
        start_times.append(start)
        
        # ================ CALCULATE AVERAGE POSITIONS ================
        if x_positions is not None and y_positions is not None and frame_numbers is not None:
            # Find localizations in this time segment.
            # Raw frame numbers are floats, but the trace is indexed by integers.
            # Convert frame_numbers to int to find all corresponding localizations.
            event_mask = np.isin(frame_numbers.astype(int), segment_frames)
            if np.any(event_mask):
                avg_x = np.mean(x_positions[event_mask])
                avg_y = np.mean(y_positions[event_mask])
            else:
                # This fallback is kept for safety but should not be reached.
                avg_x = avg_y = np.nan
        else:
            avg_x = avg_y = np.nan
        
        average_x_positions.append(avg_x)
        average_y_positions.append(avg_y)
        
        # Calculate OFF time to next segment (if not last segment)
        if i < len(on_starts) - 1:
            off_time = on_starts[i + 1] - end
            t_off.append(off_time)
    
    # ================ STEP 5: CALCULATE DERIVED METRICS ================
    # Convert to numpy arrays
    t_on = np.array(t_on)
    t_off = np.array(t_off)
    sum_photons = np.array(sum_photons)
    avg_photons = np.array(avg_photons)
    std_photons = np.array(std_photons)
    start_times = np.array(start_times)
    average_x_positions = np.array(average_x_positions)
    average_y_positions = np.array(average_y_positions)
    
    # Calculate signal metrics
    SNR = np.divide(avg_photons, std_photons, out=np.zeros_like(avg_photons), where=std_photons!=0)
    SBR = avg_photons / bkg if bkg > 0 else np.zeros_like(avg_photons)
    
    # Create photon intensity array (all photons from ON segments)
    photon_intensity = photons_trace[binary_trace > 0]
    
    # ================ DOUBLE EVENT DETECTION ================
    # Detect events with multiple molecules based on intensity analysis
    double_events_counts = np.zeros(len(t_on))
    
    if len(avg_photons) > 0:
        median_intensity = np.median(avg_photons)
        
        for i, (start, end) in enumerate(zip(on_starts, on_ends)):
            segment = photons_trace[start:end]
            
            # Method 1: High intensity events (>1.7x median suggests multiple molecules)
            if avg_photons[i] > 1.7 * median_intensity and len(segment) > 2:
                double_events_counts[i] = 1
            
            # Method 2: Detect intensity steps within long events (>5 frames)
            elif len(segment) > 5:
                # Look for significant intensity changes within the segment
                segment_diff = np.abs(np.diff(segment))
                # If there's a big intensity jump/drop (>0.5x median), it's likely a binding/unbinding within the event
                if np.any(segment_diff > 0.5 * median_intensity):
                    double_events_counts[i] = 1
    
    if verbose_flag:
        print(f'Found {len(t_on)} binding events')
        if len(t_on) > 0:
            print(f'Average ON time: {np.mean(t_on) * exposure_time:.2f} s')
            print(f'Average photons per event: {np.mean(avg_photons):.1f}')
    
    # ================ STEP 6: RETURN RESULTS ================
    # Return results in same format as original function
    return (
        t_on * exposure_time,           # 0: on times in seconds
        t_off * exposure_time,          # 1: off times in seconds  
        binary_trace,                   # 2: binary trace
        start_times * exposure_time,    # 3: start times in seconds
        SNR,                           # 4: signal to noise ratio
        SBR,                           # 5: signal to background ratio
        sum_photons,                   # 6: sum photons per event
        avg_photons,                   # 7: average photons per event
        photon_intensity,              # 8: all photon intensities from ON segments
        std_photons,                   # 9: standard deviation of photons
        start_times * exposure_time,   # 10: start times for avg photons (same as start_times)
        double_events_counts,          # 11: double events detection
        average_x_positions,           # 12: average x positions per event
        average_y_positions            # 13: average y positions per event
    )
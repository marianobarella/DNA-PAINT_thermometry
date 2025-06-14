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
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
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
    # exposure_time in ms
    # threshold in number of photons (integer)

    # ================ INITIALIZE TRACE PROCESSING ================
    # Initial trace processing
    binary_trace = np.where(trace < threshold, 0, 1)
    photons_trace = np.where(trace < threshold, 0, trace)
    diff_binary = np.diff(binary_trace)
    stitched_photons = photons_trace.copy()
    
    # ================ APPLY MASKING FOR DIPS ================
    # Apply masking based on mask_level
    if verbose_flag and mask_level > 0:
        print(f'Using convolution to mask {mask_level} dips...')
    
    if mask_level > 0:
        # Apply appropriate mask based on mask_level
        conv = sig.convolve(diff_binary, mask(mask_level))
        localization_index_dips = np.where(conv == 1)[0] - 1
        binary_trace[localization_index_dips] = 1
        
        # ================ HANDLE DIFFERENT MASK LEVELS ================
        if mask_level == 1:
            # Handle single dips
            for idx in localization_index_dips:
                if idx > 0 and idx < len(photons_trace) - 1:
                    stitched_photons[idx] = (photons_trace[idx - 1] + photons_trace[idx + 1]) / 2
        elif mask_level == 2:
            # Handle double dips
            dips2 = np.where(conv == 1)[0] - 2
            binary_trace[dips2] = 1
            for idx in dips2:
                if idx > 0 and idx < len(photons_trace) - 2:
                    stitched_photons[idx] = (photons_trace[idx - 1] + photons_trace[idx + 2]) / 2
                    stitched_photons[idx + 1] = stitched_photons[idx]
        elif mask_level > 2:
            # Handle multiple dips
            dips2 = np.where(conv == 1)[0] - 2
            binary_trace[dips2] = 1
            for idx in dips2:
                if idx > 1 and idx < len(photons_trace) - mask_level:
                    before = photons_trace[idx - 1]
                    after = photons_trace[idx + mask_level]
                    increment = (after - before) / (mask_level + 1)
                    for i in range(1, mask_level + 1):
                        stitched_photons[idx + i - 1] = before + increment * i
    elif verbose_flag:
        print('No convolution is going to be applied.')

    # ================ APPLY MASKING FOR BLIPS ================
    # Mask single blips if required
    if mask_singles:
        if verbose_flag:
            print('Using convolution to mask single blips...')
        conv_one_blip = sig.convolve(diff_binary, mask(-1))
        localization_index_blips = np.where(np.abs(conv_one_blip) == 1)[0] - 1
        binary_trace[localization_index_blips] = 0
    
    # ================ BEGIN BINDING TIME CALCULATIONS ================
    # Calculate binding times
    if verbose_flag:
        print('Calculating binding times...')
    
    # ================ PROCESS LOCALIZATION INDICES ================
    # Find localization indices and steps
    localization_index = np.where(binary_trace > 0)[0]
    if len(localization_index) == 0:
        return np.array([False] * 11)
        
    localization_index_diff = np.diff(localization_index)
    keep_steps = np.where(localization_index_diff == 1)[0]
    localization_index_steps = localization_index[keep_steps]
    binary_trace[localization_index_steps] = 1
    
    # ================ IDENTIFY EVENT STARTING POINTS ================
    # Determine starting points of events
    try:
        localization_index_start = [localization_index[0] - 1]
        localization_index_start.extend([
            localization_index[i+1] - 1 
            for i, k in enumerate(localization_index_diff) 
            if k > 1
        ])
    except:
        return np.array([False] * 11)
    
    # ================ APPLY BINARY MASK TO TRACE ================
    # Process the photon trace with binary mask
    new_photon_trace = stitched_photons * binary_trace
    
    # ================ CALCULATE SEGMENT STATISTICS ================
    # Calculate segment statistics
    avg_photons = []
    std_photons = []
    start_indices_of_interest = []
    
    for start_index in localization_index_start:
        segment_start = start_index + 1  # Use separate variable instead of modifying loop variable
        # Find end of current segment
        end_index = next((i for i in range(segment_start + 1, len(new_photon_trace)) 
                          if new_photon_trace[i] == 0), len(new_photon_trace))
        
        segment = new_photon_trace[segment_start:end_index]
        if len(segment) > 4:
            avg_photons.append(np.mean(segment[1:-1]))
            std_photons.append(np.std(segment[1:-1], ddof=1))
            start_indices_of_interest.append(segment_start)
    
    # ================ PROCESS BINDING EVENTS ================
    # Process on and off times using groupby
    t_on = []
    double_events_counts = []
    photon_intensity = []
    
    # Group by consecutive nonzero elements
    for is_on, group in groupby(new_photon_trace, key=lambda x: x > 0.01):
        if is_on:  # Process ON segments
            group_list = list(group)
            if len(group_list) > 3:
                group_mean = np.mean(group_list[1:-1])
            else:
                group_mean = np.mean(group_list)
            
            # ================ DETECT DOUBLE EVENTS ================
            # Handle double events detection - ensure every ON segment gets a count
            if len(group_list) > 7:
                window_detection_index = detect_double_events_rolling(group_list, 4)
                double_events_counts.append(len(window_detection_index))
            else:
                double_events_counts.append(0)  # Add zero count for consistency
            
            # ================ HANDLE PARTIAL FRAMES ================
            # Process first and last frames to account for partial events
            first_frame = min(group_list[0]/group_mean, 1)
            last_frame = min(group_list[-1]/group_mean, 1)
            
            # Calculate on-time and collect photon intensity
            if len(group_list) > 2:
                t_on.append(len(group_list[1:-1]) + first_frame + last_frame)
                photon_intensity.extend(group_list[1:-1])
            else:
                t_on.append(len(group_list))
                photon_intensity.extend(group_list)
    
    # ================ PROCESS OFF TIMES ================
    # Process off-times
    t_off = [len(list(group)) for is_off, group in groupby(new_photon_trace, key=lambda x: x < 0.01) if is_off]
    
    # Calculate sum of photons for ON segments
    sum_photons = [np.sum(list(group)) for is_on, group in groupby(new_photon_trace, key=lambda x: x != 0) if is_on]
    
    # ================ HANDLE EDGE CASES ================
    # Handle edge cases - adjust arrays based on trace start/end conditions
    if binary_trace[0] == 1:
        t_on = t_on[1:]
        localization_index_start = localization_index_start[1:]
    else:
        t_off = t_off[1:]
        
    if binary_trace[-1] == 1:
        t_on = t_on[:-1]
        localization_index_start = localization_index_start[:-1]
    else:
        t_off = t_off[:-1]
    
    # ================ PREPARE FINAL RESULTS ================
    # Convert all lists to numpy arrays
    t_on = np.asarray(t_on)
    t_off = np.asarray(t_off)
    sum_photons = np.asarray(sum_photons)
    photon_intensity = np.asarray(photon_intensity)
    double_events_counts = np.asarray(double_events_counts)
    start_time = np.asarray(localization_index_start)
    start_time_avg_photons = np.asarray(start_indices_of_interest)
    avg_photons_np = np.asarray(avg_photons)
    std_photons_np = np.asarray(std_photons)
    
    # ================ CALCULATE SIGNAL METRICS ================
    # Calculate SNR and SBR
    SNR = avg_photons_np / std_photons_np
    SBR = avg_photons_np / bkg
    
    if verbose_flag:
        print('---------------------------')
    
    # Debug visualization (disabled by default)
    if False and index in range(9):
        plt.scatter(start_time*exposure_time, t_on*exposure_time, s=0.8)
        plt.show()
    
    # Return all calculated results with exposure time applied to time-based values
    return (t_on*exposure_time, t_off*exposure_time, binary_trace, start_time*exposure_time, 
            SNR, SBR, sum_photons, avg_photons, photon_intensity, std_photons, 
            start_time_avg_photons*exposure_time, double_events_counts)


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




# -*- coding: utf-8 -*-
"""
Created on Tuesday April 4th of 2023

@author: Mariano Barella

Version 3. Changes:
    - new flag that select if structures are origamis or hybridized structures
    - automatically selects the best threshold of the peak finding algorithm

This script analyzes already-processed Picasso data. It opens .dat files that 
were generated with "extract_and_save_data_from_hdf5_picasso_files.py".

When the program starts select ANY .dat file. This action will determine the 
working folder.

As input it uses:
    - main folder
    - number of frames
    - exposure time
    - if NP is present (hybridized structure)
    - pixel size of the original video
    - size of the pick you used in picasso analysis pipeline
    - desired radius of analysis to average localization position
    - number of docking sites you are looking for (defined by origami design)
    
Outputs are:
    - plots per pick (scatter plot of locs, fine and coarse 2D histograms,
                      binary image showing center of docking sites,
                      matrix of relative distances, matrix of localization precision)
    - traces per pick
    - a single file with ALL traces of the super-resolved image
    - global figures (photons vs time, localizations vs time and background vs time)

Warning: the program is coded to follow filename convention of the script
"extract_and_save_data_from_hdf5_picasso_files.py".

"""
# ================ IMPORT LIBRARIES ================
import os

import scipy.signal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Circle as plot_circle
import tkinter as tk
import tkinter.filedialog as fd
import re
from auxiliary_functions import detect_peaks, detect_peaks_improved, get_peak_detection_histogram, distance, fit_linear, \
    perpendicular_distance, manage_save_directory, plot_vs_time_with_hist, update_pkl, \
    calculate_tau_on_times_average
from sklearn.mixture import GaussianMixture
import time
from auxiliary_functions_gaussian import plot_gaussian_2d
import scipy
import glob

# ================ MATPLOTLIB CONFIGURATION ================
plt.ioff()  # Turn off interactive mode
plt.close("all")
cmap = plt.cm.get_cmap('viridis')
bkg_color = cmap(0)

##############################################################################

def process_dat_files(number_of_frames, exp_time, working_folder,
                      docking_sites, NP_flag, pixel_size, pick_size, 
                      radius_of_pick_to_average, th, plot_flag, verbose_flag):
    
    print('\nStarting STEP 2.')
    
    # ================ CALCULATE TIME PARAMETERS ================
    total_time_sec = number_of_frames*exp_time # in sec
    total_time_min = total_time_sec/60 # in min
    #print('Total time %.1f min' % total_time_min)
        
    # ================ CREATE FOLDER STRUCTURE FOR SAVING DATA ================
    # Create step2 folder structure with method-specific separation
    # working_folder is .../analysis/step1/data, need to go back to main experiment folder
    main_folder = os.path.dirname(os.path.dirname(os.path.dirname(working_folder)))  # Go back to main experiment folder
    analysis_folder = os.path.join(main_folder, 'analysis')
    
    # Create method-specific subfolders to prevent file mixing
    method_subfolder = 'position_averaging_method'  # This is the position averaging method
    step2_base_folder = manage_save_directory(analysis_folder, 'step2')
    step2_method_folder = manage_save_directory(step2_base_folder, method_subfolder)
    
    figures_folder = manage_save_directory(step2_method_folder, 'figures')
    figures_per_pick_folder = manage_save_directory(figures_folder, 'per_pick')
    data_folder = manage_save_directory(step2_method_folder, 'data')
    traces_per_pick_folder = manage_save_directory(data_folder, 'traces')
    traces_per_site_folder = manage_save_directory(traces_per_pick_folder, 'traces_per_site')
    kinetics_folder = manage_save_directory(data_folder, 'kinetics_data')
    gaussian_folder = manage_save_directory(kinetics_folder, 'gaussian_data')

    # ================ CLEAN UP EXISTING TRACE FILES ================
    if os.path.exists(traces_per_site_folder):
        for f in os.listdir(traces_per_site_folder):
            file_path = os.path.join(traces_per_site_folder, f)  # Combine directory path and file name
            if os.path.isfile(file_path):  # Ensure it's a file (not a directory)
                os.remove(file_path)  # Remove the file
            else:
                print(f"{file_path} is not a file, skipping.")

    # ================ LIST AND FILTER INPUT FILES ================
    list_of_files = os.listdir(working_folder)
    list_of_files = [f for f in list_of_files if re.search('.dat', f)]
    list_of_files.sort()
    if NP_flag:
        list_of_files_origami = [f for f in list_of_files if re.search('NP_subtracted',f)]
        list_of_files_NP = [f for f in list_of_files if re.search('raw',f)]
    else:
        list_of_files_origami = list_of_files
    
    ##############################################################################
    # ================ LOAD INPUT DATA ================
    
    # frame number, used for time estimation
    frame_file = [f for f in list_of_files_origami if re.search('_frame',f)][0]
    frame_filepath = os.path.join(working_folder, frame_file)
    frame = np.loadtxt(frame_filepath)
    
    # photons
    photons_file = [f for f in list_of_files_origami if re.search('_photons',f)][0]
    photons_filepath = os.path.join(working_folder, photons_file)
    photons = np.loadtxt(photons_filepath)
    if NP_flag:
        photons_file_NP = [f for f in list_of_files_NP if re.search('_photons', f)][0]
        photons_filepath_NP = os.path.join(working_folder, photons_file_NP)
        photons = np.loadtxt(photons_filepath_NP)

    
    # bkg
    bkg_file = [f for f in list_of_files_origami if re.search('_bkg',f)][0]
    bkg_filepath = os.path.join(working_folder, bkg_file)
    bkg = np.loadtxt(bkg_filepath)
    
    # xy positions
    # origami
    position_file = [f for f in list_of_files_origami if re.search('_xy',f)][0]
    position_filepath = os.path.join(working_folder, position_file)
    position = np.loadtxt(position_filepath)
    x = position[:,0]*pixel_size
    y = position[:,1]*pixel_size
    # NP
    if NP_flag:
        position_file_NP = [f for f in list_of_files_NP if re.search('_xy',f)][0]
        position_filepath_NP = os.path.join(working_folder, position_file_NP)
        xy_NP = np.loadtxt(position_filepath_NP)
        x_NP = xy_NP[:,0]*pixel_size
        y_NP = xy_NP[:,1]*pixel_size
    
    # number of pick
    # origami
    pick_file = [f for f in list_of_files_origami if re.search('_pick_number',f)][0]
    pick_filepath = os.path.join(working_folder, pick_file)
    pick_list = np.loadtxt(pick_filepath)
    # NP
    if NP_flag:
        pick_file_NP = [f for f in list_of_files_NP if re.search('_pick_number',f)][0]
        pick_filepath_NP = os.path.join(working_folder, pick_file_NP)
        pick_list_NP = np.loadtxt(pick_filepath_NP)
    
    ##############################################################################
    
    # ================ INITIALIZE ANALYSIS VARIABLES ================
    # how many picks?
    pick_number = np.unique(pick_list)
    total_number_of_picks = len(pick_number)
    #print('Total picks', total_number_of_picks)
    
    # allocate arrays for statistics
    locs_of_picked = np.zeros(total_number_of_picks)
    # number of bins for temporal binning
    number_of_bins = 60
    locs_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
    photons_concat = np.array([])
    bkg_concat = np.array([])
    frame_concat = np.array([])
    positions_concat_NP = np.array([])
    positions_concat_origami = np.array([])
    gmm_stds = []
    gmm_stds_x = np.array([])
    gmm_stds_y = np.array([])
    all_traces = np.zeros(number_of_frames)
    all_traces_per_site = {}
    
    # ================ INITIALIZE KINETICS ARRAYS ================
    # Arrays for collecting kinetics data from all picks
    tons_all = np.array([])
    toffs_all = np.array([])
    tstarts_all = np.array([])
    SNR_all = np.array([])
    SBR_all = np.array([])
    sum_photons_all = np.array([])
    avg_photons_all = np.array([])
    photon_intensity_all = np.array([])
    std_photons_all = np.array([])
    double_events_all = np.array([])
    
    # ================ HISTOGRAM CONFIGURATION ================
    # set number of bins for FINE histograming 
    N = int(1 * 2*pick_size*pixel_size*1000/10)
    hist_2D_bin_size = pixel_size*1000*pick_size/N # this should be around 5 nm
    if verbose_flag:
        print(f'2D histogram bin size: {hist_2D_bin_size:.2f} nm')
    ########################################################################
    ########################################################################
    ########################################################################
    site_index = -1
    # ================ BEGIN ANALYSIS OF EACH PICK ================
    # data assignment per pick
    # TODO: If it doesn't find the correct amount of binding sites either discard or find less.
    for i in range(total_number_of_picks):
        pick_id = pick_number[i]
        if verbose_flag:
            print('\n---------- Pick number %d of %d\n' % (i+1, total_number_of_picks))

        # ================ EXTRACT PICK DATA ================
        # Get data for current pick
        index_picked = np.where(pick_list == pick_id)[0]
        frame_of_picked = frame[index_picked]
        photons_of_picked = photons[index_picked]
        bkg_of_picked = bkg[index_picked]
        x_position_of_picked_raw = x[index_picked]
        y_position_of_picked_raw = y[index_picked]
        
        # ================ CALCULATE AVERAGED POSITIONS FROM BINDING EVENTS ================
        # Create a trace for this pick to identify binding events
        pick_trace = np.zeros(number_of_frames)
        np.add.at(pick_trace, frame_of_picked.astype(int), photons_of_picked)
        
        # Use calculate_tau_on_times_average to get averaged positions for binding events
        # Set parameters for binding event detection
        photons_threshold = np.mean(photons_of_picked) * 0.1  # Simple threshold
        background_level = np.mean(bkg_of_picked)
        mask_level = 1  # Simple masking
        mask_singles = False
        
        # Get averaged positions for binding events
        tau_results = calculate_tau_on_times_average(
            pick_trace, photons_threshold, background_level, exp_time,
            mask_level, mask_singles, False, i,  # verbose_flag=False
            x_position_of_picked_raw, y_position_of_picked_raw, frame_of_picked
        )
        
        # Extract averaged positions from results
        if tau_results[0] is not False and len(tau_results) >= 14:
            avg_x_positions = tau_results[12]  # average_x_positions
            avg_y_positions = tau_results[13]  # average_y_positions
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(avg_x_positions) | np.isnan(avg_y_positions))
            if np.any(valid_mask):
                x_position_of_picked = avg_x_positions[valid_mask]
                y_position_of_picked = avg_y_positions[valid_mask]
                
                # Create corresponding frame and photon data for averaged positions
                # Use the start times from the binding events
                binding_start_times = tau_results[3]  # start_time
                valid_start_times = binding_start_times[valid_mask]
                frame_of_picked = (valid_start_times / exp_time).astype(int)
                
                # Use sum photons for each event instead of individual photons
                sum_photons_events = tau_results[6]  # sum_photons
                photons_of_picked = sum_photons_events[valid_mask]
                
                if verbose_flag:
                    print(f'Using {len(x_position_of_picked)} averaged positions from {len(x_position_of_picked_raw)} raw localizations')
            else:
                # Fall back to raw data if no valid averaged positions
                x_position_of_picked = x_position_of_picked_raw
                y_position_of_picked = y_position_of_picked_raw
                if verbose_flag:
                    print('No valid averaged positions found, using raw localizations')
        else:
            # Fall back to raw data if averaging failed
            x_position_of_picked = x_position_of_picked_raw
            y_position_of_picked = y_position_of_picked_raw
            if verbose_flag:
                print('Position averaging failed, using raw localizations')
        
        # Set boundaries for histograms
        x_min = min(x_position_of_picked)
        y_min = min(y_position_of_picked)
        x_max = x_min + pick_size*pixel_size
        y_max = y_min + pick_size*pixel_size
        hist_bounds = [[x_min, x_max], [y_min, y_max]]

        # ================ CREATE FINE 2D HISTOGRAM ================
        z_hist, x_hist, y_hist = np.histogram2d(x_position_of_picked, y_position_of_picked, 
                                               bins=N, range=hist_bounds)
        z_hist = z_hist.T
        x_hist_step = np.diff(x_hist)
        y_hist_step = np.diff(y_hist)
        x_hist_centers = x_hist[:-1] + x_hist_step/2
        y_hist_centers = y_hist[:-1] + y_hist_step/2
        
        # ================ IMPROVED PEAK DETECTION ================
        # Use improved peak detection function from auxiliary_functions
        min_distance_nm = 15  # Minimum 15nm separation between peaks
        
        # Call the function we created instead of duplicating code
        peak_coords = detect_peaks_improved(
            x_position_of_picked, y_position_of_picked, 
            hist_bounds, expected_peaks=docking_sites, 
            min_distance_nm=min_distance_nm
        )
        
        total_peaks_found = len(peak_coords)
        docking_sites_temp = min(total_peaks_found, docking_sites)
                
        # ================ VERIFY PEAK DETECTION RESULTS ================
        peaks_flag = total_peaks_found > 0
        
        # ================ INITIALIZE BINDING SITE ARRAYS ================
        # Initialize arrays for binding sites
        analysis_radius = radius_of_pick_to_average*pixel_size
        cm_binding_sites_x = np.array([])
        cm_binding_sites_y = np.array([])
        cm_std_dev_binding_sites_x = np.array([])
        cm_std_dev_binding_sites_y = np.array([])
        all_traces_per_pick = np.zeros(number_of_frames)
        inv_cov_init = []
        
        if docking_sites_temp < docking_sites and verbose_flag:
            print(f'Did not find {docking_sites} docking sites for origami nr {i}.')
            
        # ================ PROCESS EACH DETECTED PEAK ================
        if peaks_flag:
            # peak_coords is already calculated by detect_peaks_improved
            
            for j in range(total_peaks_found):
                if docking_sites_temp != 1 and verbose_flag:
                    print('Binding site %d of %d' % (j+1, total_peaks_found))
                
                x_peak, y_peak = peak_coords[j]
                
                # ================ FILTER LOCALIZATIONS BY DISTANCE ================
                # Calculate distances once
                d = np.sqrt((x_position_of_picked - x_peak)**2 + 
                           (y_position_of_picked - y_peak)**2)
                
                # Filter by radius
                index_inside_radius = d < analysis_radius
                x_position_filtered = x_position_of_picked[index_inside_radius]
                y_position_filtered = y_position_of_picked[index_inside_radius]
                
                # ================ CALCULATE BINDING SITE STATISTICS ================
                # Calculate stats
                cm_binding_site_x = np.mean(x_position_filtered)
                cm_binding_site_y = np.mean(y_position_filtered)
                cm_std_dev_binding_site_x = np.std(x_position_filtered, ddof=1)
                cm_std_dev_binding_site_y = np.std(y_position_filtered, ddof=1)
                
                # Append to arrays
                cm_binding_sites_x = np.append(cm_binding_sites_x, cm_binding_site_x)
                cm_binding_sites_y = np.append(cm_binding_sites_y, cm_binding_site_y)
                cm_std_dev_binding_sites_x = np.append(cm_std_dev_binding_sites_x, cm_std_dev_binding_site_x)
                cm_std_dev_binding_sites_y = np.append(cm_std_dev_binding_sites_y, cm_std_dev_binding_site_y)
                
                # ================ CREATE AND COMPILE TRACES ================
                # Process trace data
                frame_of_picked_filtered = frame_of_picked[index_inside_radius].astype(int)
                photons_of_picked_filtered = photons_of_picked[index_inside_radius]
                
                # Vectorized trace creation
                trace = np.zeros(number_of_frames)
                np.add.at(trace, frame_of_picked_filtered, photons_of_picked_filtered)
                
                # Compile traces
                all_traces_per_pick = np.vstack([all_traces_per_pick, trace])
                all_traces = np.vstack([all_traces, trace])
                
                # ================ CALCULATE COVARIANCE MATRIX ================
                # Calculate inverse covariance matrix for GMM
                try:
                    cov_data = np.array([x_position_filtered, y_position_filtered])
                    inv_cov_init.append(np.linalg.inv(np.cov(cov_data)))
                except:
                    inv_cov_init = 'False'
            
            # ================ CLEAN UP AND SAVE TRACES ================
            # Clean up traces data
            all_traces_per_pick = np.delete(all_traces_per_pick, 0, axis=0)
            all_traces_per_pick = all_traces_per_pick.T
            
            # Save traces per pick if peaks were found
            if peaks_flag:
                new_filename = 'TRACE_pick_%02d.dat' % i
                new_filepath = os.path.join(traces_per_pick_folder, new_filename)
                np.savetxt(new_filepath, trace, fmt='%05d')
        
        # ================ PROCESS NANOPARTICLE (NP) DATA ================
        x_avg_NP = y_avg_NP = x_std_dev_NP = y_std_dev_NP = None
        if NP_flag:
            # Filter out high photon events
            low_photons_indices = photons < (np.mean(photons) + 0.5*np.std(photons))
            index_picked_NP = pick_list_NP == pick_id
            filtered_indices_NP_2 = np.where(index_picked_NP & low_photons_indices)[0]
            
            x_position_of_picked_NP = x_NP[filtered_indices_NP_2]
            y_position_of_picked_NP = y_NP[filtered_indices_NP_2]
            x_avg_NP = np.mean(x_position_of_picked_NP)
            y_avg_NP = np.mean(y_position_of_picked_NP)
            x_std_dev_NP = np.std(x_position_of_picked_NP, ddof=1)
            y_std_dev_NP = np.std(y_position_of_picked_NP, ddof=1)

        # ================ FIT LINEAR DIRECTION OF BINDING SITES ================
        if peaks_flag and len(cm_binding_sites_x) > 1:
            x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(
                cm_binding_sites_x, cm_binding_sites_y)
            
            # Calculate perpendicular distances
            perpendicular_dist_of_picked = perpendicular_distance(
                slope, intercept, x_position_of_picked, y_position_of_picked)
            
            # Filter localizations based on perpendicular distance
            filter_dist = 45e-3  # Arbitrary value
            perpendicular_mask = perpendicular_dist_of_picked < filter_dist
            x_filtered_perpendicular = x_position_of_picked[perpendicular_mask]
            y_filtered_perpendicular = y_position_of_picked[perpendicular_mask]
            
            # Save filtered coordinates
            new_filename = f'xy_perpendicular_filtered_{i}.dat'
            new_filepath = os.path.join(gaussian_folder, new_filename)
            np.savetxt(new_filepath, np.column_stack((x_filtered_perpendicular, y_filtered_perpendicular)))
            
            # ================ CALCULATE NP DISTANCES ================
            if NP_flag and peaks_flag:
                distance_to_NP = perpendicular_distance(slope, intercept, x_avg_NP, y_avg_NP)
                distance_to_NP_nm = distance_to_NP * 1e3
                binding_site_radial_distance_to_NP = np.sqrt(
                    (cm_binding_sites_x - x_avg_NP)**2 + (cm_binding_sites_y - y_avg_NP)**2)
                binding_site_radial_distance_to_NP_nm = binding_site_radial_distance_to_NP * 1e3
        
        # ================ CALCULATE DISTANCE MATRICES ================
        if peaks_flag:
            # Initialize matrices
            matrix_distance = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
            matrix_std_dev = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
            
            # NP to binding sites distances
            if NP_flag and peaks_flag:
                # Vectorized distance calculation
                np_to_binding_distances = np.sqrt(
                    (cm_binding_sites_x - x_avg_NP)**2 + 
                    (cm_binding_sites_y - y_avg_NP)**2) * 1e3
                
                matrix_distance[0, 1:] = np_to_binding_distances
                matrix_distance[1:, 0] = np_to_binding_distances
                matrix_std_dev[0, 0] = max(x_std_dev_NP, y_std_dev_NP) * 1e3
                positions_concat_NP = np.append(positions_concat_NP, np_to_binding_distances)
            
            # ================ CALCULATE BINDING SITE DISTANCES ================
            # Binding site to binding site distances
            peak_distances = np.array([])
            for j in range(total_peaks_found):
                x_binding_row = cm_binding_sites_x[j]
                y_binding_row = cm_binding_sites_y[j]
                matrix_std_dev[j + 1, j + 1] = max(
                    cm_std_dev_binding_sites_x[j], cm_std_dev_binding_sites_y[j]) * 1e3
                
                for k in range(j + 1, total_peaks_found):
                    x_binding_col = cm_binding_sites_x[k]
                    y_binding_col = cm_binding_sites_y[k]
                    distance_between_locs_CM = np.sqrt(
                        (x_binding_col - x_binding_row)**2 + 
                        (y_binding_col - y_binding_row)**2) * 1e3
                    
                    matrix_distance[j + 1, k + 1] = distance_between_locs_CM
                    matrix_distance[k + 1, j + 1] = distance_between_locs_CM
                    peak_distances = np.append(peak_distances, distance_between_locs_CM)
                    positions_concat_origami = np.append(positions_concat_origami, distance_between_locs_CM)
            
            # ================ LABEL BINDING SITES ================
            # Assigning peak labels using distances
            peak_mean_distance = np.zeros(total_peaks_found)
            for l in range(1, total_peaks_found+1):
                peak_mean_distance[l-1] = np.mean(matrix_distance[l, 1:])
            
            ascending_index = peak_mean_distance.argsort()
            ranks = ascending_index.argsort()
            
            # ================ PROCESS TRACES PER SITE ================
            all_traces_per_site_per_pick = {}
            site_index = -1
            
            for h in range(total_peaks_found):
                site_index += 1
                trace = all_traces_per_pick[:, site_index]
                trace_no_zeros = trace[trace != 0]
                all_traces_per_site_per_pick[str(ranks[h])] = trace
                
                if total_peaks_found == docking_sites:
                    if str(ranks[h]) in all_traces_per_site:
                        all_traces_per_site[str(ranks[h])] = np.append(all_traces_per_site[str(ranks[h])], trace_no_zeros)
                    else:
                        all_traces_per_site[str(ranks[h])] = trace_no_zeros
            
            # ================ SAVE TRACES PER SITE ================
            for index, key in enumerate(all_traces_per_site_per_pick.keys()):
                site_trace = all_traces_per_site_per_pick[key]
                # Handle NaN values in distance matrix - set to 0 if NaN or not available
                if not NP_flag:
                    dist_value = 0
                else:
                    raw_dist = matrix_distance[0, index+1]
                    dist_value = 0 if np.isnan(raw_dist) else round(raw_dist, 2)
                new_filename = f'TRACE_pick_{i}_site_{int(key)}_dist_{dist_value}.dat'
                new_filepath = os.path.join(traces_per_site_folder, new_filename)
                np.savetxt(new_filepath, site_trace, fmt='%05d')
                
                # ================ CALCULATE PER-SITE KINETICS ================
                # Calculate kinetics for this individual site
                site_tau_results = calculate_tau_on_times_average(
                    site_trace, photons_threshold, background_level, exp_time,
                    mask_level, mask_singles, False, site_index
                )
                
                if site_tau_results[0] is not False and len(site_tau_results) >= 14:
                    # Extract and save per-site kinetics data
                    site_tons = site_tau_results[0]
                    site_toffs = site_tau_results[1] 
                    site_sum_photons = site_tau_results[6]
                    site_avg_photons = site_tau_results[7]
                    site_photon_intensity = site_tau_results[8]
                    site_std_photons = site_tau_results[9]
                    
                    # Create per-site kinetics folders
                    ton_per_site_folder = manage_save_directory(data_folder, 'ton_per_site')
                    toff_per_site_folder = manage_save_directory(data_folder, 'toff_per_site')
                    mean_photons_per_site_folder = manage_save_directory(data_folder, 'mean_photons_per_site')
                    std_photons_per_site_folder = manage_save_directory(data_folder, 'std_photons_per_site')
                    sum_photons_per_site_folder = manage_save_directory(data_folder, 'sum_photons_per_site')
                    photons_per_site_folder = manage_save_directory(data_folder, 'photons_per_site')
                    
                    # Save per-site kinetics files
                    # Ensure dist_value is valid for integer conversion
                    safe_dist_value = 0 if np.isnan(dist_value) else int(dist_value)
                    base_name = f'pick_{i}_site_{int(key)}_dist_{safe_dist_value}'
                    
                    if len(site_tons) > 0:
                        np.savetxt(os.path.join(ton_per_site_folder, f'ton_{base_name}.dat'), 
                                  site_tons, fmt='%.3f')
                    if len(site_toffs) > 0:
                        np.savetxt(os.path.join(toff_per_site_folder, f'toff_{base_name}.dat'), 
                                  site_toffs, fmt='%.3f')
                    if len(site_avg_photons) > 0:
                        np.savetxt(os.path.join(mean_photons_per_site_folder, f'meanphotons_{base_name}.dat'), 
                                  site_avg_photons, fmt='%.3f')
                    if len(site_std_photons) > 0:
                        np.savetxt(os.path.join(std_photons_per_site_folder, f'stdphotons_{base_name}.dat'), 
                                  site_std_photons, fmt='%.3f')
                    if len(site_sum_photons) > 0:
                        np.savetxt(os.path.join(sum_photons_per_site_folder, f'sumphotons_{base_name}.dat'), 
                                  site_sum_photons, fmt='%.3f')
                    if len(site_photon_intensity) > 0:
                        np.savetxt(os.path.join(photons_per_site_folder, f'photons_{base_name}.dat'), 
                                  site_photon_intensity, fmt='%.3f')
        
        # ================ COMPILE DATA FOR HISTOGRAMS ================
        photons_concat = np.concatenate([photons_concat, photons_of_picked])
        bkg_concat = np.concatenate([bkg_concat, bkg_of_picked])
        frame_concat = np.concatenate([frame_concat, frame_of_picked])
        locs_of_picked[i] = len(frame_of_picked)
        
        # ================ CREATE TIME HISTOGRAM ================
        hist_range = [0, number_of_frames]
        bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
        locs_of_picked_vs_time[i,:], bin_edges = np.histogram(
            frame_of_picked, bins=number_of_bins, range=hist_range)
        bin_centers = bin_edges[:-1] + bin_size/2
        bin_centers_minutes = bin_centers*exp_time/60
        
        # ================ GENERATE PICK PLOTS ================
        if plot_flag:
            # Plot when the pick was bright vs time
            plt.figure()
            plt.step(bin_centers_minutes, locs_of_picked_vs_time[i,:], where='mid', label=f'Pick {i:04d}')
            plt.xlabel('Time (min)')
            plt.ylabel('Locs')
            plt.ylim([0, 80])
            ax = plt.gca()
            ax.axvline(x=10, ymin=0, ymax=1, color='k', linewidth='2', linestyle='--')
            ax.set_title(f'Number of locs per pick vs time. Bin size {bin_size*0.1/60:.1f} min')
            aux_folder = manage_save_directory(figures_per_pick_folder, 'locs_vs_time_per_pick')
            figure_name = f'locs_per_pick_vs_time_pick_{i:02d}'
            figure_path = os.path.join(aux_folder, f'{figure_name}.png')
            plt.savefig(figure_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # ================ PLOT SCATTER WITH NP ================
            if peaks_flag:
                # RAW scatter plot
                plt.figure(100)
                plt.scatter(x_position_of_picked, y_position_of_picked, color='C0', label='Fluorophore Emission', s=0.5)
                
                if NP_flag:
                    plt.scatter(x_position_of_picked_NP, y_position_of_picked_NP, color='C1', s=0.5, alpha=0.2)
                    plt.scatter(x_avg_NP, y_avg_NP, color='C1', label='NP Scattering', s=0.5, alpha=1)
                    plt.plot(x_avg_NP, y_avg_NP, 'x', color='k', label='Center of NP')
                    plt.legend(loc='upper left')
                
                plt.ylabel(r'y ($\mu$m)')
                plt.xlabel(r'x ($\mu$m)')
                plt.xlim(left=0)
                plt.ylim(bottom=0)
                plt.axis('square')
                ax = plt.gca()
                ax.set_title(f'Position of locs per pick. Pick {i:02d}')
                aux_folder = manage_save_directory(figures_per_pick_folder, 'scatter_plots')
                figure_name = f'xy_pick_scatter_NP_and_PAINT_{i:02d}'
                figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # ================ PLOT SCATTER WITH PAINT POINTS ================
                plt.figure(2)
                plt.plot(x_position_of_picked, y_position_of_picked, '.', color='C0', label='PAINT', linewidth=0.5)
                
                if NP_flag:
                    plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=10, markerfacecolor='C1', 
                            markeredgecolor='k', label='NP')
                    plt.legend(loc='upper left')
                
                plt.ylabel(r'y ($\mu$m)')
                plt.xlabel(r'x ($\mu$m)')
                plt.axis('square')
                ax = plt.gca()
                ax.set_title(f'Position of locs per pick. Pick {i:02d}')
                aux_folder = manage_save_directory(figures_per_pick_folder, 'scatter_plots')
                figure_name = f'xy_pick_scatter_PAINT_{i:02d}'
                figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # ================ PLOT FINE 2D IMAGE ================
                plt.figure(3)
                plt.imshow(z_hist, interpolation='none', origin='lower',
                          extent=[x_hist_centers[0], x_hist_centers[-1], 
                                  y_hist_centers[0], y_hist_centers[-1]])
                ax = plt.gca()
                ax.set_facecolor(bkg_color)
                
                if total_peaks_found > 0:
                    peak_x = [coord[0] for coord in peak_coords]
                    peak_y = [coord[1] for coord in peak_coords]
                    plt.plot(peak_x, peak_y, 'x', markersize=9, 
                            color='white', markeredgecolor='black', mew=1, label='binding sites')
                    
                    for k, (circle_x, circle_y) in enumerate(peak_coords):
                        circ = plot_circle((circle_x, circle_y), radius=analysis_radius, 
                                          facecolor='none', edgecolor='white', linewidth=1)
                        ax.add_patch(circ)
                        
                        # Peak labels  
                        label_x = circle_x + analysis_radius*1.25
                        label_y = circle_y + analysis_radius*0.25
                        text_content = f"{k}"
                        ax.text(label_x, label_y, text_content, ha='center', va='center', 
                               rotation=0, fontsize=12, color='white')
                
                if NP_flag and x_avg_NP is not None:
                    plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=8, markerfacecolor='C1', 
                            markeredgecolor='white', label='NP')
                    plt.legend(loc='upper right')
                
                plt.ylabel(r'y ($\mu$m)')
                plt.xlabel(r'x ($\mu$m)')
                cbar = plt.colorbar()
                cbar.ax.set_title(u'Locs', fontsize=16)
                cbar.ax.tick_params(labelsize=16)
                ax.set_title(f'Pick {i:02d}', fontsize=10)
                aux_folder = manage_save_directory(figures_per_pick_folder, 'image_FINE')
                figure_name = f'xy_pick_image_PAINT_{i:02d}'
                figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # ================ PLOT PEAK DETECTION PROCESS ================
                if total_peaks_found > 0:
                    # Use the auxiliary function to get histogram data (same logic as detect_peaks_improved)
                    z_hist_smooth_detect, x_centers_detect, y_centers_detect = get_peak_detection_histogram(
                        x_position_of_picked, y_position_of_picked, hist_bounds
                    )
                    
                    plt.figure(5)
                    plt.imshow(z_hist_smooth_detect, interpolation='none', origin='lower',
                              extent=[x_centers_detect[0], x_centers_detect[-1], 
                                      y_centers_detect[0], y_centers_detect[-1]], 
                              cmap='viridis')
                    ax = plt.gca()
                    ax.set_facecolor(bkg_color)
                    
                    # Show detected peaks as red circles
                    peak_x = [coord[0] for coord in peak_coords]
                    peak_y = [coord[1] for coord in peak_coords]
                    plt.scatter(peak_x, peak_y, c='red', s=150, marker='o', 
                              edgecolors='white', linewidths=2, label=f'{total_peaks_found} detected peaks')
                    
                    # Add peak numbers
                    for k, (px, py) in enumerate(peak_coords):
                        ax.text(px, py, f'{k}', ha='center', va='center', 
                               fontsize=10, color='white', fontweight='bold')
                    
                    if NP_flag and x_avg_NP is not None:
                        plt.plot(x_avg_NP, y_avg_NP, 's', markersize=10, markerfacecolor='yellow', 
                                markeredgecolor='black', label='NP')
                    
                    plt.legend(loc='upper right')
                    plt.ylabel(r'y ($\mu$m)')
                    plt.xlabel(r'x ($\mu$m)')
                    cbar = plt.colorbar()
                    cbar.ax.set_title(u'Density')
                    cbar.ax.tick_params()
                    ax.set_title(f'Peak detection - Pick {i:02d}', fontsize=10)
                    aux_folder = manage_save_directory(figures_per_pick_folder, 'image_peak_detection')
                    figure_name = f'peak_detection_process_{i:02d}'
                    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                    plt.close()
                
                # ================ PLOT DISTANCE MATRICES ================
                if len(cm_binding_sites_x) > 1:
                    # Matrix distance plot
                    plt.figure(10)
                    plt.imshow(matrix_distance, interpolation='none', cmap='spring')
                    ax = plt.gca()
                    
                    for l in range(matrix_distance.shape[0]):
                        for m in range(matrix_distance.shape[1]):
                            if l == m:
                                ax.text(m, l, '-', ha="center", va="center", color=[0,0,0], fontsize=18)
                            else:
                                ax.text(m, l, '%.0f' % matrix_distance[l, m],
                                       ha="center", va="center", color=[0,0,0], fontsize=18)
                    
                    ax.xaxis.tick_top()
                    ax.set_xticks(np.array(range(matrix_distance.shape[1])))
                    ax.set_yticks(np.array(range(matrix_distance.shape[0])))
                    axis_string = ['NP']
                    for j in range(total_peaks_found):
                        axis_string.append(f'Site {j+1}')
                    ax.set_xticklabels(axis_string)
                    ax.set_yticklabels(axis_string)
                    aux_folder = manage_save_directory(figures_per_pick_folder, 'matrix_distance')
                    figure_name = f'matrix_distance_{i:02d}'
                    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                    plt.savefig(figure_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Matrix std dev plot
                    plt.figure(11)
                    plt.imshow(matrix_std_dev, interpolation='none', cmap='spring')
                    ax = plt.gca()
                    
                    for l in range(matrix_distance.shape[0]):
                        for m in range(matrix_distance.shape[1]):
                            if l != m:
                                ax.text(m, l, '-', ha="center", va="center", color=[0,0,0], fontsize=18)
                            else:
                                ax.text(m, l, '%.0f' % matrix_std_dev[l, m],
                                       ha="center", va="center", color=[0,0,0], fontsize=18)
                    
                    ax.xaxis.tick_top()
                    ax.set_xticks(np.array(range(matrix_distance.shape[1])))
                    ax.set_yticks(np.array(range(matrix_distance.shape[0])))
                    axis_string = ['NP']
                    for j in range(total_peaks_found):
                        axis_string.append(f'Site {j+1}')
                    ax.set_xticklabels(axis_string)
                    ax.set_yticklabels(axis_string)
                    aux_folder = manage_save_directory(figures_per_pick_folder, 'matrix_std_dev')
                    figure_name = f'matrix_std_dev_{i:02d}'
                    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                    plt.savefig(figure_path, dpi=100, bbox_inches='tight')
                    plt.close()

        # ================ COLLECT KINETICS DATA ================
        # Collect kinetics data from tau_results for overall analysis
        if tau_results[0] is not False and len(tau_results) >= 14:
            # Extract kinetics data from tau_results
            tons_pick = tau_results[0]    # on times
            toffs_pick = tau_results[1]   # off times
            tstarts_pick = tau_results[3] # start times
            SNR_pick = tau_results[4]     # signal to noise ratio
            SBR_pick = tau_results[5]     # signal to background ratio
            sum_photons_pick = tau_results[6]      # sum photons per event
            avg_photons_pick = tau_results[7]      # average photons per event
            photon_intensity_pick = tau_results[8] # photon intensities
            std_photons_pick = tau_results[9]      # std photons per event
            double_events_pick = tau_results[11]   # double events count
            
            # Append to global arrays (will be initialized before the loop)
            if len(tons_pick) > 0:  # Only append if there are valid events
                tons_all = np.append(tons_all, tons_pick)
                toffs_all = np.append(toffs_all, toffs_pick)
                tstarts_all = np.append(tstarts_all, tstarts_pick)
                SNR_all = np.append(SNR_all, SNR_pick)
                SBR_all = np.append(SBR_all, SBR_pick)
                sum_photons_all = np.append(sum_photons_all, sum_photons_pick)
                avg_photons_all = np.append(avg_photons_all, avg_photons_pick)
                photon_intensity_all = np.append(photon_intensity_all, photon_intensity_pick)
                std_photons_all = np.append(std_photons_all, std_photons_pick)
                double_events_all = np.append(double_events_all, double_events_pick)

    # ================ GLOBAL ANALYSIS AND VISUALIZATION ================
    # ================ PLOT NP RELATIVE POSITIONS ================
    number_of_bins = 16
    hist_range = [25, 160]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    position_bins, bin_edges = np.histogram(positions_concat_NP, bins = number_of_bins, \
                                            range = hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    plt.figure()
    plt.bar(bin_centers, position_bins, width = 0.8*bin_size, align = 'center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    figure_name = 'relative_positions_NP_sites'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    # ================ PLOT BINDING SITE RELATIVE POSITIONS ================
    number_of_bins = 16
    hist_range = [25, 160]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    #print('\nRelative position between binding sites bin size', bin_size)
    position_bins, bin_edges = np.histogram(positions_concat_origami, bins = number_of_bins, \
                                            range = hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    plt.figure()
    plt.bar(bin_centers, position_bins, width = 0.8*bin_size, align = 'center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    figure_name = 'relative_positions_binding_sites'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    # ================ PLOT GLOBAL TIME SERIES ANALYSIS ================
    # plot global variables, all the picks of the video
    time_concat = frame_concat*exp_time/60
    
    ## LOCS
    sum_of_locs_of_picked_vs_time = np.sum(locs_of_picked_vs_time, axis=0)
    plt.figure()
    plt.step(bin_centers_minutes, sum_of_locs_of_picked_vs_time, where = 'mid')
    plt.xlabel('Time (min)')
    plt.ylabel('Locs')
    x_limit = [0, total_time_min]
    plt.xlim(x_limit)
    ax = plt.gca()
    ax.set_title('Sum of localizations vs time. Binning time %d s. %d picks. ' \
                 % ((bin_size*0.1), total_number_of_picks))
    figure_name = 'locs_vs_time_all'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()

    # ================ PROCESS TIME DATA ================
    # Sorting time.
    index_concat = np.argsort(time_concat)
    ordered_time_concat = time_concat[index_concat]

    # Print the sum of the photons.
    photons_sum = np.sum(photons_concat, axis=None)
    photons_mean = np.mean(photons_concat, axis=None)

    # ================ SAVE KINETICS DATA ================
    new_filename = 'PHOTONS.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, photons_concat)

    new_filename = 'TIME.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, time_concat)

    # ================ PLOT BACKGROUND SIGNAL ================
    ## BACKGROUND
    ax = plot_vs_time_with_hist(bkg_concat, time_concat, order = 2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Total background [photons]')
    ax.set_title('Total background received vs time.')
    figure_name = 'bkg_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()

    # ================ PLOT SITE-SPECIFIC PHOTON DISTRIBUTIONS ================
    # Sort the keys to ensure plotting in numerical order
    keys = sorted(all_traces_per_site.keys(), key=float)

    # Custom colors and line styles for better distinction
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'black', 'gray']
    line_styles = ['-']*10

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    try:
        global_min = min([data.min() for data in all_traces_per_site.values()])
        global_max = max([data.max() for data in all_traces_per_site.values()])

        x_lim_lower = global_min - 1
        x_lim_upper = global_max + 1

        for key, color, line_style in zip(keys, colors, line_styles):
            data = all_traces_per_site[key]
            bins = int(np.ceil(np.sqrt(len(data))))
            print(key, len(data))
            ax.hist(data, bins=bins, label=f'{key}', histtype='step', density=False, color=color, linestyle=line_style,
                    linewidth=1, alpha=0.6)

        ax.set_xlim(x_lim_lower, x_lim_upper)
        ax.set_xlabel("Photons", fontsize=24)
        ax.set_ylabel("Counts", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(title='Binding site')
        ax.set_title("Binding site photon distributions", fontsize=24)
        plt.tight_layout()
        figure_name = 'binding_site_photon_distributions'
        figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass

    # ================ KINETICS ANALYSIS AND PLOTTING ================
    # Process and plot kinetics data collected from all picks
    if len(tons_all) > 0:
        # Clean up kinetics data
        tons_all = np.trim_zeros(tons_all)
        toffs_all = np.trim_zeros(toffs_all)
        
        # Filter out invalid SNR/SBR values
        filter_indices = np.logical_and.reduce((
            ~np.isnan(SNR_all), ~np.isnan(SBR_all),
            ~np.isinf(SNR_all), ~np.isinf(SBR_all)
        ))
        SNR_filtered = SNR_all[filter_indices]
        SBR_filtered = SBR_all[filter_indices]
        tstarts_filtered = tstarts_all[filter_indices]
        
        # ================ SAVE KINETICS DATA ================
        # Save kinetics data files for Step 4 compatibility
        np.savetxt(os.path.join(kinetics_folder, 't_on.dat'), tons_all, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 't_off.dat'), toffs_all, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 't_start.dat'), tstarts_all, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 'snr.dat'), SNR_filtered, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 'sbr.dat'), SBR_filtered, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 'sum_photons.dat'), sum_photons_all, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 'avg_photons.dat'), avg_photons_all, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 'std_photons.dat'), std_photons_all, fmt='%.3f')
        np.savetxt(os.path.join(kinetics_folder, 'double_events.dat'), double_events_all, fmt='%.3f')
        
        # ================ KINETICS PLOTS ================
        # Plot binding time vs start time
        plt.figure()
        ax, slope, intercept = plot_vs_time_with_hist(tons_all, tstarts_all/60, order=1, fit_line=True)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Binding time [s]')
        ax.set_title(f'Binding time vs. start time\nSlope: {slope:.3f}, Intercept: {intercept:.3f}')
        figure_path = os.path.join(figures_folder, 'binding_time_vs_time.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot total photons vs time
        plt.figure()
        ax = plot_vs_time_with_hist(sum_photons_all, tstarts_all/60)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Total photons [photons]')
        ax.set_title('Total photons per binding event vs time')
        figure_path = os.path.join(figures_folder, 'total_photons_vs_time.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Photon count histogram
        plt.figure()
        bin_edges = np.histogram_bin_edges(sum_photons_all, 'fd')
        plt.hist(sum_photons_all, bins=bin_edges)
        plt.xlabel('Photon count [photons]')
        plt.ylabel('Frequency')
        plt.title('Histogram of photon count per binding event')
        plt.yscale('log')
        figure_path = os.path.join(figures_folder, 'photon_histogram.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Standard deviation histogram
        if len(std_photons_all) > 0:
            plt.figure()
            bin_edges = np.histogram_bin_edges(std_photons_all, 'fd')
            plt.hist(std_photons_all, bins=bin_edges)
            plt.xlabel('Standard deviation [photons]')
            plt.ylabel('Frequency')
            plt.title('Histogram of photon standard deviation per binding event')
            plt.yscale('log')
            figure_path = os.path.join(figures_folder, 'std_photons_histogram.png')
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # SNR and SBR scatter plot
        if len(SNR_filtered) > 0:
            plt.figure()
            plt.scatter(tstarts_filtered/60, SNR_filtered, s=0.85, alpha=0.6, label='SNR')
            plt.scatter(tstarts_filtered/60, SBR_filtered, s=0.85, alpha=0.6, label='SBR')
            plt.yscale('log')
            plt.xlabel('Time [min]')
            plt.ylabel('SNR / SBR')
            plt.title('Signal-to-Noise and Signal-to-Background Ratios vs Time')
            plt.legend()
            figure_path = os.path.join(figures_folder, 'SNR_SBR_scatter_plot.png')
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Update summary with kinetics data
        kinetics_summary = {
            'Total Binding Events': len(tons_all),
            'Mean Binding Time (s)': np.mean(tons_all),
            'Mean Unbinding Time (s)': np.mean(toffs_all) if len(toffs_all) > 0 else 0,
            'Mean SNR': np.mean(SNR_filtered) if len(SNR_filtered) > 0 else 0,
            'Mean SBR': np.mean(SBR_filtered) if len(SBR_filtered) > 0 else 0,
            'Binding Time Slope': slope if 'slope' in locals() else 0,
            'Binding Time Intercept': intercept if 'intercept' in locals() else 0
        }
        
        for key, value in kinetics_summary.items():
            update_pkl(main_folder, key, value)

    # ================ SAVE BACKGROUND DATA ================
    new_filename = 'BKG.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, bkg_concat, fmt='%05d')
    
    # ================ SAVE ALL TRACES ================
    # delete first fake and empty trace (needed to make the proper array)
    all_traces = np.delete(all_traces, 0, axis = 0)
    all_traces = all_traces.T
    # save ALL traces in one file

    new_filename = 'TRACES_ALL.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, all_traces, fmt='%05d')

    # ================ SAVE ADDITIONAL DATA ================
    # number of locs
    data_to_save = np.asarray([pick_number, locs_of_picked]).T
    new_filename = 'number_of_locs_per_pick.dat'
    new_filepath = os.path.join(data_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%i')
    
    # relative positions
    data_to_save = positions_concat_NP
    new_filename = 'relative_positions_NP_sites_in_nm.dat'
    new_filepath = os.path.join(data_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%.1f')
    data_to_save = positions_concat_origami
    new_filename = 'relative_positions_binding_sites_in_nm.dat'
    new_filepath = os.path.join(data_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%.1f')

    # ================ OUTPUT SUMMARY INFORMATION ================
    # Summary information
    summary_info = {
        'Total Picks Processed': total_number_of_picks,
        'Total Time (min)': total_time_min,
        '2D Histogram Bin Size (nm)': hist_2D_bin_size,
        'Mean Amount of Photons': photons_mean,
        'Mean Background Signal': np.mean(bkg_concat, axis=None),
    }

    for key in summary_info.keys():
        update_pkl(main_folder, key, summary_info[key])

    # Elegant printing of the summary
    print('\n' + '='*23 + '📊 STEP 2 SUMMARY 📊' + '='*23)
    print(f'   Total Picks Processed: {total_number_of_picks}')
    print(f'   Total Time (min): {total_time_min:.1f}')
    print(f'   2D Histogram Bin Size (nm): {hist_2D_bin_size:.2f}')
    print(f'   Mean Amount of Photons: {photons_mean:.1f}')
    print(f'   Mean Background Signal: {np.mean(bkg_concat, axis=None):.1f}')
    if len(tons_all) > 0:
        print(f'   Total Binding Events: {len(tons_all)}')
        print(f'   Mean Binding Time (s): {np.mean(tons_all):.2f}')
        print(f'   Mean SNR: {np.mean(SNR_filtered) if len(SNR_filtered) > 0 else 0:.1f}')
        print(f'   Mean SBR: {np.mean(SBR_filtered) if len(SBR_filtered) > 0 else 0:.1f}')
    print(f'   Data analysis and plotting completed.')
    print('='*70)
    print('\nDone with STEP 2.')

    # ================ RETURN RESULTS FOR CONSOLIDATION ================
    results = {
        'total_picks_processed': total_number_of_picks,
        'total_time_minutes': total_time_min,
        'histogram_bin_size_nm': hist_2D_bin_size,
        'mean_photons': photons_mean,
        'mean_background': np.mean(bkg_concat, axis=None),
        'total_localizations': len(photons_concat),
        'mean_locs_per_pick': np.mean(locs_of_picked),
        'std_locs_per_pick': np.std(locs_of_picked)
    }
    
    # Add kinetics data if available
    if len(tons_all) > 0:
        results.update({
            'total_binding_events': len(tons_all),
            'mean_binding_time_seconds': np.mean(tons_all),
            'std_binding_time_seconds': np.std(tons_all),
            'mean_unbinding_time_seconds': np.mean(toffs_all) if len(toffs_all) > 0 else 0,
            'std_unbinding_time_seconds': np.std(toffs_all) if len(toffs_all) > 0 else 0,
            'mean_snr': np.mean(SNR_filtered) if len(SNR_filtered) > 0 else 0,
            'mean_sbr': np.mean(SBR_filtered) if len(SBR_filtered) > 0 else 0,
            'binding_time_slope': slope if 'slope' in locals() else 0,
            'binding_time_intercept': intercept if 'intercept' in locals() else 0
        })
    
    # Add relative positions statistics if available
    if len(positions_concat_origami) > 0:
        results.update({
            'mean_binding_site_distance_nm': np.mean(positions_concat_origami),
            'std_binding_site_distance_nm': np.std(positions_concat_origami)
        })
    
    if NP_flag and len(positions_concat_NP) > 0:
        results.update({
            'mean_np_distance_nm': np.mean(positions_concat_NP),
            'std_np_distance_nm': np.std(positions_concat_NP)
        })
    
    return results
        
#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':
    
    # load and open folder and file
    base_folder = "C:\\Users\\olled\\Documents\\DNA-PAINT\\Data\\single_channel_DNA-PAINT_example\\Week_4\\All_DNA_Origami\\17_picks"
    root = tk.Tk()
    selected_file = fd.askopenfilename(initialdir = base_folder,
                                          filetypes=(("", "*.dat") , ("", "*.")))   
    root.withdraw()
    working_folder = os.path.dirname(selected_file)
    
    # docking site per origami
    docking_sites = 3
    # is there any NP (hybridized structure)
    NP_flag = False
    # camera pixel size
    pixel_size = 0.130 # in um
    # size of the pick used in picasso
    pick_size = 3 # in camera pixels (put the same number used in Picasso)
    # size of the pick to include locs around the detected peaks
    radius_of_pick_to_average = 0.25 # in camera pixel size
    # set an intensity threshold to avoid dumb peak detection in the background
    # this threshold is arbitrary, don't worry about this parameter, the code 
    # change it automatically to detect the number of docking sites set above
    th = 1
    # time parameters☺
    number_of_frames = 12000
    exp_time = 0.1 # in s
    plot_flag = True
    
    process_dat_files(number_of_frames, exp_time, working_folder,
                          docking_sites, NP_flag, pixel_size, pick_size, 
                          radius_of_pick_to_average, th, plot_flag, True)

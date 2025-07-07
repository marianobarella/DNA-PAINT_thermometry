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
from sklearn.cluster import DBSCAN
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
                      radius_of_pick_to_average, th, plot_flag, verbose_flag, photons_threshold, mask_level, mask_singles, use_dbscan=False):
    
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

    # Arrays for collecting kinetics data per site (using docking_sites parameter)
    tons_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    toffs_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    tstarts_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    SNR_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    SBR_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    sum_photons_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    avg_photons_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    photon_intensity_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    std_photons_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    double_events_per_site = {str(site): np.array([]) for site in range(docking_sites)}
    
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

        background_level = np.mean(bkg_of_picked)

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
        
        # ================ IMPROVED PEAK DETECTION OR DBSCAN CLUSTERING ================
        min_distance_nm = 15  # Minimum 15nm separation between peaks
        
        # Calculate analysis radius early for use in both DBSCAN and traditional methods
        analysis_radius = radius_of_pick_to_average*pixel_size
        
        if use_dbscan:
            # ================ DBSCAN CLUSTERING ================
            # Convert positions to array for DBSCAN
            positions = np.column_stack((x_position_of_picked, y_position_of_picked))
            
            # DBSCAN parameters - use GUI-relevant values for better adaptation
            # Use analysis radius as base for clustering, but make it slightly smaller for tight clustering
            analysis_radius_um = radius_of_pick_to_average * pixel_size  # Convert to micrometers
            eps_um = analysis_radius_um * 0.6  # 60% of analysis radius for tighter clustering
            
            # Adaptive min_samples based on expected density and docking sites
            expected_points_per_site = max(3, len(positions) // (docking_sites * 2))
            min_samples = max(2, min(expected_points_per_site // 2, 5))  # Between 2-5 points
            
            if verbose_flag:
                print(f'DBSCAN parameters: eps={eps_um*1000:.1f}nm, min_samples={min_samples}')
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps_um, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(positions)
            
            # Get cluster centers (excluding noise points labeled as -1)
            peak_coords = []
            unique_labels = np.unique(cluster_labels)
            valid_labels = unique_labels[unique_labels != -1]  # Remove noise label
            
            for label in valid_labels:
                cluster_mask = cluster_labels == label
                cluster_points = positions[cluster_mask]
                # Use centroid as peak coordinate
                centroid_x = np.mean(cluster_points[:, 0])
                centroid_y = np.mean(cluster_points[:, 1])
                peak_coords.append((centroid_x, centroid_y))
            
            # Sort by cluster size (largest first) to prioritize main clusters
            if len(peak_coords) > docking_sites:
                cluster_sizes = []
                for label in valid_labels:
                    cluster_mask = cluster_labels == label
                    cluster_sizes.append(np.sum(cluster_mask))
                
                # Get indices sorted by cluster size (descending)
                sorted_indices = np.argsort(cluster_sizes)[::-1]
                peak_coords = [peak_coords[i] for i in sorted_indices[:docking_sites]]
            
            if verbose_flag:
                print(f'DBSCAN found {len(valid_labels)} clusters, using top {len(peak_coords)} peaks')
                
            # Convert peak_coords to arrays for consistency with original code
            cm_binding_sites_x = np.array([peak[0] for peak in peak_coords])
            cm_binding_sites_y = np.array([peak[1] for peak in peak_coords])
            
            # Calculate standard deviations for each DBSCAN cluster
            cm_std_dev_binding_sites_x = np.array([])
            cm_std_dev_binding_sites_y = np.array([])
            
            for idx, label in enumerate(valid_labels[:len(peak_coords)]):
                cluster_mask = cluster_labels == label
                cluster_points = positions[cluster_mask]
                std_x = np.std(cluster_points[:, 0], ddof=1) if len(cluster_points) > 1 else 0.01
                std_y = np.std(cluster_points[:, 1], ddof=1) if len(cluster_points) > 1 else 0.01
                cm_std_dev_binding_sites_x = np.append(cm_std_dev_binding_sites_x, std_x)
                cm_std_dev_binding_sites_y = np.append(cm_std_dev_binding_sites_y, std_y)
            
            # ================ DBSCAN VISUALIZATION ================
            if plot_flag:
                # Create DBSCAN clustering visualization
                plt.figure(figsize=(10, 8))
                
                # Create a colormap for clusters
                colors = plt.cm.tab10(np.linspace(0, 1, len(valid_labels)))
                
                # Plot each cluster with different colors
                for idx, label in enumerate(valid_labels):
                    cluster_mask = cluster_labels == label
                    cluster_points = positions[cluster_mask]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              c=[colors[idx]], label=f'Cluster {label}', alpha=0.7, s=20)
                
                # Plot noise points in black
                noise_mask = cluster_labels == -1
                if np.any(noise_mask):
                    noise_points = positions[noise_mask]
                    plt.scatter(noise_points[:, 0], noise_points[:, 1], 
                              c='black', label='Noise', alpha=0.5, s=10, marker='.')
                
                # Plot cluster centers as large stars
                for idx, (center_x, center_y) in enumerate(peak_coords):
                    plt.scatter(center_x, center_y, c='red', marker='*', 
                              s=200, edgecolors='white', linewidth=2, 
                              label='Centers' if idx == 0 else "", zorder=10)
                
                # Add analysis radius circles around centers
                for center_x, center_y in peak_coords:
                    circle = plt.Circle((center_x, center_y), analysis_radius, 
                                      fill=False, color='red', linestyle='--', alpha=0.8)
                    plt.gca().add_patch(circle)
                
                plt.xlabel('X position (μm)')
                plt.ylabel('Y position (μm)')
                plt.title(f'DBSCAN - Pick {i:02d}\n'
                         f'eps={eps_um:.3f}μm, min_samples={min_samples}, '
                         f'{len(valid_labels)} clusters found')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.tight_layout()
                
                # Save the plot
                dbscan_folder = manage_save_directory(figures_per_pick_folder, 'dbscan_clustering')
                figure_name = f'dbscan_clustering_pick_{i:02d}'
                figure_path = os.path.join(dbscan_folder, f'{figure_name}.png')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                if verbose_flag:
                    print(f'DBSCAN plot saved: {figure_path}')
        else:
            # ================ ORIGINAL IMPROVED PEAK DETECTION ================
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
        # Initialize arrays for binding sites (only if not already set by DBSCAN)
        if not use_dbscan:
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
            
            # ================ ASSIGN BINDING EVENTS TO SITES ================
            # Now assign the already-calculated binding events to sites based on averaged positions
            if tau_results[0] is not False and len(tau_results) >= 14:
                # Extract binding event data
                avg_x_positions = tau_results[12]  # average_x_positions
                avg_y_positions = tau_results[13]  # average_y_positions
                event_tons = tau_results[0]       # on times
                event_toffs = tau_results[1]      # off times
                event_tstarts = tau_results[3]    # start times
                event_SNR = tau_results[4]        # signal to noise ratio
                event_SBR = tau_results[5]        # signal to background ratio
                event_sum_photons = tau_results[6]      # sum photons per event
                event_avg_photons = tau_results[7]      # average photons per event
                event_photon_intensity = tau_results[8] # photon intensities
                event_std_photons = tau_results[9]      # std photons per event
                event_double_events = tau_results[11]   # double events count
                
                # Filter out NaN positions
                valid_mask = ~(np.isnan(avg_x_positions) | np.isnan(avg_y_positions))
                
                if np.any(valid_mask):
                    valid_x = avg_x_positions[valid_mask]
                    valid_y = avg_y_positions[valid_mask]
                    
                    # For kinetics arrays, check if they have the same length as position arrays
                    # If not, take only the first N elements where N = number of valid positions
                    n_valid_positions = len(valid_x)
                    
                    # Safely get kinetics data - either filter by valid_mask or take first n_valid_positions
                    if len(event_tons) == len(avg_x_positions):
                        valid_tons = event_tons[valid_mask]
                    else:
                        valid_tons = event_tons[:n_valid_positions] if len(event_tons) >= n_valid_positions else event_tons
                    
                    if len(event_toffs) == len(avg_x_positions):
                        valid_toffs = event_toffs[valid_mask]
                    else:
                        valid_toffs = event_toffs[:n_valid_positions] if len(event_toffs) >= n_valid_positions else event_toffs
                    
                    if len(event_tstarts) == len(avg_x_positions):
                        valid_tstarts = event_tstarts[valid_mask]
                    else:
                        valid_tstarts = event_tstarts[:n_valid_positions] if len(event_tstarts) >= n_valid_positions else event_tstarts
                    
                    if len(event_SNR) == len(avg_x_positions):
                        valid_SNR = event_SNR[valid_mask]
                    else:
                        valid_SNR = event_SNR[:n_valid_positions] if len(event_SNR) >= n_valid_positions else event_SNR
                    
                    if len(event_SBR) == len(avg_x_positions):
                        valid_SBR = event_SBR[valid_mask]
                    else:
                        valid_SBR = event_SBR[:n_valid_positions] if len(event_SBR) >= n_valid_positions else event_SBR
                    
                    if len(event_sum_photons) == len(avg_x_positions):
                        valid_sum_photons = event_sum_photons[valid_mask]
                    else:
                        valid_sum_photons = event_sum_photons[:n_valid_positions] if len(event_sum_photons) >= n_valid_positions else event_sum_photons
                    
                    if len(event_avg_photons) == len(avg_x_positions):
                        valid_avg_photons = event_avg_photons[valid_mask]
                    else:
                        valid_avg_photons = event_avg_photons[:n_valid_positions] if len(event_avg_photons) >= n_valid_positions else event_avg_photons
                    
                    if len(event_photon_intensity) == len(avg_x_positions):
                        valid_photon_intensity = event_photon_intensity[valid_mask]
                    else:
                        valid_photon_intensity = event_photon_intensity[:n_valid_positions] if len(event_photon_intensity) >= n_valid_positions else event_photon_intensity
                    
                    if len(event_std_photons) == len(avg_x_positions):
                        valid_std_photons = event_std_photons[valid_mask]
                    else:
                        valid_std_photons = event_std_photons[:n_valid_positions] if len(event_std_photons) >= n_valid_positions else event_std_photons
                    
                    if len(event_double_events) == len(avg_x_positions):
                        valid_double_events = event_double_events[valid_mask]
                    else:
                        valid_double_events = event_double_events[:n_valid_positions] if len(event_double_events) >= n_valid_positions else event_double_events
                    
                    # Ensure all arrays have the same length (take minimum to be safe)
                    min_length = min(len(valid_x), len(valid_tons), len(valid_toffs), len(valid_tstarts), 
                                   len(valid_SNR), len(valid_SBR), len(valid_sum_photons), 
                                   len(valid_avg_photons), len(valid_photon_intensity), 
                                   len(valid_std_photons), len(valid_double_events))
                    
                    # Truncate all arrays to the minimum length
                    valid_x = valid_x[:min_length]
                    valid_y = valid_y[:min_length]
                    valid_tons = valid_tons[:min_length]
                    valid_toffs = valid_toffs[:min_length]
                    valid_tstarts = valid_tstarts[:min_length]
                    valid_SNR = valid_SNR[:min_length]
                    valid_SBR = valid_SBR[:min_length]
                    valid_sum_photons = valid_sum_photons[:min_length]
                    valid_avg_photons = valid_avg_photons[:min_length]
                    valid_photon_intensity = valid_photon_intensity[:min_length]
                    valid_std_photons = valid_std_photons[:min_length]
                    valid_double_events = valid_double_events[:min_length]
                    
                    # First, add all valid binding events to overall arrays
                    tons_all = np.append(tons_all, valid_tons)
                    toffs_all = np.append(toffs_all, valid_toffs)
                    tstarts_all = np.append(tstarts_all, valid_tstarts)
                    SNR_all = np.append(SNR_all, valid_SNR)
                    SBR_all = np.append(SBR_all, valid_SBR)
                    sum_photons_all = np.append(sum_photons_all, valid_sum_photons)
                    avg_photons_all = np.append(avg_photons_all, valid_avg_photons)
                    photon_intensity_all = np.append(photon_intensity_all, valid_photon_intensity)
                    std_photons_all = np.append(std_photons_all, valid_std_photons)
                    double_events_all = np.append(double_events_all, valid_double_events)
                    
                    # Then assign each binding event to the nearest detected peak
                    for event_idx in range(len(valid_x)):
                        event_x = valid_x[event_idx]
                        event_y = valid_y[event_idx]
                        
                        # Find nearest peak
                        min_distance = float('inf')
                        nearest_site = -1
                        
                        for peak_idx in range(total_peaks_found):
                            peak_x, peak_y = peak_coords[peak_idx]
                            distance = np.sqrt((event_x - peak_x)**2 + (event_y - peak_y)**2)
                            
                            if distance < min_distance and distance < analysis_radius:
                                min_distance = distance
                                nearest_site = peak_idx
                        
                        # Assign event to site if within analysis radius
                        if nearest_site >= 0:
                            site_rank = str(ranks[nearest_site])
                            
                            # Add to per-site arrays
                            if site_rank in tons_per_site:
                                tons_per_site[site_rank] = np.append(tons_per_site[site_rank], valid_tons[event_idx])
                                toffs_per_site[site_rank] = np.append(toffs_per_site[site_rank], valid_toffs[event_idx])
                                tstarts_per_site[site_rank] = np.append(tstarts_per_site[site_rank], valid_tstarts[event_idx])
                                SNR_per_site[site_rank] = np.append(SNR_per_site[site_rank], valid_SNR[event_idx])
                                SBR_per_site[site_rank] = np.append(SBR_per_site[site_rank], valid_SBR[event_idx])
                                sum_photons_per_site[site_rank] = np.append(sum_photons_per_site[site_rank], valid_sum_photons[event_idx])
                                avg_photons_per_site[site_rank] = np.append(avg_photons_per_site[site_rank], valid_avg_photons[event_idx])
                                photon_intensity_per_site[site_rank] = np.append(photon_intensity_per_site[site_rank], valid_photon_intensity[event_idx])
                                std_photons_per_site[site_rank] = np.append(std_photons_per_site[site_rank], valid_std_photons[event_idx])
                                double_events_per_site[site_rank] = np.append(double_events_per_site[site_rank], valid_double_events[event_idx])
    
    # ================ SAVE OVERALL KINETICS DATA TO FILES ================
    # After all picks are processed, save the overall kinetics data first
    print(f"\nSaving overall kinetics data...")
    
    # Create method-specific folder structure
    if use_dbscan:
        dbscan_kinetics_folder = manage_save_directory(kinetics_folder, 'dbscan_data')
        save_kinetics_folder = dbscan_kinetics_folder
        method_suffix = ""  # No suffix needed since we have separate folder
    else:
        save_kinetics_folder = kinetics_folder
        method_suffix = ""
    
    # Save overall t_on and t_off data
    if len(tons_all) > 0:
        ton_filepath = os.path.join(save_kinetics_folder, f't_on{method_suffix}.dat')
        np.savetxt(ton_filepath, tons_all, fmt='%.3f')
        
        toff_filepath = os.path.join(save_kinetics_folder, f't_off{method_suffix}.dat')
        np.savetxt(toff_filepath, toffs_all, fmt='%.3f')
        
        # Save other overall kinetics parameters
        tstarts_filepath = os.path.join(save_kinetics_folder, f't_starts{method_suffix}.dat')
        np.savetxt(tstarts_filepath, tstarts_all, fmt='%.3f')
        
        SNR_filepath = os.path.join(save_kinetics_folder, f'SNR{method_suffix}.dat')
        np.savetxt(SNR_filepath, SNR_all, fmt='%.3f')
        
        SBR_filepath = os.path.join(save_kinetics_folder, f'SBR{method_suffix}.dat')
        np.savetxt(SBR_filepath, SBR_all, fmt='%.3f')
        
        sum_photons_filepath = os.path.join(save_kinetics_folder, f'sum_photons{method_suffix}.dat')
        np.savetxt(sum_photons_filepath, sum_photons_all, fmt='%.3f')
        
        avg_photons_filepath = os.path.join(save_kinetics_folder, f'avg_photons{method_suffix}.dat')
        np.savetxt(avg_photons_filepath, avg_photons_all, fmt='%.3f')
        
        print(f"Overall kinetics data saved: {len(tons_all):,} events")
        print(f"   t_on.dat: {ton_filepath}")
        print(f"   t_off.dat: {toff_filepath}")
    else:
        print("Warning: No overall kinetics data to save!")
    
    # ================ SAVE PER-SITE KINETICS DATA TO FILES ================
    # After all picks are processed, save the collected per-site data
    print(f"\nSaving per-site kinetics data...")
    
    # Create per-site combined folder in appropriate kinetics folder
    per_site_combined_folder = manage_save_directory(save_kinetics_folder, 'per_site_combined')
    
    # Save data for each site
    for site_rank in tons_per_site.keys():
        if len(tons_per_site[site_rank]) > 0:
            # Save t_on data
            ton_filename = f't_on_site_{site_rank}.dat'
            ton_filepath = os.path.join(per_site_combined_folder, ton_filename)
            np.savetxt(ton_filepath, tons_per_site[site_rank], fmt='%.3f')
            
            # Save t_off data  
            toff_filename = f't_off_site_{site_rank}.dat'
            toff_filepath = os.path.join(per_site_combined_folder, toff_filename)
            np.savetxt(toff_filepath, toffs_per_site[site_rank], fmt='%.3f')
            
            # Save other kinetics parameters
            tstarts_filename = f't_starts_site_{site_rank}.dat'
            tstarts_filepath = os.path.join(per_site_combined_folder, tstarts_filename)
            np.savetxt(tstarts_filepath, tstarts_per_site[site_rank], fmt='%.3f')
            
            SNR_filename = f'SNR_site_{site_rank}.dat'
            SNR_filepath = os.path.join(per_site_combined_folder, SNR_filename)
            np.savetxt(SNR_filepath, SNR_per_site[site_rank], fmt='%.3f')
            
            SBR_filename = f'SBR_site_{site_rank}.dat'
            SBR_filepath = os.path.join(per_site_combined_folder, SBR_filename)
            np.savetxt(SBR_filepath, SBR_per_site[site_rank], fmt='%.3f')
            
            sum_photons_filename = f'sum_photons_site_{site_rank}.dat'
            sum_photons_filepath = os.path.join(per_site_combined_folder, sum_photons_filename)
            np.savetxt(sum_photons_filepath, sum_photons_per_site[site_rank], fmt='%.3f')
            
            avg_photons_filename = f'avg_photons_site_{site_rank}.dat'
            avg_photons_filepath = os.path.join(per_site_combined_folder, avg_photons_filename)
            np.savetxt(avg_photons_filepath, avg_photons_per_site[site_rank], fmt='%.3f')
            
            if verbose_flag:
                print(f"   Site {site_rank}: {len(tons_per_site[site_rank]):,} events saved")
        else:
            if verbose_flag:
                print(f"   Site {site_rank}: No events to save (empty)")
    
    print(f"Per-site data saved to: {per_site_combined_folder}")
    
    plt.close()
        
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
    
    # Add missing parameters for position averaging
    photons_threshold = 20  # Photon threshold for binding events
    mask_level = 3          # Mask level for noise filtering
    mask_singles = True     # Mask single-frame events
    
    process_dat_files(number_of_frames, exp_time, working_folder,
                          docking_sites, NP_flag, pixel_size, pick_size, 
                          radius_of_pick_to_average, th, plot_flag, True, 
                          photons_threshold, mask_level, mask_singles)

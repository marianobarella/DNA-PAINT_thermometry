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
from auxiliary_functions import detect_peaks, distance, fit_linear, \
    perpendicular_distance, manage_save_directory, plot_vs_time_with_hist, update_pkl
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
    method_subfolder = 'original_method'  # This is the original method
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
    try:
        frame = np.loadtxt(frame_filepath, encoding='utf-8')
    except (UnicodeDecodeError, TypeError):
        # Fallback for encoding issues or older numpy versions
        with open(frame_filepath, 'r', encoding='utf-8') as f:
            frame = np.loadtxt(f)
    
    # photons
    photons_file = [f for f in list_of_files_origami if re.search('_photons',f)][0]
    photons_filepath = os.path.join(working_folder, photons_file)
    try:
        photons = np.loadtxt(photons_filepath, encoding='utf-8')
    except (UnicodeDecodeError, TypeError):
        with open(photons_filepath, 'r', encoding='utf-8') as f:
            photons = np.loadtxt(f)
    if NP_flag:
        photons_file_NP = [f for f in list_of_files_NP if re.search('_photons', f)][0]
        photons_filepath_NP = os.path.join(working_folder, photons_file_NP)
        try:
            photons = np.loadtxt(photons_filepath_NP, encoding='utf-8')
        except (UnicodeDecodeError, TypeError):
            with open(photons_filepath_NP, 'r', encoding='utf-8') as f:
                photons = np.loadtxt(f)

    
    # bkg
    bkg_file = [f for f in list_of_files_origami if re.search('_bkg',f)][0]
    bkg_filepath = os.path.join(working_folder, bkg_file)
    try:
        bkg = np.loadtxt(bkg_filepath, encoding='utf-8')
    except (UnicodeDecodeError, TypeError):
        with open(bkg_filepath, 'r', encoding='utf-8') as f:
            bkg = np.loadtxt(f)
    
    # xy positions
    # origami
    position_file = [f for f in list_of_files_origami if re.search('_xy',f)][0]
    position_filepath = os.path.join(working_folder, position_file)
    try:
        position = np.loadtxt(position_filepath, encoding='utf-8')
    except (UnicodeDecodeError, TypeError):
        with open(position_filepath, 'r', encoding='utf-8') as f:
            position = np.loadtxt(f)
    x = position[:,0]*pixel_size
    y = position[:,1]*pixel_size
    # NP
    if NP_flag:
        position_file_NP = [f for f in list_of_files_NP if re.search('_xy',f)][0]
        position_filepath_NP = os.path.join(working_folder, position_file_NP)
        try:
            xy_NP = np.loadtxt(position_filepath_NP, encoding='utf-8')
        except (UnicodeDecodeError, TypeError):
            with open(position_filepath_NP, 'r', encoding='utf-8') as f:
                xy_NP = np.loadtxt(f)
        x_NP = xy_NP[:,0]*pixel_size
        y_NP = xy_NP[:,1]*pixel_size
    
    # number of pick
    # origami
    pick_file = [f for f in list_of_files_origami if re.search('_pick_number',f)][0]
    pick_filepath = os.path.join(working_folder, pick_file)
    try:
        pick_list = np.loadtxt(pick_filepath, encoding='utf-8')
    except (UnicodeDecodeError, TypeError):
        with open(pick_filepath, 'r', encoding='utf-8') as f:
            pick_list = np.loadtxt(f)
    # NP
    if NP_flag:
        pick_file_NP = [f for f in list_of_files_NP if re.search('_pick_number',f)][0]
        pick_filepath_NP = os.path.join(working_folder, pick_file_NP)
        try:
            pick_list_NP = np.loadtxt(pick_filepath_NP, encoding='utf-8')
        except (UnicodeDecodeError, TypeError):
            with open(pick_filepath_NP, 'r', encoding='utf-8') as f:
                pick_list_NP = np.loadtxt(f)
    
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
    all_traces = np.zeros((1, number_of_frames))  # Initialize as 2D array for vstack compatibility
    all_traces_per_site = {}
    
    # ================ HISTOGRAM CONFIGURATION ================
    # set number of bins for FINE histograming 
    N = int(0.7 * 2*pick_size*pixel_size*1000/10)
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
        x_position_of_picked = x[index_picked]
        y_position_of_picked = y[index_picked]
        
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
        
        # ================ PEAK DETECTION INITIALIZATION ================
        # Initialize variables for peak detection
        total_peaks_found = 0
        threshold_COARSE = th
        bins_COARSE = 1 if docking_sites == 1 else 20
        docking_sites_temp = docking_sites
        site_goal = docking_sites
        z_hist_COARSE = None
        
        # ================ ADAPTIVE PEAK DETECTION LOOP ================
        while total_peaks_found != site_goal:
            if docking_sites_temp == docking_sites - 1 or total_peaks_found == docking_sites - 1:
                docking_sites_temp = docking_sites - 1
                if total_peaks_found == docking_sites - 1:
                    break
            
            # Make COARSE 2D histogram - only once per iteration
            z_hist_COARSE, x_hist_COARSE, y_hist_COARSE = np.histogram2d(
                x_position_of_picked, y_position_of_picked, 
                bins=bins_COARSE, range=hist_bounds, density=True
            )
            z_hist_COARSE = z_hist_COARSE.T
            z_hist_COARSE = np.where(z_hist_COARSE < threshold_COARSE, 0, z_hist_COARSE)
            
            # Peak detection
            detected_peaks = detect_peaks(z_hist_COARSE)
            index_peaks = np.where(detected_peaks == True)
            total_peaks_found = len(index_peaks[0])
            
            threshold_COARSE += 5
            if threshold_COARSE > 5000:
                break
                
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
            x_hist_COARSE_centers = x_hist_COARSE[:-1] + np.diff(x_hist_COARSE)/2
            y_hist_COARSE_centers = y_hist_COARSE[:-1] + np.diff(y_hist_COARSE)/2
            
            # Pre-calculate coordinates of all peaks
            peak_coords = [(x_hist_COARSE_centers[index_peaks[1][j]], 
                           y_hist_COARSE_centers[index_peaks[0][j]]) 
                          for j in range(total_peaks_found)]
            
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
                dist_value = 0 if not NP_flag else round(matrix_distance[0, index+1], 2)
                new_filename = f'TRACE_pick_{i}_site_{int(key)}_dist_{dist_value}.dat'
                new_filepath = os.path.join(traces_per_site_folder, new_filename)
                np.savetxt(new_filepath, site_trace, fmt='%05d')
        
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
                
                if peaks_flag and len(cm_binding_sites_x) > 1:
                    plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize=9, 
                            color='white', label='binding sites')
                    plt.plot(x_fitted, y_fitted, '--', linewidth=1, color='white')
                    
                    for k, (circle_x, circle_y) in enumerate(zip(cm_binding_sites_x, cm_binding_sites_y)):
                        circ = plot_circle((circle_x, circle_y), radius=analysis_radius, 
                                          color='white', fill=False)
                        ax.add_patch(circ)
                        
                        # Peak labels
                        theta = np.arctan(slope)
                        perpendicular_x = circle_x + analysis_radius*1.25*np.cos(theta+np.pi/2)
                        perpendicular_y = circle_y + -1/slope * (perpendicular_x-circle_x)
                        text_position = (perpendicular_x, perpendicular_y)
                        text_content = f"{ranks[k]}"
                        ax.text(*text_position, text_content, ha='center', va='center', 
                               rotation=0, fontsize=12, color='white')
                
                if NP_flag:
                    plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=8, markerfacecolor='C1', 
                            markeredgecolor='white', label='NP')
                    plt.legend(loc='upper right')
                
                plt.ylabel(r'y ($\mu$m)')
                plt.xlabel(r'x ($\mu$m)')
                cbar = plt.colorbar()
                cbar.ax.set_title(u'Locs', fontsize=16)
                cbar.ax.tick_params(labelsize=16)
                ax.set_title(f'Position of locs per pick. Pick {i:02d}')
                aux_folder = manage_save_directory(figures_per_pick_folder, 'image_FINE')
                figure_name = f'xy_pick_image_PAINT_{i:02d}'
                figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # ================ PLOT FINE 2D IMAGE FOR PAPER ================
                plt.figure(3)
                plt.imshow(z_hist, interpolation='none', origin='lower',
                          extent=[x_hist_centers[0], x_hist_centers[-1], 
                                  y_hist_centers[0], y_hist_centers[-1]])
                
                if peaks_flag and len(cm_binding_sites_x) > 1:
                    plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize=5,
                            color='white', mew=2, label='Binding Sites', alpha=0.65)
                    ax = plt.gca()
                    ax.set_facecolor(bkg_color)
                    
                    for k, (circle_x, circle_y) in enumerate(zip(cm_binding_sites_x, cm_binding_sites_y)):
                        circ = plot_circle((circle_x, circle_y), radius=analysis_radius, 
                                          color='white', fill=False, linewidth=1, alpha=0.65)
                        ax.add_patch(circ)
                        
                        # Peak labels
                        theta = np.arctan(-abs(slope))
                        perpendicular_x = circle_x + analysis_radius*1.25*np.cos(theta+np.pi/2)
                        perpendicular_y = circle_y + -1/slope * (perpendicular_x-circle_x)
                        text_position = (perpendicular_x, perpendicular_y)
                        text_content = f"{ranks[k]}"
                        ax.text(*text_position, text_content, ha='center', va='center', 
                               rotation=0, fontsize=11, color='white', alpha=0.65)
                
                if NP_flag:
                    plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=8, markerfacecolor='white', 
                            markeredgecolor='black', label='Center of NP')
                    plt.legend(loc='upper left')
                
                scalebar = ScaleBar(1e3, 'nm', location='lower left') 
                ax.add_artist(scalebar)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax = plt.gca()
                ax.set_facecolor(bkg_color)
                cbar = plt.colorbar()
                cbar.ax.set_title(u'Locs')
                cbar.ax.tick_params()
                aux_folder = manage_save_directory(figures_per_pick_folder, 'image_FINE')
                figure_name = f'PAPER_xy_pick_image_PAINT_{i:02d}'
                figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # ================ PLOT COARSE AND BINARY IMAGES ================
                if docking_sites_temp != 1:
                    # COARSE 2d image
                    plt.figure(4)
                    plt.imshow(z_hist_COARSE, interpolation='none', origin='lower',
                              extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                                      y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
                    
                    if NP_flag:
                        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=10, markerfacecolor='white', 
                                markeredgecolor='k', label='NP')
                        plt.legend(loc='upper right')
                    
                    plt.ylabel(r'y ($\mu$m)')
                    plt.xlabel(r'x ($\mu$m)')
                    ax = plt.gca()
                    ax.set_facecolor(bkg_color)
                    cbar = plt.colorbar()
                    cbar.ax.set_title(u'Locs')
                    cbar.ax.tick_params()
                    ax.set_title(f'Position of locs per pick. Pick {i:02d}')
                    aux_folder = manage_save_directory(figures_per_pick_folder, 'image_COARSE')
                    figure_name = f'xy_pick_image_COARSE_PAINT_{i:02d}'
                    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
                    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # BINARY 2d image
                    plt.figure(5)
                    plt.imshow(detected_peaks, interpolation='none', origin='lower', cmap='binary',
                              extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                                      y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
                    
                    if NP_flag:
                        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=10, markerfacecolor='C1', 
                                markeredgecolor='k', label='NP')
                        plt.legend(loc='upper right')
                    
                    plt.ylabel(r'y ($\mu$m)')
                    plt.xlabel(r'x ($\mu$m)')
                    ax = plt.gca()
                    ax.set_title(f'Position of locs per pick. Pick {i:02d}')
                    aux_folder = manage_save_directory(figures_per_pick_folder, 'binary_image')
                    figure_name = f'xy_pick_image_peaks_PAINT_{i:02d}'
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

    # ================ SAVE BACKGROUND DATA ================
    new_filename = 'BKG.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, bkg_concat, fmt='%05d')
    
    # ================ SAVE ALL TRACES ================
    # delete first fake and empty trace (needed to make the proper array)
    if all_traces.shape[0] > 1:  # Only delete if there are actual traces beyond the dummy row
        all_traces = np.delete(all_traces, 0, axis = 0)
        all_traces = all_traces.T
    else:
        # If no binding sites were found, create an empty array with correct shape
        all_traces = np.zeros((number_of_frames, 0))
    
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

# -*- coding: utf-8 -*-
"""
Analysis of single AuNPs temperature profile using DNA-PAINT super-resolution technique

Mariano Barella

5 July 2023

"""

import os
import re
import tkinter as tk
import tkinter.filedialog as fd
import pandas as pd
import json
from datetime import datetime

import numpy as np

# from auxiliary_functions import save_parameters
import step1_extract_and_save_data_from_hdf5_picasso_files as step1
# Step 2 import will be conditional based on averaging flag
import step3_calculate_tonoff_with_mask as step3
import step4_estimate_binding_time_using_MLE as step4
import time
import argparse
from ast import literal_eval

#####################################################################

def save_consolidated_results(results_dict, metadata_dict, working_folder):
    """Save consolidated results to CSV and metadata to JSON"""
    
    # ================ CREATE ANALYSIS FOLDER AND SAVE RESULTS ================
    analysis_folder = os.path.join(working_folder, 'analysis')
    os.makedirs(analysis_folder, exist_ok=True)
    
    # ================ SAVE RESULTS TO CSV (APPEND IF EXISTS) ================
    csv_path = os.path.join(analysis_folder, 'results.csv')
    new_results_df = pd.DataFrame([results_dict])
    
    try:
        if os.path.exists(csv_path):
            # Append to existing CSV
            existing_df = pd.read_csv(csv_path, encoding='utf-8')
            combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f'Results appended to existing CSV (now {len(combined_df)} rows)')
        else:
            # Create new CSV
            new_results_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f'New results CSV created with 1 row')
    except PermissionError:
        print(f'\nERROR: Cannot write to {csv_path}')
        print('   SOLUTION: Close the CSV file if it\'s open in Excel/other programs')
        print('   Results will be printed to console instead:')
        print(f'   {new_results_df.to_string(index=False)}')
    
    # ================ SAVE METADATA TO CSV (APPEND IF EXISTS) ================
    metadata_dict['analysis_timestamp'] = datetime.now().isoformat()
    metadata_csv_path = os.path.join(analysis_folder, 'analysis_metadata.csv')
    
    # Flatten the analysis_parameters dict to individual columns
    if 'analysis_parameters' in metadata_dict:
        params = metadata_dict.pop('analysis_parameters')
        for key, value in params.items():
            metadata_dict[f'param_{key}'] = value
    
    # Convert steps_executed list to string for CSV storage
    if 'steps_executed' in metadata_dict:
        metadata_dict['steps_executed'] = ', '.join(metadata_dict['steps_executed'])
    
    new_metadata_df = pd.DataFrame([metadata_dict])
    
    if os.path.exists(metadata_csv_path):
        # Append to existing metadata CSV
        existing_metadata_df = pd.read_csv(metadata_csv_path, encoding='utf-8')
        combined_metadata_df = pd.concat([existing_metadata_df, new_metadata_df], ignore_index=True)
        combined_metadata_df.to_csv(metadata_csv_path, index=False, encoding='utf-8')
        print(f'Metadata appended to existing CSV (now {len(combined_metadata_df)} rows)')
    else:
        # Create new metadata CSV
        new_metadata_df.to_csv(metadata_csv_path, index=False, encoding='utf-8')
        print(f'New metadata CSV created with 1 row')
    
    print(f'\nResults saved to: {csv_path}')
    print(f'Metadata saved to: {metadata_csv_path}')

def run_analysis(selected_file, working_folder, step, params):

    number_of_frames = params.get('number_of_frames', 0)
    exp_time = params.get('exposure_time', 0)
    docking_sites = params.get('docking_sites', 0)
    NP_flag = params.get('checkNP', 0)
    pixel_size = params.get('pixel_size', 0)*1e-3
    pick_size = params.get('pick_size', 0)
    radius_of_pick_to_average = params.get('size_to_average', 0)
    th = params.get('th', 0)
    plot_flag = params.get('checkPlot', 0)
    photons_threshold = params.get('photon_threshold', 0)
    background_level = params.get('background_level', 0)
    mask_level = params.get('mask_level', 0)
    rango = params.get('range', 0)
    initial_params = params.get('initial_params', {})
    likelihood_err_param = params.get('likelihood_error', 0)
    opt_display_flag = params.get('checkOptimizationDisplay', 0)
    hyper_exponential_flag = params.get('checkHyperExponential', 0)
    recursive_flag = params.get('recursive', 0)
    rectangles_flag = params.get('rectangle', 0)
    verbose_flag = params.get('verbose', 0)
    lpx_filter = params.get('lpx_filter', 0)
    lpy_filter = params.get('lpy_filter', 0)
    mask_singles = params.get('mask_singles', 0)
    use_position_averaging = params.get('use_position_averaging', False)  # New flag for position averaging
    use_dbscan = params.get('use_dbscan', False)  # New flag for DBSCAN clustering
    photon_threshold_flag = False

    # ================ CONDITIONAL STEP2 IMPORT ================
    # Import the appropriate Step 2 module based on averaging flag
    if use_position_averaging:
        import step2_process_picasso_extracted_data_avg as step2
        method_name = "Position Averaging Method"
        workflow_steps = "Step 1 -> Step 2 (with kinetics) -> Step 4 (MLE)"
    else:
        import step2_process_picasso_extracted_data as step2
        method_name = "Original Method"
        workflow_steps = "Step 1 -> Step 2 -> Step 3 -> Step 4"

    # ================ INITIALIZE RESULTS AND METADATA DICTIONARIES ================
    results_dict = {
        'sample_name': os.path.basename(selected_file),
        'working_folder': working_folder
    }
    
    metadata_dict = {
        'analysis_parameters': params,
        'file_path': selected_file,
        'steps_executed': [str(i+1) for i, enabled in enumerate(step) if enabled == 'True'],
        'method_used': method_name,
        'position_averaging_enabled': use_position_averaging
    }

    # ================ ANALYSIS PARAMETERS SUMMARY ================
    print('\n' + '='*18 + ' ANALYSIS PARAMETERS ' + '='*18)
    print(f'   Method: {method_name}')
    print(f'   Workflow: {workflow_steps}')
    print(f'   Exposure Time: {exp_time}s | Frames: {number_of_frames} | Docking Sites: {docking_sites}')
    print(f'   Photon Threshold: {photons_threshold} | Background: {background_level} | Mask Level: {mask_level}')
    print(f'   Pixel Size: {pixel_size*1000:.0f}nm | Pick Size: {pick_size}px | LP Filters: {lpx_filter}/{lpy_filter}')
    print(f'   Position Averaging: {use_position_averaging} | DBSCAN Clustering: {use_dbscan}')
    if hyper_exponential_flag:
        print(f'   Mode: Hyper-Exponential | Range: {rango} | Error: {likelihood_err_param}')
    else:
        print(f'   Mode: Mono-Exponential | Range: {rango} | Error: {likelihood_err_param}')
    steps_enabled = [f'Step {i+1}' for i, enabled in enumerate(step) if enabled == 'True']
    print(f'   Steps: {", ".join(steps_enabled)}')
    print('='*65)

    if step[0] == 'True':
        # ================ RUN STEP 1 ================
        step1_results = step1.split_hdf5(selected_file, working_folder, recursive_flag, rectangles_flag,
                         lpx_filter, lpy_filter, verbose_flag, NP_flag)
        if step1_results:
            results_dict.update({f'step1_{k}': v for k, v in step1_results.items()})
    else:
        print('\nSTEP 1 was not executed.')
        
    #####################################################################
    
    # folder and file management
    step2_working_folder = os.path.join(working_folder, 'analysis', 'step1', 'data')
    if step[1] == 'True':
        # ================ RUN STEP 2 ================
        step2_results = step2.process_dat_files(number_of_frames, exp_time, step2_working_folder, \
                              docking_sites, NP_flag, pixel_size, pick_size, \
                              radius_of_pick_to_average, th, plot_flag, verbose_flag, photons_threshold, mask_level, mask_singles, use_dbscan)
        if step2_results:
            results_dict.update({f'step2_{k}': v for k, v in step2_results.items()})
    else:
        print('\nSTEP 2 was not executed.')
        
    #####################################################################
        
    # folder and file management
    if use_position_averaging:
        # For position averaging method, kinetics data is in step2/position_averaging_method/data/kinetics_data
        step3_working_folder = os.path.join(working_folder, 'analysis', 'step2', 'position_averaging_method', 'data', 'kinetics_data')
        step2_main_folder = os.path.join(working_folder, 'analysis', 'step2', 'position_averaging_method', 'data')
    else:
        # For original method, step3 data should be in step2/original_method/data for input to step3
        step3_working_folder = os.path.join(working_folder, 'analysis', 'step2', 'original_method', 'data')
        step2_main_folder = os.path.join(working_folder, 'analysis', 'step2', 'original_method', 'data')
    if step[2] == 'True':
        if use_position_averaging:
            # ================ SKIP STEP 3 WHEN USING POSITION AVERAGING ================
            print('\nSTEP 3 was skipped (not needed with position averaging method).')
            print('   Kinetics analysis is integrated into Step 2 with position averaging.')
        else:
            # ================ RUN ORIGINAL STEP 3 ================
            # Load photons data for Step 3
            photons_file = os.path.join(step2_main_folder, 'kinetics_data', 'PHOTONS.dat')
            try:
                photons = np.loadtxt(photons_file, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 encoding if UTF-8 fails
                photons = np.loadtxt(photons_file, encoding='latin-1')
            except ValueError:
                # Fallback for older numpy versions that don't support encoding parameter
                with open(photons_file, 'r', encoding='utf-8') as f:
                    photons = np.loadtxt(f)
            
            step3_results = step3.calculate_kinetics(exp_time, photons_threshold, background_level, 
                                                   photons, working_folder, 'TRACES_ALL.dat', 
                                                   mask_level, mask_singles, number_of_frames,
                                                   verbose_flag, photon_threshold_flag)
            if step3_results:
                results_dict.update({f'step3_{k}': v for k, v in step3_results.items()})
    else:
        if use_position_averaging:
            print('\nSTEP 3 was not executed (not needed with position averaging method).')
        else:
            print('\nSTEP 3 was not executed.')
    
    #####################################################################
        
    # folder and file management
    if use_position_averaging:
        # For position averaging method, kinetics data is in step2/position_averaging_method/data/kinetics_data
        step4_working_folder = os.path.join(working_folder, 'analysis', 'step2', 'position_averaging_method', 'data', 'kinetics_data')
        expected_source = "Step 2 (position averaging method)"
    else:
        # For original method, kinetics data should be in step3/data
        step4_working_folder = os.path.join(working_folder, 'analysis', 'step3', 'data')
        expected_source = "Step 3 (original method)"
        
    if step[3] == 'True':
        # ================ RUN STEP 4 ================
        # Check if the required kinetics data folder exists
        if not os.path.exists(step4_working_folder):
            print(f'\nERROR: Step 4 cannot run - kinetics data folder not found: {step4_working_folder}')
            print(f'Make sure {expected_source} completed successfully and generated kinetics data.')
        else:
            try:
                # Pass the kinetics folder to Step 4, but it will create its own analysis/step4 structure
                step4_results = step4.estimate_binding_unbinding_times(exp_time, rango, step4_working_folder,
                                                initial_params, likelihood_err_param,
                                                opt_display_flag, hyper_exponential_flag, verbose_flag, use_dbscan)
                if step4_results:
                    results_dict.update({f'step4_{k}': v for k, v in step4_results.items()})
            except FileNotFoundError as e:
                print(f'\nERROR: Step 4 cannot run - required file not found: {e}')
                print(f'Make sure {expected_source} completed successfully and generated t_on.dat and t_off.dat files.')
            except Exception as e:
                print(f'\nERROR: Step 4 failed with error: {e}')
                print(f'Check that {expected_source} generated valid binding time data.')
    else:
        print('\nSTEP 4 was not executed.')
    
    #####################################################################
    
    # ================ SAVE CONSOLIDATED RESULTS ================
    save_consolidated_results(results_dict, metadata_dict, working_folder)
    
    print('\nProcess done.')
    
    return


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Run analysis on HDF5 file.')
    parser.add_argument('--selected_file', type=str, required=True, help='Path to selected HDF5 file')
    parser.add_argument('--working_folder', type=str, required=True, help='Working folder')
    parser.add_argument('--step', type=str, nargs=4, required=True,
                        help='List of steps to execute (1 or 0)')
    parser.add_argument('--params', type=str, required=True, help='Parameters')

    args = parser.parse_args()
    params = literal_eval(args.params)
    if params.get('recursive', 0):
        for directory in os.listdir(os.path.dirname(args.working_folder)):
            dir = os.path.join(os.path.dirname(args.working_folder), directory)
            files = os.listdir(dir)

            # Filter the files to only those containing '_picked' in their name
            picked_files = [f for f in files if '_picked' in f]

            # Sort or process to ensure consistent selection of the first file
            picked_files.sort()

            # Take the first file from the filtered list, if any
            picked_file = picked_files[0] if picked_files else None
            path_to_picked_file = os.path.join(dir, picked_file)
            try:
                run_analysis(path_to_picked_file, dir, args.step, params)
            except:
                print(f'An error occurred with {directory}')
                pass
    else:
        run_analysis(args.selected_file, args.working_folder, args.step, params)
    print("Total execution time: --- %s seconds ---" % (time.time() - start_time))

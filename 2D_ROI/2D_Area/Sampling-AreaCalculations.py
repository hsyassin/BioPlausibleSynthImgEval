import pandas as pd
import numpy as np
import nibabel as nib
import os
import re

def dataframe(data, Real):

    # Combine left and right hippocampus volumes
    data['lateral_ventricle_volume'] = data['left_lateral_ventricle'] + data['right_lateral_ventricle']
    data['hippocampus_volume'] = data['left_hippocampus'] + data['right_hippocampus']

    # Initialize total_intracranial column with NaN or zeros
    data['total_intracranial'] = pd.np.nan

    if Real:
        # Loop over each row in the DataFrame
        for index, row in data.iterrows():
            # Extract the original path
    
            original_path = row['path_org']
        
        
            # Replace the basename of the original path
            new_path = os.path.dirname(original_path) + "/synthseg_7_3_2/vols_T1toMNIlin.csv"

            # Read the matching CSV file
            csv_data = pd.read_csv(new_path)
            
            # Extract the 'total intracranial' value
            total_intracranial_value = csv_data['total intracranial'].iloc[0]
            
            # Insert the value into the 'total_intracranial' column of the DataFrame
            data.at[index, 'total_intracranial'] = total_intracranial_value
  
    return data




def read_and_preprocess_data(real_data_path, synthetic_data_path, test_data, MCI):
    # Read real data
    real_data = pd.read_csv(real_data_path)
    real_data.columns = real_data.columns.str.strip()
    real_data = real_data.drop_duplicates()

    # Preprocessing steps (as per your provided code)
    real_data = real_data.rename(columns={'age': 'Age', 'sex': 'Sex', 'cdr': 'CDGLOBAL'})
    real_data = real_data[real_data.CDGLOBAL.notna()]
    #Rewrite CDGLOBAL>1 as 1
    real_data.loc[real_data.CDGLOBAL > 1, 'CDGLOBAL'] = 1

    if test_data == "Org_Adni":
        #Rewrite Sex as 0: F, 1: M
        real_data.loc[real_data.Sex =='F', 'Sex'] = 0 #Female
        real_data.loc[real_data.Sex =='M', 'Sex'] = 1 #Male

    # Read synthetic data
    synthetic_data = pd.read_csv(synthetic_data_path)

    synthetic_data.columns = synthetic_data.columns.str.strip()
    synthetic_data = synthetic_data.drop_duplicates()

    # Preprocessing steps (as per your provided code)
    synthetic_data = synthetic_data[synthetic_data.CDGLOBAL.notna()]
    #Rewrite CDGLOBAL>1 as 1
    synthetic_data.loc[synthetic_data.CDGLOBAL > 1, 'CDGLOBAL'] = 1


    if not "includeMCI-":
        real_data = real_data[real_data['CDGLOBAL']!=0.5] # Exclude MCI
        synthetic_data = synthetic_data[synthetic_data['CDGLOBAL']!=0.5] # Exclude MCI
  
    return real_data, synthetic_data

def create_random_samples(real_data, synthetic_data, test_data, s_type, n_samples=5, scale=1, discard=True, MCI="includeMCI-", cov="Age", dom = 'M'):
    sampled_sets_Gen = []
    sampled_sets_Real = []
    all_selected_sub_ids = set()  # Global set to track all used Sub_IDs

    for sample_num in range(n_samples):

        if s_type == "matched":
            synthetic, data = create_matched_sample(real_data, synthetic_data, scale, discard, test_data, all_selected_sub_ids)
        elif s_type == "unmatched":
            synthetic, data = create_unmatched_sample(real_data, synthetic_data, scale, discard, test_data, cov, dom, all_selected_sub_ids)

        synthetic = dataframe(synthetic, Real=False)
        data = dataframe(data, Real=True)

        sampled_sets_Gen.append(synthetic)
        sampled_sets_Real.append(data)

        # Update the global set with new Sub_IDs from the current sample
        current_sub_ids = set(synthetic['Sub_IDs'])
        all_selected_sub_ids.update(current_sub_ids)

        # Save each sample to CSV
        extra = "discardedUnique" if discard else "duplicated"

        # Concatenate the two DataFrames
        combined_df = pd.concat([data, synthetic], ignore_index=True)

        # # Save the combined DataFrame to a CSV file
        # combined_df.to_csv(f'data-dist/{MCI}_{s_type}-{cov}-{dom}_{test_data}_{extra}Sub_IDs-scld_{scale}smpl_RealVsGen_cdr_sex_sample_{sample_num+1}.csv', index=False)

    return sampled_sets_Gen, sampled_sets_Real


def extract_number_from_path(path):
    basename = os.path.basename(path)
    match = re.search(r'seed_(\d+)', basename)
    return match.group(1) if match else None


def select_for_diversity(group, age_freq):
        # Filter the group to include only scans within the age range 60 to 90
        group = group[(group['Age'] >= 60) & (group['Age'] <= 90)]

        # If the filtered group is empty (no scans in the age range), return None or handle as needed
        if group.empty:
            return None

        group['Freq'] = group['Age'].map(age_freq)
        min_freq = group['Freq'].min()
        least_freq_scans = group[group['Freq'] == min_freq]

        # Randomly select one scan if there are multiple with the least frequency
        return least_freq_scans.sample(n=1).reset_index(drop=True)

def discard_repeated(data):

    # Calculate the frequency distribution of ages
    age_freq = data['Age'].value_counts()

    # Group by 'Sub_IDs' and apply the 'select_for_diversity' function
    unique_scans = data.groupby('Sub_IDs').apply(lambda x: select_for_diversity(x, age_freq)).reset_index(drop=True)

    print(unique_scans)

    # Reset the index for clean indexing
    unique_scans.reset_index(drop=True, inplace=True)

    return unique_scans

def validate_median_age(synthetic_sample, target_median_age):
    # Calculate the median age of the synthetic sample
    actual_median_age = synthetic_sample['Age'].median()

    print(f"Target Median Age: {target_median_age}")
    print(f"Actual Median Age of Synthetic Sample: {actual_median_age}")

    # Check if the actual median is close to the target median
    if abs(actual_median_age - target_median_age) <= 1:  # Tolerance of +/- 1 year
        print("Success: The median age of the synthetic sample aligns well with the target.")
    else:
        raise RuntimeError("Adjustment needed: The median age of the synthetic sample does not align with the target.")

def validate_altered_distribution(synthetic_sample, focus_covariate, target_distribution):
    # Calculate the distribution of the focus covariate in the synthetic sample
    synthetic_distribution = synthetic_sample[focus_covariate].value_counts(normalize=True)

    print(f"Target Distribution: {target_distribution}")
    print(f"Actual Distribution in Synthetic Sample: {synthetic_distribution}")

    # Check if the actual distribution aligns closely with the target distribution
    distribution_diff = (synthetic_distribution - target_distribution).abs().sum() / 2  # Total variation distance
    if distribution_diff <= 0.1:  # Tolerance threshold, e.g., 10%
        print("Success: The distribution of the synthetic sample aligns well with the target distribution.")
    else:
        raise RuntimeError("Adjustment needed: The distribution of the synthetic sample does not align with the target.")




def create_matched_sample(data, synthetic_data, scale, discard, test_data, excluded_sub_ids=None):

    # Apply the function to each row
    synthetic_data['Sub_IDs'] = synthetic_data['path'].apply(extract_number_from_path)

    # Exclude rows from synthetic_data where Sub_IDs is in excluded_sub_ids
    synthetic_data = synthetic_data[~synthetic_data['Sub_IDs'].isin(excluded_sub_ids)]

    matched_synthetic = pd.DataFrame()

    selected_sub_ids = set() if excluded_sub_ids is None else set(excluded_sub_ids)

    if test_data == "adni":
        # Extract Sub_IDs from the path_org column
        data['Sub_IDs'] = data['path_org'].str.extract(r'(\d+_S_\d+)')

    elif test_data == "Org_Adni":
        data['Sub_IDs'] = data['SUBJECT']


    # Initialize the list for drop indices
    drop_indices = []

    data = discard_repeated(data) if discard else data 
  
    # Group by 'Age', 'CDGLOBAL', and 'Sex' and iterate over each group
    for _, group in data.groupby(['Age', 'CDGLOBAL', 'Sex']):

        # Number of records for this demographic combination
        n_records = len(group)

        # Filter synthetic_data to exclude already selected Sub_IDs in the same sample
        synthetic_data = synthetic_data[~synthetic_data['Sub_IDs'].isin(selected_sub_ids)]

        # Convert 'col2' to int
        synthetic_data['Age'] = synthetic_data['Age'].astype(int)
        synthetic_data['Sex'] = synthetic_data['Sex'].astype(int)

        # Get matching rows in the synthetic dataset
        matched_samples = synthetic_data[
            (synthetic_data['Age'] == group['Age'].iloc[0]) &
            (synthetic_data['CDGLOBAL'] == group['CDGLOBAL'].iloc[0]) &
            (synthetic_data['Sex'] == group['Sex'].iloc[0])
        ]

        # If there are matched samples in the synthetic data
        if not matched_samples.empty:
            # Sample rows based on the scale
            sampled_data = matched_samples.sample(n=scale*n_records, random_state=42, replace=True)
            # Append the sampled rows to the result dataset
            matched_synthetic = pd.concat([matched_synthetic, sampled_data], axis=0)

            # Update the set of selected Sub_IDs
            selected_sub_ids.update(set(sampled_data['Sub_IDs']))

        else:
            print(f"No matching synthetic data for group: Age={group['Age'].iloc[0]}, CDGLOBAL={group['CDGLOBAL'].iloc[0]}, Sex={group['Sex'].iloc[0]}")
            drop_indices.extend(group.index.tolist())

            pass


    # Drop the rows from real_data using the collected indices
    data = data.drop(drop_indices)
    data.reset_index(drop=True, inplace=True)

    return matched_synthetic, data

def calculate_weights(row_age, target_age):
    difference = row_age - target_age
    if difference >= 0:  # For ages equal to or above the target
        weight = 1 / (1 + difference)
    else:  # For ages below the target
        weight = 1 / (1 + abs(difference) ** 2)  # Exponential penalty for ages below target
    return weight


def get_aggressive_weights(category, target_dist, current_dist):
    target_prop = target_dist.get(category, 0)
    current_prop = current_dist.get(category, 0)
    weight = target_prop / current_prop if current_prop > 0 else 0
    return weight

def create_unmatched_sample(data, synthetic_data, scale, discard, test_data, focus_covariate, dom, excluded_sub_ids=None):

    # Apply the function to each row
    synthetic_data['Sub_IDs'] = synthetic_data['path'].apply(extract_number_from_path)

    # Exclude rows from synthetic_data where Sub_IDs is in excluded_sub_ids
    synthetic_data = synthetic_data[~synthetic_data['Sub_IDs'].isin(excluded_sub_ids)]

    unmatched_synthetic = pd.DataFrame()

    if test_data == "adni":
        # Extract Sub_IDs from the path_org column
        data['Sub_IDs'] = data['path_org'].str.extract(r'(\d+_S_\d+)')

    elif test_data == "Org_Adni":
        data['Sub_IDs'] = data['SUBJECT']



    #The goal is to ensure that each subject in the dataset is represented by only one scan.
    data = discard_repeated(data)if discard else data

    if "Age" in focus_covariate:

        # Calculate the median age of the real data
        median_age_real = data['Age'].median()

        # Determine the shift for the synthetic data's median age
        # For example, shifting the median age by +/- 5 years
        shift_amount = 5    
        target_median_age = median_age_real + shift_amount if dom == "High" else median_age_real - shift_amount  # Can be -shift_amount for the opposite shift

        # Filter out rows where 'Age' is less than 60 or greater than 90
        synthetic_data = synthetic_data[(synthetic_data['Age'] >= 60) & (synthetic_data['Age'] <= 90)]

        # # Apply weights inversely proportional to the distance from the target median age
        if dom == "High":
            synthetic_data['weights'] = synthetic_data['Age'].apply(lambda x: calculate_weights(x, target_median_age))
        elif dom == "Low":
            synthetic_data['weights'] = 1 / (1 + abs(synthetic_data['Age'] - target_median_age))

        # Sample based on these weights
        sampled_data = synthetic_data.sample(n=scale * len(data), weights='weights', random_state=42, replace=False)
        unmatched_synthetic = pd.concat([unmatched_synthetic, sampled_data], axis=0)

        validate_median_age(sampled_data, target_median_age) #TODO

    elif "flip" in focus_covariate:
        # Define the focus covariate distribution from the real dataset
        real_covariate_distribution = data[focus_covariate].value_counts(normalize=True)

        # Flip the distribution for the unmatched sample
        flipped_distribution = 1 - real_covariate_distribution
        flipped_distribution = flipped_distribution / flipped_distribution.sum()

        # Sample based on the flipped distribution
        synthetic_data['Weight'] = synthetic_data[focus_covariate].map(flipped_distribution)
        sampled_data = synthetic_data.sample(n=scale * len(data), weights='Weight', random_state=42, replace=False)

        unmatched_synthetic = pd.concat([unmatched_synthetic, sampled_data], axis=0)

        validate_altered_distribution(sampled_data, focus_covariate, flipped_distribution)


    else:
        # Define the focus covariate distribution from the real dataset
        real_covariate_distribution = data[focus_covariate].value_counts(normalize=True)

        if 'Sex' in focus_covariate:
            if '5' in dom:
                lower = 0.45
                upper = 0.55
            elif '10' in dom:
                lower = 0.4
                upper = 0.6
            elif '15' in dom:
                lower = 0.35
                upper = 0.65
            
            if 'M' in dom:
                real_covariate_distribution[0] = lower
                real_covariate_distribution[1] = upper
            elif 'F' in dom:
                real_covariate_distribution[0] = upper
                real_covariate_distribution[1] = lower
            target_distribution = real_covariate_distribution

            # Sample based on the target distribution
            synthetic_data['Weight'] = synthetic_data[focus_covariate].map(target_distribution)

        elif 'CDGLOBAL' in focus_covariate:
            # Percentage of original dom MCI
            extra = real_covariate_distribution[0.5]
            if 'CN' in dom:
                #switch percentage between original dom MCI with CN
                real_covariate_distribution[0.5] = real_covariate_distribution[0]
                real_covariate_distribution[0] = extra

            elif 'AD' in dom:
                
                #switch percentage between original dom MCI with AD
                real_covariate_distribution[0.5] = real_covariate_distribution[1]
                real_covariate_distribution[1] = extra

            target_distribution = real_covariate_distribution

            current_distribution = synthetic_data[focus_covariate].value_counts(normalize=True)
            synthetic_data['Weight'] = synthetic_data[focus_covariate].apply(lambda x: get_aggressive_weights(x, target_distribution, current_distribution))

        # Sample based on the target distribution
        sampled_data = synthetic_data.sample(n=scale * len(data), weights='Weight', random_state=42, replace=False)

        unmatched_synthetic = pd.concat([unmatched_synthetic, sampled_data], axis=0)

        validate_altered_distribution(sampled_data, focus_covariate, target_distribution)
    
    
    return unmatched_synthetic, data


def check_duplicates_within_samples(sample_list):
    for i, df in enumerate(sample_list):
        if df['Sub_IDs'].duplicated().any():
            print(f"Duplicate found in sample {i}")
            # Print out the duplicated rows
            duplicates = df[df['Sub_IDs'].duplicated(keep=False)] 
            print("Duplicated Rows:")
            print(duplicates)
        else:
            print(f"No duplicates in sample {i}")


def check_duplicates_across_samples(sample_list):
    sub_id_occurrences = {}
    for i, df in enumerate(sample_list):
        for sub_id in df['Sub_IDs']:
            if sub_id in sub_id_occurrences:
                sub_id_occurrences[sub_id].append(i)
            else:
                sub_id_occurrences[sub_id] = [i]

    for sub_id, occurrences in sub_id_occurrences.items():
        if len(occurrences) > 1:
            print(f"Duplicate across samples found for Sub_IDs {sub_id} in samples {occurrences}")


# Function to load a NIfTI file and perform Area calculations
def Calc_Area_nifti(file_path):
        try:
            mask = nib.load(file_path)
            return np.sum(mask.get_fdata() > 0)

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        

def calculate_areas(n_samples, df_Real, df_Gen ,Image_Types, test_data, s_type, network, scale, cov="", dom='M'):

    Real_MASK_IDENTIFIER =  "synthseg_7_3_2/Pred_T1toMNIlin_synthseg.nii.gz"
    Syn_MASK_IDENTIFIER = 'Pred_mask_seed_'
    network = "D_Unet"

    for j in range(n_samples):
        for Image_Type in Image_Types:

            main_path=f"/path/to/segmentation/masks"

            if Image_Type == "Real":
                df = df_Real[j]
            elif Image_Type == "Gen":
                df = df_Gen[j]

            regions = ["VV", "HV", "IV"]
            for region in regions:

                if Image_Type == "Real":
                    df[f'new_path_{region}'] = df.apply(lambda row: main_path.replace("T1_unbiased_brain.nii.gz", Real_MASK_IDENTIFIER), axis=1)
            
                elif Image_Type == "Gen":
                    df[f'new_path_{region}'] = df.apply(lambda row: main_path + Syn_MASK_IDENTIFIER + row['Sub_IDs'] + '.nii.gz', axis=1)


            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                path_LV = row['new_path_VV']
                path_IV = row['new_path_IV']
                path_H = row['new_path_HV']

                area_LV = Calc_Area_nifti(path_LV)
                area_IV = Calc_Area_nifti(path_IV)
                area_H = Calc_Area_nifti(path_H)

                #Potential Division by Zero
                if area_IV == 0 or None:
                    raise RuntimeError("Check why IV pixel count = 0 or None is concerning")
                else:
                    correct_area_LV = area_LV / area_IV
                    correct_area_H = area_H / area_IV


                # Store the results in the DataFrame or process them as needed
                # Example: storing the mean of each image in a new column
                df.at[index, 'LV_Area'] = area_LV
                df.at[index, 'LV_correct_Area'] = correct_area_LV
                df.at[index, 'H_Area'] = area_H
                df.at[index, 'H_correct_Area'] = correct_area_H

                if Image_Type == "Real":
                    area_results_Real = df
                elif Image_Type == "Gen":
                    area_results_Gen = df


        # Add 'Image_Type' column to each DataFrame
        area_results_Gen['Image_Type'] = 'Synthetic'
        area_results_Real['Image_Type'] = 'Real'

        # Concatenate the two DataFrames
        combined_df = pd.concat([area_results_Real, area_results_Gen], ignore_index=True)

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(f'/path/to/...{s_type}{cov}{dom}{scale}.csv', index=False)


def main():

    scale = 1
    discard = True
    match = False
    network = "D_Unet"
    n_samples=5
    test_data = "Org_Adni"               
    extra = "discardedUnique" if discard else "duplicated"
    s_type = "matched" if match else "unmatched"
    Image_Types = ["Real", "Gen"]

    real_data_path ='/path/to/original/ADNI/df_mprage.csv'
    synthetic_data_path = "/path/to/synthetic/data/labels.csv"

    # Read and preprocess data
    real_data, synthetic_data = read_and_preprocess_data(real_data_path, synthetic_data_path, test_data)

    if match:
        # Create matched random samples
        samples_Gen, samples_Real = create_random_samples(real_data, synthetic_data, test_data, s_type, n_samples, scale, discard)

        check_duplicates_within_samples(samples_Gen)
        check_duplicates_across_samples(samples_Gen)

        calculate_areas(n_samples, samples_Real, samples_Gen, Image_Types, test_data, s_type, network, extra, scale)

    else:
        # Create unmatched samples
        focus_covariates = ['Age','CDGLOBAL','Sex']

        for cov in focus_covariates:
            
            if 'Age' in cov:
                doms = ['High', 'Low']  #+- 5 years default
            elif 'Sex' in cov:
                doms = ['M10', 'F10']   #10% deviation from balanced 50:50 ratio    
            elif 'CDGLOBAL' in cov:
                doms = ['CN', 'AD']     #CN or AD dominated instead of MCI dominated in real data   

            for dom in doms:
                # Create matched random samples
                samples_Gen, samples_Real = create_random_samples(real_data, synthetic_data, test_data, s_type, n_samples, scale, discard, MCI, cov, dom)

                check_duplicates_within_samples(samples_Gen)
                check_duplicates_across_samples(samples_Gen)

                calculate_areas(n_samples, samples_Real, samples_Gen, Image_Types, test_data, s_type, network, extra, scale, MCI, cov, dom)



if __name__ == '__main__':
    main()



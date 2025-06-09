import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from collections import defaultdict
from pandas.tseries.offsets import DateOffset


# Helper function from Ruiwu's script for data preprocessing, slicing and aggragating.
def drop_high_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with more than 50% missing values (NaN or None) from a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame after dropping columns.
    """
    # Calculate empty rate
    missing_ratio = df.isnull().mean()
    
    # Find the columns with an empty rate over 50%
    cols_to_drop = missing_ratio[missing_ratio >= 0.5].index
    
    # delete them
    df_cleaned = df.drop(columns=cols_to_drop)
    
    return df_cleaned

def object_processing(df):
    """
   Project columns whose dtype = object.
   For columns containing datetime, convert it to pd.datetime.
   For other object columns, encode them.
    """
    decode_dict = {}  # Dictionary

    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert data to datetime
            try:
                temp_dt = pd.to_datetime(df[col], errors='raise')
                df[col] = temp_dt.dt.strftime('%Y-%m-%d')
                df[col] = pd.to_datetime(df[col], errors='raise')
            except Exception:
                # If conversion unsuccessful, encode them
                unique_vals = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
                decode_dict[col] = mapping
    return df, decode_dict

def object_decoding(df, decode_dict):
    """
    Decoding
    """
    for col, mapping in decode_dict.items():
        # Reverse the dictionary for decoding
        reverse_mapping = {v: k for k, v in mapping.items()}
        df[col] = df[col].map(reverse_mapping)
    return df

def days_cal_v2(df, date_late, date_early, new_col_name):
    """
    Calculate how many days are there in the difference between column date_late and date_early.

    Args:
        df (pandas.DataFrame): Input DataFrame
        date_late (str): The name of the column containing later dates
        date_early (str): The name of the column containing earlier dates
        new_col_name (str): New column containing the calculation result

    Return:
        pandas.DataFrame: New DataFrame after processing，including new_col_name，but excluding date_late and date_early.
    """
    # Make sure that date_late and date_early are in datetime format.
    df[date_late] = pd.to_datetime(df[date_late])
    df[date_early] = pd.to_datetime(df[date_early])
    
    # Calculate how many days are there in the difference between column date_late and date_early.
    df[new_col_name] = ((df[date_late] - df[date_early]).dt.days)*1.00/365.00
    
    # Delete original columns.
    # df = df.drop(columns=[date_late, date_early])
    return df

def one_hot_encode_procedure_and_treatment(df):
    """
    Perform one-hot encoding on both the "procedure_code_y" and "treatment_category" columns of the input DataFrame.
    Replace the original columns with the one-hot encoded columns and return the modified DataFrame.

    Parameters:
        df (pandas.DataFrame): The original DataFrame containing the "procedure_code_y" and "treatment_category" columns.

    Returns:
        pandas.DataFrame: The modified DataFrame with the original columns replaced by their one-hot encoded counterparts.
    """
    # Make a copy of the input DataFrame to avoid modifying the original data.
    df_modified = df.copy()

    # Perform one-hot encoding on the "procedure_code_y" column
    procedure_dummies = pd.get_dummies(df_modified['procedure_code_y'], prefix='procedure_code_y')
    
    # Perform one-hot encoding on the "treatment_category" column
    treatment_dummies = pd.get_dummies(df_modified['treatment_category'], prefix='treatment_category')
    
    # Drop the original columns that have been encoded
    df_modified = df_modified.drop(columns=['procedure_code_y', 'treatment_category'])
    
    # Concatenate the one-hot encoded columns back to the DataFrame
    df_modified = pd.concat([df_modified, procedure_dummies, treatment_dummies], axis=1)
    
    return df_modified
def slice_and_aggregate(df):
    """
    Perform slicing and aggregation on the input DataFrame.
    
    Process:
      1. Construct age intervals based on the maximum value in the "procedure_age" column:
         - Starting from 10, create intervals of 10 years.
         - For non-final intervals, the range is [lower, lower+10) with the midpoint as lower + 5.
         - For the final interval, the range is [lower, max_age] with the midpoint as (lower + max_age) / 2.
      2. For each age interval (slice), filter the records within that interval.
      3. For each slice, group the data by "patient_id" and perform the following aggregations:
         - Sum the "amount" column for records with the same patient_id.
         - For all one-hot encoded columns with the prefix "procedure_code_y_", aggregate using a logical OR (using max as aggregation).
         - For all one-hot encoded columns with the prefix "treatment_category_", aggregate using a logical OR (using max as aggregation).
         - For the "procedure_age" column within the group, sort the values in ascending order, compute the differences between consecutive values, and take the mean as the "average_treatment_interval" (if only one record exists, use NaN).
         - Replace the "procedure_age" with the midpoint of the current interval.
         - For every other column (i.e., those not in the special set: "patient_id", "amount", "procedure_age", and the one-hot encoded columns),
           select the most frequently occurring element (mode) for that patient.
         - Add a new column "average_treatment_interval" to store the computed mean treatment interval.
      4. Each aggregated slice will have unique patient_id values.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame. It must include the following columns:
           - "procedure_age" (numeric, used for defining age intervals)
           - "patient_id"
           - "amount"
           - One-hot encoded columns with prefixes "procedure_code_y_" and "treatment_category_"
    
    Returns:
        list: A list of aggregated slice DataFrames.
    """
    # 1. Construct age intervals based on "procedure_age"
    max_age = df['procedure_age'].max()
    intervals = []
    lower_bound = 10
    while lower_bound < max_age:
        if lower_bound + 5 < max_age:
            upper_bound = lower_bound + 5
            center = lower_bound + 2.5
            intervals.append((lower_bound, upper_bound, center))
            lower_bound += 5
        else:
            # Final interval, including max_age
            upper_bound = max_age
            center = (lower_bound + max_age) / 2
            intervals.append((lower_bound, upper_bound, center))
            break

    result_slices = []
    
    # Identify one-hot encoded columns for procedure_code_y and treatment_category
    one_hot_proc_cols = [col for col in df.columns if col.startswith("procedure_code_y_")]
    one_hot_treatment_cols = [col for col in df.columns if col.startswith("treatment_category_")]
    
    # Define special columns that will be aggregated differently
    special_cols = {'patient_id', 'amount', 'procedure_age'} | set(one_hot_proc_cols) | set(one_hot_treatment_cols)
    # Other columns will be aggregated by taking the mode (most frequent value)
    other_cols = [col for col in df.columns if col not in special_cols]
    
    # Process each age interval (slice)
    for lower, upper, center in intervals:
        # Use [lower, upper) for non-final intervals and [lower, upper] (inclusive) for the final interval
        if lower + 5 < max_age:
            slice_df = df[(df['procedure_age'] >= lower) & (df['procedure_age'] < upper)]
        else:
            slice_df = df[(df['procedure_age'] >= lower) & (df['procedure_age'] <= upper)]
        
        # Skip this interval if there is no data
        if slice_df.empty:
            continue
        
        aggregated_rows = []
        # Group by "patient_id" to aggregate data for each patient
        grouped = slice_df.groupby('patient_id')
        for patient, group in grouped:
            aggregated_record = {}
            aggregated_record['patient_id'] = patient
            # Sum the "amount" column
            aggregated_record['amount'] = group['amount'].sum()
            
            # Aggregate one-hot encoded procedure_code_y columns using logical OR (max)
            for col in one_hot_proc_cols:
                aggregated_record[col] = group[col].max()
            
            # Aggregate one-hot encoded treatment_category columns using logical OR (max)
            for col in one_hot_treatment_cols:
                aggregated_record[col] = group[col].max()
            
            # Calculate the average treatment interval using the sorted "procedure_age" values of the group
            ages = group['procedure_age'].sort_values().tolist()
            if len(ages) > 1:
                diffs = [ages[i+1] - ages[i] for i in range(len(ages) - 1)]
                avg_interval = np.mean(diffs)
            else:
                avg_interval = np.nan
            aggregated_record['average_treatment_interval'] = avg_interval
            
            # Replace "procedure_age" with the midpoint of the current interval
            aggregated_record['procedure_age'] = center
            
            # For all remaining columns, use the mode (most frequent value) for this patient
            for col in other_cols:
                mode_series = group[col].mode()
                aggregated_record[col] = mode_series.iloc[0] if not mode_series.empty else np.nan
            
            aggregated_rows.append(aggregated_record)
        
        # Create an aggregated DataFrame for the current slice and append it to the result list
        agg_df = pd.DataFrame(aggregated_rows)
        result_slices.append(agg_df)
    
    return result_slices

def merge_columns_with_priority(df: pd.DataFrame, col1: str, col2: str, defined_new_col: str) -> pd.DataFrame:
    """
    For each row, if col2 is null and col1 is not null, copy value from col1 to col2.
    Then create a new column with name being the common prefix of col1 and col2.
    Drop the original two columns and insert the new column.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    col1 (str): First column name
    col2 (str): Second column name

    Returns:
    pd.DataFrame: Modified DataFrame with merged column
    """
    # Generate a new column for joining
    new_col = df[col2].combine_first(df[col1])

    # Get the shared prefix of col1 and col2
    def common_prefix(s1, s2):
        from os.path import commonprefix
        return commonprefix([s1, s2])
    
    new_col_name = defined_new_col

    # Delete old columns and add new one
    df = df.drop(columns=[col1, col2])
    df[new_col_name] = new_col

    return df


def classify_patient(group):
    procedure_dates = group.sort_values('procedure_date')['procedure_date']

    # Calculate total treatment span in years
    span_years = (group['last_visit'].iloc[0] - group['first_visit'].iloc[0]).days / 365.0

    # Calculate max gap between consecutive visits in years
    gaps = procedure_dates.diff().dropna().dt.days / 365.0
    max_gap = gaps.max() if not gaps.empty else 0

    # Classification rules
    if max_gap <= 2:
        if span_years >= 3:
            return 'V1'  # Consistent long history
        else:
            return 'V2'  # Consistent short history
    elif max_gap > 2 and span_years >= 7:
        return 'V3'  # Inconsistent but long history
    else:
        return 'V4'  # Inconsistent and short history (everything else)
    
# 5 year window
def age_to_group_5_year(age):
    try:
        age = int(age)
        if age < 0:
            return "invalid"
        lower = (age // 5) * 5
        upper = lower + 5
        if lower > upper:
            return "invalid"
        return f"{lower}-{upper}"
    except:
        return "invalid"

# Kmeans, Optimal K and PCA plots 
def kmeans_clustering(df_combined, lower, upper, scale=False, silent=False):
    df_cluster_input = df_combined.copy()
    X = df_cluster_input.drop(columns=['patient_id'])

    # Scaling
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Elbow method to determine optimal K
    inertia = []
    ks = list(range(2, 8))
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    kl = KneeLocator(ks, inertia, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    if not silent:
        print(f'The optimal K being select is {optimal_k}')

        # Elbow curve
        plt.figure(figsize=(8, 6))
        plt.plot(ks, inertia, marker='o')
        plt.title("Elbow Method for Optimal K")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    kmeans = KMeans(n_clusters=optimal_k, random_state=823)
    df_cluster_input['cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    if not silent:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=df_cluster_input['cluster'], cmap='viridis', s=30)
        plt.title(f"Patient Clusters (PCA view of age {lower}-{upper} + history)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_cluster_input, optimal_k


def plot_procedure_distribution(df_proc_timelines, target_code, average=False, age_range=(0, 101), bin_size=5):
    bins = list(range(age_range[0], age_range[1] + bin_size, bin_size))
    labels = [f"{b}-{b+bin_size}" for b in bins[:-1]]

    for cluster_id, df_proc in df_proc_timelines.items():
        df_proc = df_proc.copy()
        df_proc['age'] = df_proc['age'].astype(int)

        # Filter target procedure only
        df_code = df_proc[df_proc['procedure_code'] == target_code].copy()
        if df_code.empty:
            print(f"No data for procedure {target_code} in Cluster {cluster_id}")
            continue

        # Assign age bins
        df_code['age_bin'] = pd.cut(df_code['age'], bins=bins, right=False, labels=labels)

        # Group by individual age to get per-age counts
        age_counts = df_code.groupby('age').size().reset_index(name='count')
        age_counts['age_bin'] = pd.cut(age_counts['age'], bins=bins, right=False, labels=labels)

        # Total occurrences per bin
        total_occurrences = age_counts.groupby('age_bin')['count'].sum()

        if average:
            # Number of distinct ages with usage per bin (denominator)
            active_years = age_counts.groupby('age_bin')['age'].nunique()
            values = total_occurrences / active_years
            ylabel = "Mean Occurrences per Active Age"
            title = f"Avg Occurrences per Active Age for Procedure {target_code} (Cluster {cluster_id})"
        else:
            values = total_occurrences
            ylabel = "Total Occurrences"
            title = f"Total Occurrences of Procedure {target_code} per Age Bin - Cluster {cluster_id}"

        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(values.index.astype(str), values.values)
        plt.title(title)
        plt.xlabel("Age Bin")
        plt.ylabel(ylabel)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

# Define filter to exclude basic treatments
def is_non_basic(code):
    return not (code.startswith('1') or len(code) == 4)

# Mapping codes and their description
def get_procedure_description(input_code, Procedure_code_description_path):
    """
    Given an Excel file path and a string like 'procedure_code_y_###',
    returns a dictionary with description, price, and category of service
    for the corresponding procedure code.
    """
    import pandas as pd

    # Load mapping table
    df = pd.read_excel(Procedure_code_description_path, sheet_name="Mapping Table")
    code_number = input_code.replace("procedure_code_y_", "")

    # Find matching row
    match = df[df['CODE'].astype(str) == code_number]

    if not match.empty:
        description = match['DESCRIPTION'].values[0]
        price = match['PRICE'].values[0]
        category = match['CATEGORY OF SERVICE'].values[0]

        return {
            "DESCRIPTION": description,
            "PRICE": price,
            "CATEGORY OF SERVICE": category
        }
    else:
        return {
            "DESCRIPTION": f"No description found for code: {input_code}",
            "PRICE": None,
            "CATEGORY OF SERVICE": None
        }



def get_top_procedures_by_cluster(cluster_id, df_proc_timelines, description_mapping_path, non_basic=True, top_n=10):
    """
    Extract top N procedures for a given cluster ID.
    Filters by non-basic or basic procedures.
    Returns a DataFrame with procedure code, count, description, price, and category.
    """
    # Extract the cluster-specific timeline
    df_proc = df_proc_timelines[cluster_id]

    # Count procedure occurrences
    code_counts = df_proc['procedure_code'].value_counts()

    # Filter by non-basic or basic
    if non_basic:
        filtered_counts = code_counts[code_counts.index.map(is_non_basic)]
    else:
        filtered_counts = code_counts[~code_counts.index.map(is_non_basic)]

    # Take top N
    filtered_counts = filtered_counts.head(top_n)

    # Format to DataFrame
    top_codes_df = filtered_counts.reset_index()
    top_codes_df.columns = ['procedure_code', 'count']

    # Map descriptions, prices, and categories
    def extract_info(code):
        info = get_procedure_description(f'procedure_code_y_{code}', description_mapping_path)
        if isinstance(info, dict):
            return pd.Series([info.get('PRICE'), info.get('CATEGORY OF SERVICE'), info.get('DESCRIPTION'),])
        else:
            return pd.Series([info, None, None])

    top_codes_df[['price', 'category', 'description']] = top_codes_df['procedure_code'].apply(extract_info)

    return top_codes_df

def generate_sliding_windows(df, window_months=6, step_months=3, sequential = False, date_col='procedure_date', code_col='procedure_code_y'):
    """
    Generates overlapping time windows for each patient and collects procedure codes in each window.
    Returns:
        pd.DataFrame: DataFrame with columns ['patient_id', 'window_start', 'window_end', 'procedure_codes']
    """
    # Ensure the date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(['patient_id', date_col])
    results = []

    for patient_id, group in df.groupby('patient_id'):
        group = group.reset_index(drop=True)
        min_date = group[date_col].min()
        max_date = group[date_col].max()
        
        start_date = min_date
        while start_date <= max_date:
            if not sequential:
                end_date = start_date + DateOffset(months=window_months)
                mask = (group[date_col] >= start_date) & (group[date_col] < end_date)
                window_data = group.loc[mask]
                
                if not window_data.empty:
                    codes = [str(code) for code in window_data[code_col].tolist()]
                    results.append({
                        'patient_id': patient_id,
                        'window_start': start_date,
                        'window_end': end_date,
                        'procedure_codes': sorted(codes)
                    })

                start_date += DateOffset(months=step_months)    
            else:
                end_date = start_date + DateOffset(months=window_months)    
                # Filter rows in the current 6-month window
                mask_current = (group['procedure_date'] >= start_date) & (group['procedure_date'] < end_date)
                window_data_current = group.loc[mask_current]
                
                # Define the end date for the next 6-month window (look ahead)
                start_date_next = end_date
                end_date_next = start_date_next + DateOffset(months=6)
                
                # Filter rows in the next 6-month window
                mask_next = (group['procedure_date'] >= start_date_next) & (group['procedure_date'] < end_date_next)
                window_data_next = group.loc[mask_next]
            
                if not window_data_current.empty or not window_data_next.empty:
                    results.append({
                        'patient_id': patient_id,
                        'current_window_start': start_date,
                        'current_window_end': end_date,
                        'next_window_start': start_date_next,
                        'next_window_end': end_date_next,
                        'current_procedure_codes': sorted([str(code) for code in window_data_current[code_col].tolist()]),
                        'next_procedure_codes': sorted([str(code) for code in window_data_next[code_col].tolist()])
                    })


                start_date += DateOffset(months=step_months)

    return pd.DataFrame(results)

# Function to filter out procedure codes and remove duplicates
def filter_codes(codes):
    # Keep codes with length >= 5 and exclude codes with 5 digits starting with 1
    filtered_codes = [code for code in codes if len(str(code)) >= 5 and not (len(str(code)) == 5 and str(code).startswith('1'))]
    # Remove duplicates by converting to a set and then back to a list
    return list(set(filtered_codes))

# Create a DataFrame for encoding procedure codes as one-hot features
def encode_procedures(df, col_name):
    # Create a list of all unique procedure codes (from both current and next)
    all_codes = list(set([code for codes in df[col_name] for code in codes]))
    
    # Create a one-hot encoded DataFrame
    encoded_df = pd.DataFrame(columns=all_codes)
    
    # Fill the DataFrame with 1s where the procedure code is present
    for idx, row in df.iterrows():
        codes = set(row[col_name])
        encoded_df.loc[idx] = [1 if code in codes else 0 for code in all_codes]
    
    return encoded_df


def plot_lift_heatmap(pivot_table, 
                      title='Lift Heatmap', 
                      xlabel='Consequents', 
                      ylabel='Antecedents', 
                      figsize=(28, 18), 
                      cmap="YlGnBu"):
    """
    Plots a formatted heatmap for a given lift pivot table.

    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.0f',
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        linecolor='lightgray',
        square=True,
        annot_kws={"size": 8}
    )
    
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=14, labelpad=15)
    plt.ylabel(ylabel, fontsize=14, labelpad=15)
    
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

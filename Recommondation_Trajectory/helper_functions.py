import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

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
def days_cal(df, date_late, date_early, new_col_name):
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
    df = df.drop(columns=[date_late, date_early])
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
    
def cluster_patients_kmeans(df_svd, n_clusters=6):
    # Select only the SVD columns
    svd_cols = [col for col in df_svd.columns if col.startswith("SVD_")]
    X = df_svd[svd_cols]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to the DataFrame
    df_svd['Kmeans_cluster'] = cluster_labels

    return df_svd, kmeans, scaler


# AgglomerativeClustering
def cluster_patients_agglomerative(df_svd, n_clusters=6):
    svd_cols = [col for col in df_svd.columns if col.startswith("SVD_")]
    X = df_svd[svd_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    agglo = AgglomerativeClustering(n_clusters=n_clusters)

    df_svd['Agg_cluster'] = agglo.fit_predict(X_scaled)

    return df_svd, agglo, scaler

# GMM
def cluster_patients_gmm(df_svd, n_clusters=6):
    svd_cols = [col for col in df_svd.columns if col.startswith("SVD_")]
    X = df_svd[svd_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df_svd['gmm_cluster'] = gmm.fit_predict(X_scaled)

    return df_svd, gmm, scaler

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# KMeans
def grid_search_kmeans(df_svd, svd_prefix='SVD_', k_range=range(4, 11)):
    svd_cols = [col for col in df_svd.columns if col.startswith(svd_prefix)]
    X = StandardScaler().fit_transform(df_svd[svd_cols].values)

    inertias, silhouettes = [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = k_range[silhouettes.index(max(silhouettes))]
    best_model = KMeans(n_clusters=best_k, random_state=42).fit(X)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o')
    plt.title('KMeans: Inertia vs k')
    plt.xlabel('k')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, marker='o')
    plt.title('KMeans: Silhouette Score vs k')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()

    return best_model, best_k, silhouettes

# GMM
def grid_search_gmm(df_svd, svd_prefix='SVD_', k_range=range(4, 11)):
    svd_cols = [col for col in df_svd.columns if col.startswith(svd_prefix)]
    X = StandardScaler().fit_transform(df_svd[svd_cols].values)

    inertias, silhouettes = [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        silhouettes.append(score)
        
        # Pseudo-inertia (negative log-likelihood * n_samples)
        inertia_like = -gmm.score(X) * len(X)
        inertias.append(inertia_like)

    best_k = k_range[silhouettes.index(max(silhouettes))]
    best_model = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42).fit(X)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o')
    plt.title('GMM: Inertia vs k')
    plt.xlabel('k')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, marker='o')
    plt.title('GMM: Silhouette Score vs k')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()

    return best_model, best_k, silhouettes

# Agglomerative
def grid_search_agglomerative(df_svd, svd_prefix='SVD_', k_range=range(4, 11)):
    svd_cols = [col for col in df_svd.columns if col.startswith(svd_prefix)]
    X = StandardScaler().fit_transform(df_svd[svd_cols].values)

    inertias, silhouettes = [], []

    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        silhouettes.append(score)

        # Create Inertia: sum of squared distances to each cluster mean
        inertia = 0
        for cluster_id in np.unique(labels):
            cluster_points = X[labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            inertia += ((cluster_points - centroid) ** 2).sum()
        inertias.append(inertia)

    best_k = k_range[silhouettes.index(max(silhouettes))]
    best_model = AgglomerativeClustering(n_clusters=best_k).fit(X)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o')
    plt.title('Agg: Inertia vs k')
    plt.xlabel('k')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, marker='o')
    plt.title('Agg: Silhouette Score vs k')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()

    return best_model, best_k, silhouettes


def annotate_age_group_files(folder_path, cluster_map_kmeans, cluster_map_gmm, cluster_map_agg):
    age_group_data = {}

    for file in sorted(os.listdir(folder_path)):
        if not file.endswith(".csv"):
            continue

        # Extract age from filename
        age = int(file.split("_")[-1].replace(".csv", ""))
        file_path = os.path.join(folder_path, file)

        # Read age group data
        df = pd.read_csv(file_path)
        df = df.merge(cluster_map_kmeans, on='patient_id', how='left')
        df = df.merge(cluster_map_gmm, on='patient_id', how='left')
        df = df.merge(cluster_map_agg, on='patient_id', how='left')
        age_group_data[age] = df
        
    return age_group_data

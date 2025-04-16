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
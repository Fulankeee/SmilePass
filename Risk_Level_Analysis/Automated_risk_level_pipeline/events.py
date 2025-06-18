from datetime import datetime

# ------------------ Crowns Events ------------------
def crowns_event(treatment_df, patient_id, snapshot_year):
    crowns_codes = {code for code in range(27000, 29000) if str(code).startswith(('27', '29'))}
    data = treatment_df[
        (treatment_df['patient_id'] == patient_id) &
        (treatment_df['procedure_code_y'].isin(crowns_codes)) &
        (treatment_df['procedure_date'].dt.year > snapshot_year) &
        (treatment_df['procedure_date'].dt.year <= snapshot_year + 2)
    ]
    return not data.empty

def time_to_crowns(treatment_df, patient_id, snapshot_year):
    crowns_codes = {code for code in range(27000, 29000) if str(code).startswith(('27', '29'))}
    data = treatment_df[
        (treatment_df['patient_id'] == patient_id) &
        (treatment_df['procedure_code_y'].isin(crowns_codes)) &
        (treatment_df['procedure_date'].dt.year > snapshot_year) &
        (treatment_df['procedure_date'].dt.year <= snapshot_year + 2)
    ].sort_values('procedure_date')

    if data.empty:
        return None
    return (data.iloc[0]['procedure_date'] - datetime(snapshot_year, 12, 31)).days

# ------------------ Root Canal Events ------------------

def root_canal_event(treatment_df, patient_id, snapshot_year):
    root_canal_codes = {code for code in range(32000, 34000) if str(code).startswith(('32', '33'))}
    data = treatment_df[
        (treatment_df['patient_id'] == patient_id) &
        (treatment_df['procedure_code_y'].isin(root_canal_codes)) &
        (treatment_df['procedure_date'].dt.year > snapshot_year) &
        (treatment_df['procedure_date'].dt.year <= snapshot_year + 2)
    ]
    return not data.empty

def time_to_root_canal(treatment_df, patient_id, snapshot_year):
    root_canal_codes = {code for code in range(32000, 34000) if str(code).startswith(('32', '33'))}
    data = treatment_df[
        (treatment_df['patient_id'] == patient_id) &
        (treatment_df['procedure_code_y'].isin(root_canal_codes)) &
        (treatment_df['procedure_date'].dt.year > snapshot_year) &
        (treatment_df['procedure_date'].dt.year <= snapshot_year + 2)
    ].sort_values('procedure_date')

    if data.empty:
        return None
    return (data.iloc[0]['procedure_date'] - datetime(snapshot_year, 12, 31)).days

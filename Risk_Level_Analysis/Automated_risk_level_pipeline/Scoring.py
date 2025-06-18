import pandas as pd
from utils import recency_weight, treatment_score
from events import crowns_event, time_to_crowns, root_canal_event, time_to_root_canal
from Config import ALL_FEATURES, CROWNS, ROOT_CANALS

def get_treatments_in_window(treatment_df, patient_id, snapshot_year, years_back=3):
    start, end = snapshot_year - years_back, snapshot_year
    df = treatment_df[treatment_df['patient_id'] == patient_id]
    return df[
        (df['procedure_date'].dt.year >= start) &
        (df['procedure_date'].dt.year < end)
    ]

def visit_frequency(patient_treatments):
    n = len(patient_treatments)
    return n, 1.0 if n >= 6 else 0.7 if n >= 3 else 0.5

def generate_features(patient_treatments):
    features = {feat: 0 for feat in ALL_FEATURES}

    for _, row in patient_treatments.iterrows():
        code = row['procedure_code_y']
        for feat, code_list in ALL_FEATURES.items():
            if code in code_list:
                features[feat] += 1

    # Custom additional features
    features['crowns'] = patient_treatments['procedure_code_y'].isin(CROWNS).sum()
    features['root_canals'] = patient_treatments['procedure_code_y'].isin(ROOT_CANALS).sum()

    for d in range(4, 8):
        starts_with = patient_treatments[
            patient_treatments['procedure_code_y'].astype(str).str.startswith(str(d))
        ]
        features[f'starts_with_{d}_procedures'] = starts_with.shape[0]

    return features

def calculate_health_score(treatment_df, patient_id, snapshot_year):
    treatments = get_treatments_in_window(treatment_df, patient_id, snapshot_year)
    if treatments.empty:
        return 0, None, None

    treatments['treatment_score'] = treatments.apply(
        lambda row: treatment_score(row['risk_level'], row['procedure_date'], snapshot_year), axis=1
    )
    total_score = treatments['treatment_score'].sum()
    _, freq_weight = visit_frequency(treatments)
    total_score *= freq_weight

    # Event times (None if no future event)
    ttc = time_to_crowns(treatment_df, patient_id, snapshot_year) if crowns_event(treatment_df, patient_id, snapshot_year) else None
    ttr = time_to_root_canal(treatment_df, patient_id, snapshot_year) if root_canal_event(treatment_df, patient_id, snapshot_year) else None

    return total_score, ttc, ttr

def compute_rolling_scores(treatment_df, patient_df, end_year=2025):
    records = []

    for pid in patient_df['patient_id']:
        first_year = treatment_df[treatment_df['patient_id'] == pid]['procedure_date'].min().year

        for year in range(first_year + 1, end_year):
            score, ttc, ttr = calculate_health_score(treatment_df, pid, year)
            visits = get_treatments_in_window(treatment_df, pid, year)

            recency = visits['procedure_date'].apply(lambda d: recency_weight(d, year)).mean() if not visits.empty else 0
            n_visits, freq_weight = visit_frequency(visits)
            avg_risk = visits['risk_level'].mean() if not visits.empty else 0
            features = generate_features(visits) if not visits.empty else {}

            records.append({
                'patient_id': pid,
                'year': year,
                'health_score': score,
                'time_to_crowns': ttc,
                'time_to_root_canal': ttr,
                'recency_score': recency,
                'visit_frequency': freq_weight,
                'average_risk_level': avg_risk,
                'num_visits': n_visits,
                **features
            })

    return pd.DataFrame(records)

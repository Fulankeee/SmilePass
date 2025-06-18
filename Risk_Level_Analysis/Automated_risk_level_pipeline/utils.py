from datetime import datetime

# Date and Score Utilities
def years_since(procedure_date, snapshot_year):
    snapshot = datetime(snapshot_year, 12, 31)
    return (snapshot - procedure_date).days // 365

def recency_weight(procedure_date, snapshot_year):
    yrs = years_since(procedure_date, snapshot_year)
    return 1.0 if yrs <= 0.5 else 0.7 if yrs <= 1 else 0.5 if yrs <= 2 else 0.3

def treatment_score(risk, procedure_date, snapshot_year):
    return risk * recency_weight(procedure_date, snapshot_year)

"""
ml/duplicates.py
----------------
Duplicate Detection & Handling Module (CO1 Syllabus) for Student Placement Prediction.

Provides modular functions for:
1. Exact-row duplicate detection (all columns identical)
2. Subset / business-key duplicate detection (shared academic credentials & stream)
3. Analytical duplicate removal evaluation (before/after row counts, retention %)
4. Risk assessment of removing legitimate students with identical test scores
5. Complete audit summary for Flask dashboard
"""

import os
import pandas as pd
import numpy as np

# Standard business key subset representing unique student metric combinations
STUDENT_KEY_SUBSET = [
    'CGPA',
    'AttendancePercent',
    'CodingTestScore',
    'AptitudeTestScore',
    'Stream',
    'CollegeTier'
]


def audit_exact_duplicates(df):
    """
    Computes exact full-row duplicate count and percentage.
    A full-row duplicate means all columns across all fields are strictly identical.
    """
    total_records = len(df)
    unique_records = len(df.drop_duplicates())
    duplicate_count = total_records - unique_records
    duplicate_pct = round((duplicate_count / total_records) * 100.0, 4)

    return {
        "total_records": total_records,
        "unique_records": unique_records,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "has_exact_duplicates": duplicate_count > 0,
        "status": "No exact duplicates detected" if duplicate_count == 0 else f"{duplicate_count} exact duplicates found"
    }


def audit_subset_duplicates(df, subset=None):
    """
    Computes business-key subset duplicate count and percentage.
    Evaluates whether students share academic grades, test percentiles, and stream.
    """
    if subset is None:
        subset = [c for c in STUDENT_KEY_SUBSET if c in df.columns]

    total_records = len(df)
    unique_subset_records = len(df.drop_duplicates(subset=subset))
    subset_duplicate_count = total_records - unique_subset_records
    subset_duplicate_pct = round((subset_duplicate_count / total_records) * 100.0, 4)

    return {
        "total_records": total_records,
        "unique_records": unique_subset_records,
        "duplicate_count": subset_duplicate_count,
        "duplicate_pct": subset_duplicate_pct,
        "subset_columns": subset,
        "has_subset_duplicates": subset_duplicate_count > 0
    }


def evaluate_duplicate_removal_treatments(df, exact_audit=None, subset_audit=None):
    """
    Compares duplicate handling strategies analytically:
    1. Baseline (Retain All Records): 100% preservation
    2. Exact Row Deletion: Removes true technical duplicates
    3. Subset / Key Deletion: Removes students sharing the academic metric subset
    """
    if exact_audit is None:
        exact_audit = audit_exact_duplicates(df)
    if subset_audit is None:
        subset_audit = audit_subset_duplicates(df)

    total_records = exact_audit["total_records"]

    treatments = [
        {
            "strategy": "Retain All Records (Baseline)",
            "description": "Preserve all records without deleting duplicate score combinations.",
            "original_records": total_records,
            "records_removed": 0,
            "remaining_records": total_records,
            "retention_pct": 100.0,
            "risk_assessment": "Zero risk of data loss. Preserves legitimate student cohorts with standardized test percentiles."
        },
        {
            "strategy": "Exact-Row Deletion",
            "description": "Drop records where every feature across all columns is strictly identical.",
            "original_records": total_records,
            "records_removed": exact_audit["duplicate_count"],
            "remaining_records": exact_audit["unique_records"],
            "retention_pct": round((exact_audit["unique_records"] / total_records) * 100.0, 4),
            "risk_assessment": "Safe. True identical rows are pipeline logging repetitions or database insertion duplicates."
        },
        {
            "strategy": "Academic-Key Subset Deletion",
            "description": f"Drop records sharing identical [{', '.join(STUDENT_KEY_SUBSET[:3])}, ...].",
            "original_records": total_records,
            "records_removed": subset_audit["duplicate_count"],
            "remaining_records": subset_audit["unique_records"],
            "retention_pct": round((subset_audit["unique_records"] / total_records) * 100.0, 4),
            "risk_assessment": "High Risk. In a cohort of 50,000 university students, distinct candidates frequently earn identical rounded CGPAs and standardized aptitude percentiles. Removing them distorts branch placement rates."
        }
    ]

    return treatments


def explain_legitimate_student_risks():
    """
    Returns educational explanation regarding the operational risks
    of removing non-exact duplicate records in student datasets.
    """
    return [
        {
            "factor": "Coincidental Score Parity in Large Cohorts",
            "explanation": "In large engineering institutions (50,000 students), integer score scales (e.g. 75/100 aptitude, 80/100 coding, 7.5 CGPA) naturally lead to multiple students holding the same numerical scores."
        },
        {
            "factor": "Stream and Tier Granularity",
            "explanation": "Categorical fields like Stream (CS, IT) and CollegeTier (Tier 1, Tier 2) have low cardinalities. High score overlap within these buckets is expected and valid."
        },
        {
            "factor": "Impact on Placement Statistics",
            "explanation": "Deleting subset duplicates would artificially bias placement percentage estimations and training distributions, producing unfair predictions for common score profiles."
        }
    ]


def run_complete_duplicate_analysis(df):
    """
    Orchestrates complete CO1 Duplicate Analysis:
    1. Exact full-row duplicate audit
    2. Academic-key subset duplicate audit
    3. Analytical treatment comparison table
    4. Legitimate student risk explanations
    """
    exact_audit = audit_exact_duplicates(df)
    subset_audit = audit_subset_duplicates(df)
    treatments = evaluate_duplicate_removal_treatments(df, exact_audit, subset_audit)
    risks = explain_legitimate_student_risks()

    return {
        "exact_duplicates": exact_audit,
        "subset_duplicates": subset_audit,
        "treatment_comparison": treatments,
        "risks": risks,
        "verdict": f"Exact row duplicates: {exact_audit['duplicate_count']} ({exact_audit['duplicate_pct']}%). Academic metric duplicates represent valid independent students across the 50k cohort."
    }

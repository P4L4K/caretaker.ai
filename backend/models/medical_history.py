"""Pydantic schemas for Medical History API responses/requests."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import date, datetime


# ---------- Lab Values ----------

class LabValueSchema(BaseModel):
    id: int
    metric_name: str
    metric_value: float
    unit: Optional[str]
    normalized_value: float
    normalized_unit: Optional[str]
    reference_range_low: Optional[float]
    reference_range_high: Optional[float]
    is_abnormal: bool
    pct_change_from_previous: Optional[float]
    pct_change_from_baseline: Optional[float]
    recorded_date: date

    class Config:
        from_attributes = True


# ---------- Conditions ----------

class ConditionHistorySchema(BaseModel):
    id: int
    previous_status: Optional[str]
    new_status: str
    previous_severity: Optional[str]
    new_severity: Optional[str]
    status_version: int
    clinical_interpretation: Optional[str]
    change_reason: Optional[str]
    recorded_at: datetime

    class Config:
        from_attributes = True


class ConditionSchema(BaseModel):
    id: int
    disease_code: str
    disease_name: str
    status: str
    severity: Optional[str]
    status_version: int
    first_detected: date
    last_updated: date
    resolved_date: Optional[date]
    baseline_value: Optional[float]
    baseline_date: Optional[date]
    consecutive_normal_count: int
    confidence_score: float
    source_type: str

    class Config:
        from_attributes = True


class ConditionWithHistory(ConditionSchema):
    history: List[ConditionHistorySchema] = []


# ---------- Alerts ----------

class AlertSchema(BaseModel):
    id: int
    alert_type: str
    message: str
    severity: str
    is_read: bool
    condition_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- Trends ----------

class TrendDataPoint(BaseModel):
    date: date
    value: float
    unit: Optional[str]
    is_abnormal: bool
    pct_change_from_previous: Optional[float]
    pct_change_from_baseline: Optional[float]


class MetricTrend(BaseModel):
    metric_name: str
    data_points: List[TrendDataPoint]
    reference_range_low: Optional[float]
    reference_range_high: Optional[float]
    trend_direction: str  # increasing/decreasing/stable/fluctuating
    volatility: float
    volatility_label: str  # Low/Medium/High
    latest_value: Optional[float]
    baseline_value: Optional[float]


class TrendSummary(BaseModel):
    metrics: List[MetricTrend]
    available_metrics: List[str]


# ---------- Risk Score ----------

class RiskFactor(BaseModel):
    factor: str
    contribution: float


class RiskScoreBreakdown(BaseModel):
    risk_score: float
    risk_category: str  # Low/Moderate/High/Critical
    risk_trajectory: str  # increasing/stable/improving
    factors: List[RiskFactor]


# ---------- Full Patient State ----------

class PatientMedicalState(BaseModel):
    active_conditions: List[ConditionSchema]
    past_conditions: List[ConditionSchema]
    latest_labs: List[LabValueSchema]
    risk_score: Optional[RiskScoreBreakdown]
    unread_alert_count: int
    recent_alerts: List[AlertSchema]


# ---------- Health Analysis ----------

class HealthAnalysisResult(BaseModel):
    risk_score: float
    risk_category: str
    risk_trajectory: str
    factors: List[RiskFactor]
    overall_health_status: str
    explanation: str
    recommendations: List[str]
    monitoring_frequency: str

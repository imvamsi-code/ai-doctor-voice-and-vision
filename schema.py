from pydantic import BaseModel, Field
from typing import List

class MedicalAnalysisSchema(BaseModel):
    primary_observation: str = Field(
        description="Detailed visual analysis of lesions, rash, swelling, discoloration, or irregularities."
    )
    differential_diagnosis: List[str] = Field(
        description="Top 2-3 possible clinical conditions based on retrieved evidence and visual inspection."
    )
    severity_level: str = Field(
        description="Severity triage: 'Low (Self-limiting)', 'Moderate (Requires Evaluation)', or 'Critical (Emergency)'."
    )
    evidence_grounding: List[str] = Field(
        description="Direct clinical citations or guidelines retrieved from the medical knowledge base."
    )
    recommended_actions: List[str] = Field(
        description="Next clinical steps, self-care measures, or specialist referrals."
    )
    red_flags: List[str] = Field(
        description="Urgent symptoms that necessitate immediate emergency care."
    )
    spoken_summary: str = Field(
        description="Concise, empathetic 2-3 sentence clinical summary formatted specifically for TTS audio synthesis."
    )

from pydantic import BaseModel
from typing_extensions import TypedDict
from typing import Annotated, Optional
from langgraph.graph import add_messages
from langgraph.graph import MessagesState

# Comprehensive Decision State
class ComprehensiveDecisionState(BaseModel):
    predicted_class: str
    confidence_level: str
    report: str

# Feature Synthesizer State
class FeatureSynthesizerState(BaseModel):
    report: str

# Quantitative Reasoning State
class QuantitativeReasoningState(BaseModel):
    overall_confidence_score: float
    predicted_class: str
    preliminary_hypothesis: str

# Heuristic state
class HeuristicState(BaseModel):
    malware_confidence_score: float
    benign_confidence_score: float
    capability: str

# CNN state
class CnnState(BaseModel):
    malware_confidence_score: float
    benign_confidence_score: float
    Xai_raw_bytes: dict

# Gin state
class GinState(BaseModel):
    malware_confidence_score: float
    benign_confidence_score: float
    Xai_cfg: dict
    

class XaiDetectorState(TypedDict):
    file_name: str
    gin_state: GinState
    cnn_state: CnnState
    heuristic_state: HeuristicState
    quantitative_reasoning_state: QuantitativeReasoningState
    feature_synthesizer_state: FeatureSynthesizerState
    comprehensive_decision_state: ComprehensiveDecisionState
    
# Input State
class Input_State(BaseModel):
    file_name: str    

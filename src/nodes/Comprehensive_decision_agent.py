import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.state.state import ComprehensiveDecisionState
from src.prompts.prompt import comprehensive_decision_agent_prompt

load_dotenv()

OLLAMA = os.getenv("OLLAMA", "false").lower()
MODEL = os.getenv("MODEL", "Qwen2.5-coder:7b")
BASE_URL = os.getenv("BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")

if OLLAMA == "true":
    base_llm = ChatOllama(model=MODEL, temperature=0.1)
else:
    if BASE_URL:
        base_llm = ChatOpenAI(model=MODEL, temperature=0.1, base_url=BASE_URL, api_key=API_KEY)
    else:
        base_llm = ChatOpenAI(model=MODEL, temperature=0.1, api_key=API_KEY)

# Create structured LLM
comprehensive_decision_llm = base_llm.with_structured_output(ComprehensiveDecisionState)

def Comprehensive_decision_agent(state: dict):
    print("[Comprehensive_decision_agent] Starting analysis...")
    
    # 1. Extract Quantitative Assessment Report
    qa_state = state.get("quantitative_reasoning_state")
    if isinstance(qa_state, dict):
        overall_confidence_score = qa_state.get("overall_confidence_score", 0.0)
        predicted_class = qa_state.get("predicted_class", "unknown")
        preliminary_hypothesis = qa_state.get("preliminary_hypothesis", "")
    else:
        overall_confidence_score = getattr(qa_state, "overall_confidence_score", 0.0) if qa_state else 0.0
        predicted_class = getattr(qa_state, "predicted_class", "unknown") if qa_state else "unknown"
        preliminary_hypothesis = getattr(qa_state, "preliminary_hypothesis", "") if qa_state else ""
        
    assessor_report = {
        "overall_confidence_score": overall_confidence_score,
        "predicted_class": predicted_class,
        "preliminary_hypothesis": preliminary_hypothesis
    }
    
    # 2. Extract Feature Synthesizer Report
    qs_state = state.get("feature_synthesizer_state")
    if isinstance(qs_state, dict):
        synthesizer_report = qs_state.get("report", "")
    else:
        synthesizer_report = getattr(qs_state, "report", "") if qs_state else ""
        
    # 3. Construct Prompts
    system_prompt = comprehensive_decision_agent_prompt
    user_prompt = f"""
    Quantitative Assessment Report:
    {json.dumps(assessor_report, indent=2)}
    
    Feature Synthesizer Report:
    {synthesizer_report}
    """
    
    messages_system = SystemMessage(content=system_prompt)
    messages_user = HumanMessage(content=user_prompt)

    # 4. Invoke LLM
    result = comprehensive_decision_llm.invoke([messages_system, messages_user])

    # 5. Save output to JSON
    file_name = state.get("file_name", "unknown")
    output_dir = os.path.join("d:\\NCKH\\Xai-detector\\output", str(file_name))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "Comprehensive_decision_result.json")
    result_data = {
        "predicted_class": result.predicted_class,
        "confidence_level": result.confidence_level,
        "report": result.report
    }
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"[Comprehensive_decision_agent] Saved result to {output_file}")
    except Exception as e:
        print(f"[Comprehensive_decision_agent] Error saving result: {e}")

    # 6. Return State
    return {
        "comprehensive_decision_state": ComprehensiveDecisionState(
            predicted_class=result.predicted_class,
            confidence_level=result.confidence_level,
            report=result.report
        )
    }

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from src.state.state import QuantitativeReasoningState
from src.prompts.prompt import quantitative_reasoning_agent_prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
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

# Create structured LLM like CoderAgent
quantitative_reasoning_llm = base_llm.with_structured_output(QuantitativeReasoningState)

def Quantitative_reasoning_agent(state: dict):
    print("[Quantitative_reasoning_agent] Starting analysis...")
    
    # Helper to get score from either dict or Pydantic model
    def get_score(obj, key):
        if not obj: return 0.0
        return obj.get(key, 0.0) if isinstance(obj, dict) else getattr(obj, key, 0.0)

    # Get confidence score
    malware_gin_score = get_score(state.get("gin_state"), "malware_confidence_score")
    benign_gin_score = get_score(state.get("gin_state"), "benign_confidence_score")
    malware_cnn_score = get_score(state.get("cnn_state"), "malware_confidence_score")
    benign_cnn_score = get_score(state.get("cnn_state"), "benign_confidence_score")
    malware_heuristic_score = get_score(state.get("heuristic_state"), "malware_confidence_score")
    benign_heuristic_score = get_score(state.get("heuristic_state"), "benign_confidence_score")

    system_prompt = quantitative_reasoning_agent_prompt
    user_prompt = f"""
    Here are the confidence scores from the three models:
    - CNN Score: malware={malware_cnn_score}, benign={benign_cnn_score}
    - GNN Score: malware={malware_gin_score}, benign={benign_gin_score}
    - Heuristic Score: malware={malware_heuristic_score}, benign={benign_heuristic_score}
    If the confidence score of malware and benign are 0.0. It mean the model can't classify the class.
    Analyze these scores and produce a structured JSON output according to the system prompt.
    """
    
    messages_system = SystemMessage(content=system_prompt)
    messages_user = HumanMessage(content=user_prompt)

    result = quantitative_reasoning_llm.invoke([messages_system, messages_user])

    # Save output to JSON
    file_name = state.get("file_name", "unknown")
    output_dir = os.path.join("d:\\NCKH\\Xai-detector\\output", str(file_name))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "Quantitative_reasoning_result.json")
    result_data = {
        "overall_confidence_score": result.overall_confidence_score,
        "predicted_class": result.predicted_class,
        "preliminary_hypothesis": result.preliminary_hypothesis
    }
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"[Quantitative_reasoning_agent] Saved result to {output_file}")
    except Exception as e:
        print(f"[Quantitative_reasoning_agent] Error saving result: {e}")

    return {
        "quantitative_reasoning_state": QuantitativeReasoningState(
            overall_confidence_score=result.overall_confidence_score,
            predicted_class=result.predicted_class,
            preliminary_hypothesis=result.preliminary_hypothesis
        )
    }

   
    
    
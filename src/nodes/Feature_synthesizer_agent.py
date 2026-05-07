from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from src.state.state import FeatureSynthesizerState
from src.prompts.prompt import feature_synthesizer_agent_prompt
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
feature_synthesizer_llm = base_llm.with_structured_output(FeatureSynthesizerState)

def Feature_synthesizer_agent(state: dict):
    print("[Feature_synthesizer_agent] Starting analysis...")
    
    # 1. Extract Xai_cfg from gin_state
    gin_state = state.get("gin_state")
    xai_cfg = gin_state.get("Xai_cfg", {}) if isinstance(gin_state, dict) else (getattr(gin_state, "Xai_cfg", {}) if gin_state else {})
    
    subgraphs = xai_cfg.get("important_subgraphs", [])
    formatted_subgraphs = []
    for sg in subgraphs:
        sg_nodes = []
        for n in sg.get("nodes", []):
            opcodes = n.get("opcodes", "")
            if isinstance(opcodes, list):
                opcodes_str = "; ".join(opcodes)
            else:
                opcodes_str = "; ".join(str(opcodes).split())
            
            sg_nodes.append({
                "name": n.get("name"),
                "role": n.get("role"),
                "opcodes": opcodes_str
            })
        formatted_subgraphs.append({
            "center_importance": sg.get("center_importance"),
            "nodes": sg_nodes,
            "edges": sg.get("edges", [])
        })

    # 2. Extract Xai_raw_bytes from cnn_state
    cnn_state = state.get("cnn_state")
    xai_raw_bytes = cnn_state.get("Xai_raw_bytes", {}) if isinstance(cnn_state, dict) else (getattr(cnn_state, "Xai_raw_bytes", {}) if cnn_state else {})
    
    top_k_opcodes = xai_raw_bytes.get("top_k_opcodes", [])
    formatted_raw_bytes = []
    for item in top_k_opcodes:
        instructions = item.get("instructions", [])
        if isinstance(instructions, list):
            instructions_str = "; ".join([str(i) for i in instructions])
        else:
            instructions_str = str(instructions)
            
        formatted_raw_bytes.append({
            "heat_score": item.get("heat_score"),
            "instruction": instructions_str
        })

    # 3. Extract capability from heuristic_state
    heuristic_state = state.get("heuristic_state")
    capability = heuristic_state.get("capability", "") if isinstance(heuristic_state, dict) else (getattr(heuristic_state, "capability", "") if heuristic_state else "")

    # 4. Invoke LLM
    system_prompt = feature_synthesizer_agent_prompt
    user_prompt = f"""
    Here are the 3 signature feautures from the three models:
    Top-K ASM Blocks (Grad-CAM):
    {json.dumps(formatted_raw_bytes, indent=2)}
    
    Top-K CFG Subgraphs (GNNExplainer):
    {json.dumps(formatted_subgraphs, indent=2)}
    
    Capability Features (CAPA):
    {capability}
    """
    
    messages_system = SystemMessage(content=system_prompt)
    messages_user = HumanMessage(content=user_prompt)

    result = feature_synthesizer_llm.invoke([messages_system, messages_user])

    # 5. Save output to JSON
    file_name = state.get("file_name", "unknown")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "output", str(file_name))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "Feature_synthesizer_result.json")
    result_data = {
        "report": result.report
    }
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"[Feature_synthesizer_agent] Saved result to {output_file}")
    except Exception as e:
        print(f"[Feature_synthesizer_agent] Error saving result: {e}")

    # 6. Return State
    return {
        "feature_synthesizer_state": FeatureSynthesizerState(
            report=result.report
        )
    }
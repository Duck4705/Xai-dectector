from langgraph.graph import START, END, StateGraph
from src.nodes.gin_node import GinNode
from src.nodes.CNN_node import CnnNode
from src.nodes.Heuristic_node import HeuristicNode
from src.nodes.Quantitative_reasoning_agent import Quantitative_reasoning_agent
from src.nodes.Feature_synthesizer_agent import Feature_synthesizer_agent
from src.nodes.Comprehensive_decision_agent import Comprehensive_decision_agent
from src.state.state import XaiDetectorState, Input_State



# State Graph Builder
builder = StateGraph(XaiDetectorState, input_schema=Input_State)

# Build Nodes
builder.add_node("GinNode", GinNode)
builder.add_node("CnnNode", CnnNode)
builder.add_node("HeuristicNode", HeuristicNode)
builder.add_node("Quantitative_reasoning_agent", Quantitative_reasoning_agent)
builder.add_node("Feature_synthesizer_agent", Feature_synthesizer_agent)
builder.add_node("Comprehensive_decision_agent", Comprehensive_decision_agent)
# Build Edges
builder.add_edge(START, "GinNode")
builder.add_edge(START, "CnnNode")
builder.add_edge(START, "HeuristicNode")
builder.add_edge("GinNode", "Quantitative_reasoning_agent")
builder.add_edge("CnnNode", "Quantitative_reasoning_agent")
builder.add_edge("HeuristicNode", "Quantitative_reasoning_agent")
builder.add_edge("GinNode", "Feature_synthesizer_agent")
builder.add_edge("CnnNode", "Feature_synthesizer_agent")
builder.add_edge("HeuristicNode", "Feature_synthesizer_agent")
builder.add_edge("Quantitative_reasoning_agent", "Comprehensive_decision_agent")
builder.add_edge("Feature_synthesizer_agent", "Comprehensive_decision_agent")
builder.add_edge("Comprehensive_decision_agent", END)

# Conditional edges from CheckerAgent
# builder.add_conditional_edges(
#     "CheckerAgent",
#     should_continue,
#     {
#         "execute_builder": "Execute_Builder",
#         "end": END
#     }
# )

# Compile graph
graph = builder.compile()
from langgraph.graph import StateGraph

class RAGState(dict):
    pass

def rag_node(state):
    return state

def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rag", rag_node)
    graph.set_entry_point("rag")
    graph.set_finish_point("rag")
    return graph.compile()

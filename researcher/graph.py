"""LangGraph workflow.

planner -> researcher (one parallel branch per sub-question via Send) -> writer -> critic
critic loops back to writer until approved or MAX_REVISIONS, then pauses for human review.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from researcher.agents import Agents
from researcher.state import ResearchState

MAX_REVISIONS = 2


def build_graph(agents: Agents, checkpointer=None):
    def planner(state: ResearchState):
        return {"plan": agents.plan(state["question"]), "revisions": 0, "feedback": ""}

    def fan_out(state: ResearchState):
        return [Send("researcher", {"sub_question": q}) for q in state["plan"]]

    def researcher(payload: dict):
        return {"notes": [agents.research(payload["sub_question"])]}

    def writer(state: ResearchState):
        draft = agents.write(state["question"], state["notes"], state.get("feedback", ""))
        return {"draft": draft}

    def critic(state: ResearchState):
        review = agents.review(state["question"], state["draft"])
        return {
            "approved": review.approved,
            "feedback": review.feedback,
            "revisions": state.get("revisions", 0) + 1,
        }

    def after_critic(state: ResearchState):
        if state["approved"] or state["revisions"] > MAX_REVISIONS:
            return "human_review"
        return "writer"

    def human_review(state: ResearchState):
        decision = interrupt({"draft": state["draft"], "critic_feedback": state["feedback"]})
        if isinstance(decision, str) and decision.strip().lower() not in {"", "approve", "ok"}:
            # Human gave edit instructions: send them back to the writer.
            return Command(goto="writer", update={"feedback": decision, "human_decision": decision})
        return Command(goto=END, update={"human_decision": "approve"})

    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("human_review", human_review)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher"])
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", after_critic, ["writer", "human_review"])
    return g.compile(checkpointer=checkpointer or MemorySaver())

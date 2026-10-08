from langgraph.types import Command

from researcher.agents import Review
from researcher.graph import MAX_REVISIONS, build_graph


class StubAgents:
    def __init__(self, approve_after: int):
        self.approve_after = approve_after
        self.reviews = 0
        self.writes = []

    def plan(self, question):
        return ["q1", "q2", "q3"]

    def research(self, sub_question):
        return {"sub_question": sub_question, "findings": f"facts {sub_question}", "sources": ["u"]}

    def write(self, question, notes, feedback):
        self.writes.append(feedback)
        return f"draft v{len(self.writes)} from {len(notes)} notes"

    def review(self, question, draft):
        self.reviews += 1
        ok = self.reviews >= self.approve_after
        return Review(approved=ok, feedback="" if ok else "add citations")


def run(agents, thread="t"):
    graph = build_graph(agents)
    config = {"configurable": {"thread_id": thread}}
    return graph, config, graph.invoke({"question": "Q?"}, config)


def test_parallel_research_collects_all_notes():
    graph, config, result = run(StubAgents(approve_after=1))
    state = graph.get_state(config).values
    assert len(state["notes"]) == 3
    assert "__interrupt__" in result


def test_critic_loop_revises_then_pauses_for_human():
    agents = StubAgents(approve_after=2)
    _, _, result = run(agents)
    assert agents.writes == ["", "add citations"]
    assert "__interrupt__" in result


def test_revision_cap():
    agents = StubAgents(approve_after=99)
    run(agents)
    assert agents.reviews == MAX_REVISIONS + 1


def test_human_approve_ends_run():
    graph, config, _ = run(StubAgents(approve_after=1))
    final = graph.invoke(Command(resume="approve"), config)
    assert final["human_decision"] == "approve"
    assert "__interrupt__" not in final


def test_human_edit_goes_back_to_writer():
    agents = StubAgents(approve_after=1)
    graph, config, _ = run(agents)
    graph.invoke(Command(resume="make it shorter"), config)
    assert agents.writes[-1] == "make it shorter"

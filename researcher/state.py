import operator
from typing import Annotated, TypedDict


class Note(TypedDict):
    sub_question: str
    findings: str
    sources: list[str]


class ResearchState(TypedDict, total=False):
    question: str
    plan: list[str]
    # Reducer appends notes from parallel researcher branches instead of overwriting.
    notes: Annotated[list[Note], operator.add]
    draft: str
    feedback: str
    approved: bool
    revisions: int
    human_decision: str

"""LLM-backed agents. Each agent is a small object so the graph can run with stubs in tests."""

from collections.abc import Callable
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from researcher.state import Note

SearchFn = Callable[[str], list[dict]]


class Plan(BaseModel):
    sub_questions: list[str] = Field(description="3-5 focused, non-overlapping sub-questions")


class Review(BaseModel):
    approved: bool = Field(description="True when the report is accurate, complete and well cited")
    feedback: str = Field(description="Concrete fixes needed. Empty when approved.")


class Agents(Protocol):
    def plan(self, question: str) -> list[str]: ...
    def research(self, sub_question: str) -> Note: ...
    def write(self, question: str, notes: list[Note], feedback: str) -> str: ...
    def review(self, question: str, draft: str) -> Review: ...


class LLMAgents:
    def __init__(self, llm: BaseChatModel, search: SearchFn):
        self.llm = llm
        self.search = search

    def plan(self, question: str) -> list[str]:
        planner = self.llm.with_structured_output(Plan)
        result = planner.invoke(
            f"Break this research question into 3-5 sub-questions that together answer it:\n{question}"
        )
        return result.sub_questions[:5]

    def research(self, sub_question: str) -> Note:
        hits = self.search(sub_question)
        context = "\n\n".join(f"[{h['url']}]\n{h['content']}" for h in hits)
        summary = self.llm.invoke(
            "Summarize the facts that answer the question. Keep source URLs in brackets after each "
            f"fact. Skip anything not supported by the sources.\n\nQuestion: {sub_question}\n\n"
            f"Sources:\n{context}"
        ).content
        return {
            "sub_question": sub_question,
            "findings": summary,
            "sources": [h["url"] for h in hits],
        }

    def write(self, question: str, notes: list[Note], feedback: str) -> str:
        notes_text = "\n\n".join(f"## {n['sub_question']}\n{n['findings']}" for n in notes)
        fix = f"\n\nReviewer feedback to address:\n{feedback}" if feedback else ""
        return self.llm.invoke(
            "Write a concise markdown research report that answers the question. Use only the "
            f"notes. Cite URLs inline.\n\nQuestion: {question}\n\nNotes:\n{notes_text}{fix}"
        ).content

    def review(self, question: str, draft: str) -> Review:
        critic = self.llm.with_structured_output(Review)
        return critic.invoke(
            "You are a strict reviewer. Check the report for unsupported claims, missing citations "
            f"and gaps versus the question.\n\nQuestion: {question}\n\nReport:\n{draft}"
        )


def tavily_search(max_results: int = 4) -> SearchFn:
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=max_results)

    def run(query: str) -> list[dict]:
        return [
            {"url": r["url"], "content": r["content"]} for r in tool.invoke(query)["results"]
        ]

    return run

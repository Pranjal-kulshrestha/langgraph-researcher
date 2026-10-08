import argparse
import os
import uuid

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.types import Command

from researcher.agents import LLMAgents, tavily_search
from researcher.graph import build_graph


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    parser.add_argument("question")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "anthropic:claude-sonnet-5-5"))
    args = parser.parse_args()

    llm = init_chat_model(args.model)
    graph = build_graph(LLMAgents(llm, tavily_search()))
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"question": args.question}, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n===== DRAFT =====\n" + payload["draft"])
        if payload["critic_feedback"]:
            print("\n----- critic -----\n" + payload["critic_feedback"])
        decision = input("\nType 'approve' or give edit instructions: ")
        result = graph.invoke(Command(resume=decision), config)

    print("\n===== FINAL REPORT =====\n" + result["draft"])


if __name__ == "__main__":
    main()

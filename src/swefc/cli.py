import argparse
import asyncio
import os
from datetime import datetime

from swefc.agent.agent_factory import make_swefc_agent


def main():
    """SWEFC Command Line Interface"""
    parser = argparse.ArgumentParser(
        description="SWE Fast Context CLI",
    )

    parser.add_argument("--query", "-q", type=str, help="query to ask the agent")
    parser.add_argument(
        "--traj",
        "-t",
        type=str,
        help="agent trajectory file",
        default=f".swefc/trajectory_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.jsonl",
    )
    parser.add_argument("--max-turns", type=int, help="maximum number of turns", default=4)
    parser.add_argument("--verbose", action="store_true", help="whether to run in verbose mode")
    parser.add_argument("--citation", action="store_true", help="Only return the citations in the final answer")

    args = parser.parse_args()

    work_dir = os.getcwd()
    agent = make_swefc_agent(trajectory_file=args.traj, work_dir=work_dir)

    prompt = args.query
    final_output = asyncio.run(
        agent.run(prompt=prompt, max_turns=args.max_turns, verbose=args.verbose, citation=args.citation)
    )
    print(final_output)


if __name__ == "__main__":
    main()

"""
Test agent for Frontier Science environment.

Tests the environment locally using OpenAI's modern Responses API.
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ENV_NAME = "local/frontierscience"
BASE_URL = "http://localhost:8080"


async def test_olympiad_track():
    """Test an Olympiad track question (short answer)"""
    print("\n" + "=" * 80)
    print("TESTING OLYMPIAD TRACK")
    print("=" * 80 + "\n")

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to local environment
    environment = or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    # Select an Olympiad track question (short answer)
    # Task 51 is a chemistry question with answer "7"
    task = tasks[100]

    print(f"\nTesting task {task.task_spec['task_id']}")

    async with environment.session(
        task=task, secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        prompt = await session.get_prompt()

        print("\n--- PROMPT ---")
        print(prompt if isinstance(prompt, str) else prompt[0].text)
        print()

        input_list = [
            {"role": "user", "content": prompt if isinstance(prompt, str) else prompt[0].text}
        ]
        finished = False
        turn = 0

        while not finished and turn < 10:
            turn += 1
            print(f"\n--- TURN {turn} ---")

            response = await oai_client.responses.create(
                model=MODEL_NAME, tools=tools, input=input_list
            )

            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    print(f"Tool call: {item.name}")
                    print(f"Arguments: {item.arguments}")

                    tool_result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward

                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                            if tool_result.blocks
                            else "",
                        }
                    )

                    print(f"\nReward: {reward:.3f}")
                    print(f"Finished: {finished}")
                    print(f"\nFeedback:\n{tool_result.blocks[0].text if tool_result.blocks else ''}")

                    if tool_result.finished:
                        print("\n✅ Episode finished!")
                        break

                elif item.type == "text":
                    print(f"Model response: {item.text[:200]}...")


async def test_research_track():
    """Test a Research track question (rubric-based)"""
    print("\n" + "=" * 80)
    print("TESTING RESEARCH TRACK")
    print("=" * 80 + "\n")

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to local environment
    environment = or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    # Select a Research track question (long rubric-based)
    # Task 100 is a physics research question
    task = tasks[100]

    print(f"\nTesting task {task.task_spec['task_id']}")

    async with environment.session(
        task=task, secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        prompt = await session.get_prompt()

        print("\n--- PROMPT ---")
        print((prompt if isinstance(prompt, str) else prompt[0].text)[:500] + "...")
        print()

        input_list = [
            {"role": "user", "content": prompt if isinstance(prompt, str) else prompt[0].text}
        ]
        finished = False
        turn = 0

        while not finished and turn < 10:
            turn += 1
            print(f"\n--- TURN {turn} ---")

            response = await oai_client.responses.create(
                model=MODEL_NAME, tools=tools, input=input_list
            )

            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    print(f"Tool call: {item.name}")
                    print(f"Arguments: {item.arguments[:200]}...")

                    tool_result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward

                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                            if tool_result.blocks
                            else "",
                        }
                    )

                    print(f"\nReward: {reward:.3f}")
                    print(f"Finished: {finished}")
                    print(f"\nFeedback (truncated):\n{(tool_result.blocks[0].text if tool_result.blocks else '')[:500]}...")

                    if tool_result.finished:
                        print("\n✅ Episode finished!")
                        break

                elif item.type == "text":
                    print(f"Model response: {item.text[:200]}...")


async def main():
    """Run both test scenarios"""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("Frontier Science Test Agent")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_NAME}")
    print(f"Base URL: {BASE_URL}")

    try:
        # Test Olympiad track
        await test_olympiad_track()

        # Optional: Test Research track (commented out by default due to complexity)
        # await test_research_track()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

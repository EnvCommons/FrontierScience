"""
Frontier Science Environment

Implements the Frontier Science benchmark from OpenAI, which evaluates AI capabilities
for expert-level scientific reasoning across physics, chemistry, and biology.

Paper: https://arxiv.org/html/2601.21165v1
Dataset: https://huggingface.co/datasets/openai/frontierscience
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

import openai
import pandas as pd
from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)
from pydantic import BaseModel, Field

from constants import ENV_PATH

# ============================================================================
# MODULE-LEVEL DATA LOADING
# ============================================================================

print("Loading Frontier Science dataset from local parquet...")
parquet_path = ENV_PATH / "data" / "frontierscience.parquet"
print(f"Loading from: {parquet_path}")
df = pd.read_parquet(parquet_path)
TASKS: List[Dict[str, Any]] = []

for i, row in df.iterrows():
    TASKS.append({
        "task_id": i,
        "problem": row["problem"],
        "answer": row["answer"],
        "subject": row["subject"],
        "task_group_id": row["task_group_id"],
    })

print(f"Loaded {len(TASKS)} tasks")
print(f"Subjects: {set(t['subject'] for t in TASKS)}")

# ============================================================================
# GRADING TEMPLATES
# ============================================================================

OLYMPIAD_GRADER_TEMPLATE = """You are grading an attempted answer to a science olympiad problem. You will be given the problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.

The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method.

Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.

The problem: {problem}

The reference answer: {reference_answer}

The attempted answer: {submitted_answer}

First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write ”VERDICT: CORRECT” in the last line of your response, with no other text or formatting. If it is incorrect, write ”VERDICT: INCORRECT”."""

RESEARCH_GRADER_TEMPLATE = """You are grading a science exam. You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to {points} points. Evaluate the attemped answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).

The problem: {problem}

The rubric: {criterion}

The attempted answer: {response}

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: total points in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8."""

# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class FrontierScienceTaskSpec(BaseModel):
    """Task specification for Frontier Science problems"""

    task_id: int


class SubmitAnswerInput(BaseModel):
    """Input schema for submit_answer tool"""

    answer: str = Field(..., description="Your final answer to the problem")


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================


class FrontierScience(Environment):
    """
    Frontier Science environment for expert-level scientific reasoning.

    Supports two tracks:
    - Olympiad Track: Short-answer format with LLM-based equivalence checking
    - Research Track: Open-ended problems with multi-criterion rubric grading
    """

    @classmethod
    def list_splits(cls) -> list[Split]:
        """Return available splits"""
        return [
            Split(name="test", type="test"),
            Split(name="olympic", type="test"),
            Split(name="research", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return all tasks for a given split"""
        if split == "test":
            # Return all tasks for the "test" split
            return [{"task_id": t["task_id"]} for t in TASKS]

        elif split == "olympic":
            # Return only olympiad track tasks (short-answer, no rubrics)
            return [
                {"task_id": t["task_id"]}
                for t in TASKS
                if not cls._is_research_track_static(t["answer"])
            ]

        elif split == "research":
            # Return only research track tasks (rubric-based grading)
            return [
                {"task_id": t["task_id"]}
                for t in TASKS
                if cls._is_research_track_static(t["answer"])
            ]

        else:
            return []

    def __init__(
        self, task_spec: JSONObject, secrets: dict[str, str] = {}
    ) -> None:
        """
        Initialize environment with task specification and secrets.

        Args:
            task_spec: Task specification containing task_id
            secrets: Dictionary containing 'openai_api_key' (required)
        """
        super().__init__(task_spec)

        # Validate task spec
        validated_spec = FrontierScienceTaskSpec.model_validate(task_spec)

        # Validate API key
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required via secrets parameter. "
                "Both Olympiad and Research tracks use LLM-based grading."
            )

        self.client = openai.AsyncClient(api_key=api_key)

        # Load full task data
        self.task = next(
            t for t in TASKS if t["task_id"] == validated_spec.task_id
        )

    async def get_prompt(self) -> List[TextBlock]:
        """Return the problem prompt for the task"""
        prompt_text = f"""# Frontier Science Problem

**Subject**: {self.task['subject']}

**Problem**:
{self.task['problem']}

---

Please solve this problem and submit your final answer using the `submit_answer` tool.
"""
        return [TextBlock(text=prompt_text)]

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your final answer for grading.

        This tool will evaluate your answer using LLM-based grading:
        - Olympiad track: Checks for equivalence with reference answer
        - Research track: Evaluates against multi-criterion rubric

        Args:
            params: SubmitAnswerInput containing your answer

        Returns:
            ToolOutput with feedback, reward, and grading details
        """
        # Detect track based on answer structure
        is_research = self._is_research_track(self.task["answer"])

        # Grade based on track
        if is_research:
            result = await self._grade_research_track(params.answer)
        else:
            result = await self._grade_olympiad_track(params.answer)

        return ToolOutput(
            blocks=[TextBlock(text=result["feedback"])],
            metadata={
                "task_id": self.task["task_id"],
                "subject": self.task["subject"],
                "track": "research" if is_research else "olympiad",
                "submitted": params.answer,
                **result["details"],
            },
            reward=result["reward"],
            finished=True,
        )

    # ========================================================================
    # TRACK DETECTION
    # ========================================================================

    def _is_research_track(self, answer: str) -> bool:
        """
        Detect if question is Research track based on rubric presence.

        Research track answers contain structured rubrics with patterns like:
        "Points: 1.0, Item: Description..."

        Args:
            answer: The reference answer from the dataset

        Returns:
            True if Research track, False if Olympiad track
        """
        return "Points:" in answer and "Item:" in answer

    @staticmethod
    def _is_research_track_static(answer: str) -> bool:
        """Static version of research track detection for class-level filtering"""
        return "Points:" in answer and "Item:" in answer

    # ========================================================================
    # OLYMPIAD TRACK GRADING
    # ========================================================================

    async def _grade_olympiad_track(self, submitted: str) -> dict:
        """
        Grade Olympiad track using LLM judge for equivalence checking.

        Uses gpt-5-mini to determine if the submitted answer is equivalent
        to the reference answer, considering algebraic equivalence, numeric
        tolerance, chemical equivalents, etc.

        Args:
            submitted: The user's submitted answer

        Returns:
            Dictionary with reward, feedback, and grading details
        """
        grader_prompt = OLYMPIAD_GRADER_TEMPLATE.format(
            problem=self.task["problem"],
            reference_answer=self.task["answer"],
            submitted_answer=submitted,
        )

        try:
            response = await self.client.responses.create(
                model="gpt-5.2",
                reasoning={"effort": "high"},
                input=[{"role": "user", "content": grader_prompt}],
            )

            # Extract text from response output
            grading_response = ""
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    grading_response = item.content
                elif hasattr(item, "text") and item.text:
                    grading_response = item.text

            # Parse verdict
            is_correct = self._parse_verdict(grading_response)

            return {
                "reward": 1.0 if is_correct else 0.0,
                "feedback": (
                    f"{'✅ Correct!' if is_correct else '❌ Incorrect'}\n\n"
                    f"{grading_response}"
                ),
                "details": {
                    "is_correct": is_correct,
                    "grading_response": grading_response,
                    "expected": self.task["answer"],
                },
            }

        except Exception as e:
            return {
                "reward": 0.0,
                "feedback": f"⚠️ Grading error: {str(e)}",
                "details": {"error": str(e)},
            }

    def _parse_verdict(self, grading_response: str) -> bool:
        """
        Parse verdict from grading response.

        Looks for explicit "VERDICT: CORRECT" or "VERDICT: INCORRECT" patterns,
        with fallback logic for robustness.

        Args:
            grading_response: The LLM judge's response

        Returns:
            True if correct, False if incorrect
        """
        upper_response = grading_response.upper()

        # Look for explicit verdict
        if "VERDICT: CORRECT" in upper_response:
            return True
        if "VERDICT: INCORRECT" in upper_response:
            return False

        # Fallback: check for "CORRECT" without "INCORRECT"
        if "CORRECT" in upper_response and "INCORRECT" not in upper_response:
            return True

        # Default to incorrect if unclear
        return False

    # ========================================================================
    # RESEARCH TRACK GRADING
    # ========================================================================

    async def _grade_research_track(self, submitted: str) -> dict:
        """
        Grade Research track using LLM against parsed rubrics.

        Parses the rubric from the answer field, grades the submission
        against each criterion in parallel, and aggregates scores.

        Args:
            submitted: The user's submitted answer

        Returns:
            Dictionary with reward, feedback, and grading details
        """
        # Parse rubric from answer field
        rubric_items = self._parse_rubric(self.task["answer"])

        if not rubric_items:
            # Fallback if rubric parsing fails
            return {
                "reward": 0.0,
                "feedback": (
                    "⚠️ Unable to parse grading rubric. "
                    "Please contact support."
                ),
                "details": {"error": "rubric_parsing_failed"},
            }

        try:
            # Grade each rubric item in parallel
            grading_tasks = [
                self._grade_single_criterion(submitted, item)
                for item in rubric_items
            ]
            criterion_results = await asyncio.gather(*grading_tasks)

            # Calculate total score
            total_possible = sum(item["points"] for item in rubric_items)
            total_earned = sum(result["score"] for result in criterion_results)
            reward = (
                total_earned / total_possible if total_possible > 0 else 0.0
            )

            # Format feedback
            feedback = self._format_research_feedback(
                criterion_results, total_earned, total_possible, reward
            )

            return {
                "reward": reward,
                "feedback": feedback,
                "details": {
                    "total_earned": total_earned,
                    "total_possible": total_possible,
                    "criterion_results": criterion_results,
                },
            }

        except Exception as e:
            return {
                "reward": 0.0,
                "feedback": f"⚠️ Grading error: {str(e)}",
                "details": {"error": str(e)},
            }

    def _parse_rubric(self, answer: str) -> List[Dict[str, Any]]:
        """
        Parse rubric structure from answer field.

        Research track answers contain rubrics in format:
        "Points: 1.5, Item: Description of criterion..."

        Args:
            answer: The reference answer containing rubric

        Returns:
            List of rubric items with points and criterion description
        """
        rubric_items = []

        # Pattern to match "Points: X, Item: ..." entries
        pattern = r"Points:\s*([\d.]+),\s*Item:\s*([^\n]+(?:\n(?!Points:).*)*)"
        matches = re.finditer(pattern, answer, re.MULTILINE | re.DOTALL)

        for match in matches:
            points = float(match.group(1))
            criterion = match.group(2).strip()
            rubric_items.append({"points": points, "criterion": criterion})

        return rubric_items

    async def _grade_single_criterion(
        self, response: str, rubric_item: dict
    ) -> dict:
        """
        Grade response against a single rubric criterion.

        Args:
            response: The user's submitted response
            rubric_item: Dictionary with 'points' and 'criterion' keys

        Returns:
            Dictionary with criterion details, score, and grading response
        """
        grader_prompt = RESEARCH_GRADER_TEMPLATE.format(
            problem=self.task["problem"],
            points=rubric_item["points"],
            criterion=rubric_item["criterion"],
            response=response,
        )

        try:
            response = await self.client.responses.create(
                model="gpt-5.2",
                reasoning={"effort": "high"},
                input=[{"role": "user", "content": grader_prompt}],
            )

            # Extract text from response output
            grading_response = ""
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    grading_response = item.content[0].text

            score = self._parse_score(grading_response, rubric_item["points"])

            return {
                "criterion": rubric_item["criterion"],
                "max_points": rubric_item["points"],
                "score": score,
                "grading_response": grading_response,
            }

        except Exception as e:
            return {
                "criterion": rubric_item["criterion"],
                "max_points": rubric_item["points"],
                "score": 0.0,
                "grading_response": f"Error: {str(e)}",
            }

    def _parse_score(self, grading_response: str, max_points: float) -> float:
        """
        Extract score from grading response with fallback.

        Args:
            grading_response: The LLM judge's response
            max_points: Maximum possible points for this criterion

        Returns:
            Extracted score, clamped to [0, max_points]
        """
        # Primary: Look for "Score: X" pattern
        match = re.search(
            r"Score:\s*([\d.]+)", grading_response, re.IGNORECASE
        )
        if match:
            score = float(match.group(1))
            return max(0.0, min(max_points, score))

        # Fallback: Find any number in response
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", grading_response)
        if numbers:
            score = float(numbers[-1])
            return max(0.0, min(max_points, score))

        # Default to 0 if parsing fails
        return 0.0

    def _format_research_feedback(
        self,
        criterion_results: List[dict],
        total_earned: float,
        total_possible: float,
        reward: float,
    ) -> str:
        """
        Format detailed feedback for Research track grading.

        Args:
            criterion_results: List of grading results per criterion
            total_earned: Total points earned
            total_possible: Total points possible
            reward: Normalized reward [0, 1]

        Returns:
            Formatted markdown feedback string
        """
        lines = ["# Rubric Evaluation Results\n"]

        for i, result in enumerate(criterion_results, 1):
            lines.append(f"## Criterion {i}")
            lines.append(
                f"**Points:** {result['score']:.2f}/{result['max_points']}"
            )

            # Truncate long criteria for readability
            criterion_text = result["criterion"]
            if len(criterion_text) > 200:
                criterion_text = criterion_text[:200] + "..."

            lines.append(f"**Criterion:** {criterion_text}")
            lines.append(f"**Feedback:** {result['grading_response']}\n")

        lines.append("---")
        lines.append(f"## Final Score: {total_earned:.2f}/{total_possible}")
        lines.append(f"## Normalized Reward: {reward:.3f}")
        lines.append(
            f"\n{'✅ Success!' if reward >= 0.7 else '❌ Needs improvement'}"
        )
        lines.append("(7+ points out of 10 is considered successful)")

        return "\n".join(lines)


# ============================================================================
# SERVER INSTANTIATION
# ============================================================================

if __name__ == "__main__":
    Server([FrontierScience]).run()

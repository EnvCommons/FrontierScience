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

First, think step-by-step about whether the attempted answer matches the reference answer. Then give your verdict in the last line of your response, wrapped in answer tags and containing nothing else: <answer>CORRECT</answer> if the attempted answer is correct, or <answer>INCORRECT</answer> if it is not."""

RESEARCH_GRADER_TEMPLATE = """You are grading a science exam. You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to {points} points. Evaluate the attemped answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).

The problem: {problem}

The rubric: {criterion}

The attempted answer: {response}

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and give the total in the last line of your response, wrapped in answer tags and containing nothing else, like this: <answer>total points earned</answer>"""

# Verdict extraction. Both graders are asked to wrap their verdict in
# <answer></answer>, which confines extraction to a delimited span instead of
# scanning prose for a loose token — a bare number or keyword anywhere in the
# response used to be fair game, and these graders quote the instructions back
# inside their reasoning. Neither template contains a worked numeric example,
# so an echoed instruction cannot supply a parseable value.
#
# re.DOTALL is safe here (unlike in _parse_rubric): `.*?` is non-greedy AND
# bounded by the closing tag, so it cannot run away to end of input.
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_NUM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

# Uncapped high-effort grader calls ran for up to an hour in prod.
GRADER_MAX_OUTPUT_TOKENS = 16384
# Retries don't cancel the abandoned upstream generation, so keep them rare.
GRADER_TIMEOUT_S = 1200
GRADER_MAX_RETRIES = 1
# Hitting the output cap is usually a one-off ramble, so resample.
GRADER_MAX_ATTEMPTS = 3

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

        self.client = openai.AsyncClient(
            api_key=api_key,
            timeout=GRADER_TIMEOUT_S,
            max_retries=GRADER_MAX_RETRIES,
        )

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

    async def _call_grader(self, grader_prompt: str) -> str:
        """Run one grader call and return its message text."""
        for attempt in range(1, GRADER_MAX_ATTEMPTS + 1):
            response = await self.client.responses.create(
                model="gpt-5.2",
                reasoning={"effort": "high"},
                max_output_tokens=GRADER_MAX_OUTPUT_TOKENS,
                input=[{"role": "user", "content": grader_prompt}],
            )
            if getattr(response, "status", None) != "incomplete":
                return self._response_text(response)
            reason = getattr(
                getattr(response, "incomplete_details", None), "reason", None
            )
            print(
                f"grader response incomplete ({reason}), "
                f"attempt {attempt}/{GRADER_MAX_ATTEMPTS}"
            )
        raise RuntimeError(
            f"grader response incomplete ({reason}) after "
            f"{GRADER_MAX_ATTEMPTS} attempts; cannot grade a truncated verdict"
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract the grader's message text from a Responses API result.

        Prefers `output_text` (the concatenated assistant text, excluding
        reasoning items) and falls back to walking `output`. An empty result
        means the grader returned nothing usable, which is an env-side failure
        rather than a zero-scoring answer, so it raises.
        """
        text = (getattr(response, "output_text", "") or "").strip()
        if not text:
            for item in getattr(response, "output", []) or []:
                content = getattr(item, "content", None)
                if content:
                    if isinstance(content, list):
                        text = content[0].text if content else ""
                    else:
                        text = content
                elif getattr(item, "text", None):
                    text = item.text
            text = (text or "").strip()
        if not text:
            raise RuntimeError("grader returned an empty response")
        return text

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

        # No try/except: a grader API failure is an env crash, not a wrong
        # answer. Returning reward=0.0 here would be indistinguishable from a
        # graded-incorrect submission and would poison training data silently.
        grading_response = await self._call_grader(grader_prompt)

        # Parse verdict
        is_correct = self._parse_verdict(grading_response)

        # Only the verdict goes back. The whole ToolOutput — blocks AND
        # metadata — reaches the model, so the reference answer, or the
        # grader's reasoning (which quotes it), would give the model the
        # answer to copy into a resubmission.
        return {
            "reward": 1.0 if is_correct else 0.0,
            "feedback": "✅ Correct!" if is_correct else "❌ Incorrect",
            "details": {
                "is_correct": is_correct,
            },
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
        # Take the LAST answer tag: the grader's reasoning may quote the
        # instruction or state an interim verdict before revising it, and a
        # first-match scan would lock in the wrong one.
        tags = _ANSWER_TAG_RE.findall(grading_response)
        if tags:
            verdict = tags[-1].strip().upper()
            # INCORRECT contains CORRECT, so test it first.
            if "INCORRECT" in verdict:
                return False
            if "CORRECT" in verdict:
                return True

        # No usable verdict. The old code guessed from a bare "CORRECT"
        # substring anywhere in the response and otherwise defaulted to
        # incorrect — both fabricate a grade the judge never gave, and the
        # default silently converts a broken grader into a 0.0 reward.
        raise RuntimeError(
            "grader response contained no <answer>CORRECT|INCORRECT</answer> — "
            f"cannot grade submission. Response tail: {grading_response[-500:]!r}"
        )

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
        # Parse rubric from answer field. _parse_rubric raises on a mis-parse;
        # an empty rubric on a task routed here (i.e. one whose answer contains
        # "Points:" and "Item:") means the dataset row is malformed.
        rubric_items = self._parse_rubric(self.task["answer"])
        if not rubric_items:
            raise RuntimeError(
                f"task {self.task['task_id']} routed to the research track but "
                f"no rubric items parsed from its answer field"
            )

        # No try/except: grader failures propagate. Swallowing them into
        # reward=0.0 makes an env crash look like a submission that earned
        # nothing, which is the one outcome a reward channel must never fake.
        criterion_results = await asyncio.gather(*[
            self._grade_single_criterion(submitted, item)
            for item in rubric_items
        ])

        # Calculate total score
        total_possible = sum(item["points"] for item in rubric_items)
        total_earned = sum(result["score"] for result in criterion_results)
        if total_possible <= 0:
            raise RuntimeError(
                f"task {self.task['task_id']} rubric totals {total_possible} "
                f"points; cannot normalise a reward"
            )
        reward = total_earned / total_possible

        # Format feedback
        feedback = self._format_research_feedback(
            criterion_results, total_earned, total_possible, reward
        )

        # Scores only: the criterion text and the grader's per-criterion
        # reasoning spell out the rubric, i.e. the reference solution, and the
        # whole ToolOutput (metadata included) reaches the model.
        return {
            "reward": reward,
            "feedback": feedback,
            "details": {
                "total_earned": total_earned,
                "total_possible": total_possible,
                "criterion_scores": [r["score"] for r in criterion_results],
            },
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

        # Pattern to match "Points: X, Item: ..." entries. An item's body runs
        # to the next line that starts a new "Points:" entry, which is what the
        # `(?!Points:)` guard is for — so this must NOT be compiled with
        # re.DOTALL. Under DOTALL the `.*` spans newlines, the first item's
        # continuation line swallows the entire remaining rubric, and a 10-item
        # rubric parses as ONE criterion worth 1.0 point (destroying the reward
        # scale: see the invariant below).
        pattern = r"Points:\s*([\d.]+),\s*Item:\s*([^\n]+(?:\n(?!Points:).*)*)"
        matches = re.finditer(pattern, answer, re.MULTILINE)

        for match in matches:
            points = float(match.group(1))
            criterion = match.group(2).strip()
            rubric_items.append({"points": points, "criterion": criterion})

        # Every "Points: X, Item:" header in the answer must have produced an
        # item. A mismatch means items were merged or dropped, which silently
        # rescales total_possible and corrupts every reward for this task, so
        # fail loudly instead of grading against a wrong denominator.
        expected = len(re.findall(r"Points:\s*[\d.]+,\s*Item:", answer))
        if len(rubric_items) != expected:
            raise RuntimeError(
                f"rubric parse produced {len(rubric_items)} item(s) but the "
                f"answer contains {expected} 'Points: X, Item:' header(s); "
                f"refusing to grade against a mis-parsed rubric"
            )

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

        # No try/except: a failed criterion grade must not silently become 0.0
        # points. One swallowed API error used to drop the whole submission's
        # score by a criterion with no trace in the reward.
        grading_response = await self._call_grader(grader_prompt)
        score = self._parse_score(grading_response, rubric_item["points"])

        return {
            "criterion": rubric_item["criterion"],
            "max_points": rubric_item["points"],
            "score": score,
            "grading_response": grading_response,
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
        # RESEARCH_GRADER_TEMPLATE asks for the tally in <answer></answer> on
        # the last line. Take the LAST tag: the grader's reasoning quotes the
        # instruction back and may state an interim tally before revising it.
        tags = _ANSWER_TAG_RE.findall(grading_response)
        if not tags:
            # No answer tag means the grader did not answer the question we
            # asked. Scoring that as 0.0 is indistinguishable from a genuinely
            # worthless answer, so raise instead of inventing a grade.
            raise RuntimeError(
                "grader response contained no <answer></answer> tag — cannot "
                f"score criterion. Response tail: {grading_response[-500:]!r}"
            )

        # Tolerate decoration inside the tag ("4.0 points", "**3**") — the tag
        # already bounds where we look, which is what makes this safe.
        number = _ANSWER_NUM_RE.search(tags[-1])
        if not number:
            raise RuntimeError(
                "grader's <answer> tag contained no number — cannot score "
                f"criterion. Tag contents: {tags[-1][:200]!r}"
            )

        return max(0.0, min(max_points, float(number.group(1))))

    def _format_research_feedback(
        self,
        criterion_results: List[dict],
        total_earned: float,
        total_possible: float,
        reward: float,
    ) -> str:
        """
        Format score-only feedback for Research track grading.

        Per-criterion points only: the criterion text and the grader's
        reasoning would reveal the rubric (the reference solution).

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
            lines.append(
                f"- Criterion {i}: {result['score']:.2f}/{result['max_points']}"
            )

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

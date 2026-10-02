# FrontierScience

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/FrontierScience) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/openai/frontierscience)

## Description

FrontierScience is an environment for evaluating expert-level scientific reasoning capabilities. It contains 160 expert-level problems designed to assess frontier model capabilities in physics, chemistry, and biology through two distinct evaluation tracks: Olympiad (short-answer format) and Research (open-ended PhD-level problems).

## Capabilities

- Expert-level scientific reasoning across physics, chemistry, and biology
- Short-answer problem solving (Olympiad track)
- Open-ended research subtask completion (Research track)
- Multi-criterion rubric-based evaluation

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[Apache 2.0](https://opensource.org/licenses/Apache-2.0).

## Tasks

There are three splits in this environment:

- **test**: 160 tasks (all problems)
- **olympic**: ~92 tasks (short-answer Olympiad-style problems)
- **research**: ~68 tasks (open-ended PhD-level research problems)

Problems span Physics (70), Chemistry (60), and Biology (30).

## Reward Structure

This is a single-turn environment with two grading methodologies:

**Olympiad Track**: An LLM grader (gpt-5.2 with high reasoning effort) checks equivalence with the reference answer, considering algebraic equivalence, numeric tolerance, chemical equivalents, and unit conversions. Reward is binary: 1.0 if correct, 0.0 if incorrect.

**Research Track**: An LLM grader (gpt-5.2 with high reasoning effort) scores the answer against the task's whole rubric in a single call, as in the paper, and reports the total points earned. Reward is normalized (total earned / total possible). Success threshold is 7+ points out of 10 (0.7 reward).

## Data

Data consists of a Parquet file (`frontierscience.parquet`) sourced from [HuggingFace openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience). Each row contains a problem, answer (short answer or rubric), subject, and task group ID. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `submit_answer` | Submit your final answer for grading. Ends the episode. |

`submit_answer` returns only the verdict (Olympiad) or the total score (Research). The reference answer, the rubric text and the grader's reasoning are never returned to the agent.

## Time Horizon

Single-turn. The agent reads the scientific problem and submits one answer.

## Environment Difficulty

FrontierScience evaluates expert-level scientific reasoning designed to challenge frontier AI systems.

| Track | Model | Pass Rate |
|-------|-------|-----------|
| Olympiad | GPT-5.2 | 77% |
| Olympiad | Gemini 3 Pro | 76% |
| Research | GPT-5.2 | 25% |
| Research | GPT-5 | 25% |

## Other Environment Requirements

OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.

## Safety

Agents in FrontierScience solve expert-level scientific problems in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@article{frontierscience2025,
  title={FrontierScience: Measuring Expert-Level Scientific Reasoning in AI},
  author={OpenAI},
  journal={arXiv preprint arXiv:2601.21165},
  year={2025}
}
```

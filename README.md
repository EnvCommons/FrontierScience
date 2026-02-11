# Frontier Science Environment

An OpenReward environment implementing the **Frontier Science** benchmark from OpenAI, which evaluates AI capabilities for expert-level scientific reasoning across physics, chemistry, and biology.

## Overview

Frontier Science is a challenging benchmark containing 160 expert-level scientific problems designed to assess frontier model capabilities in scientific reasoning. The benchmark features two distinct evaluation tracks:

### Olympiad Track (~92 questions)
- **Format**: Short-answer format similar to international science olympiad problems
- **Subjects**: Physics, Chemistry, Biology
- **Answers**: Single numeric expressions, algebraic formulas, or chemical identifiers
- **Grading**: LLM-based equivalence checking (considers algebraic equivalence, numeric tolerance, chemical equivalents)

### Research Track (~68 questions)
- **Format**: PhD-level open-ended problems representing authentic research subtasks
- **Complexity**: Problems estimated to require 3-5 hours for expert completion
- **Grading**: Multi-criterion rubric-based evaluation (10-point scale, 7+ points = success)
- **Evaluation**: Assesses intermediate reasoning steps, not just final answers

## Dataset

- **Source**: [openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience) on HuggingFace
- **Size**: 160 questions (test split only)
- **Subjects**: Physics (70), Chemistry (60), Biology (30)
- **License**: Apache 2.0
- **Local Storage**: Data is stored as a parquet file for fast loading (see `DATA_UPLOAD.md` for deployment details)

## Installation

### Requirements

- Python 3.11+
- OpenAI API key (required for LLM-based grading)

### Data Setup

Before running the environment, you need to download the dataset:

```bash
# Install dependencies (including datasets library for one-time download)
pip install -r requirements.txt

# Download the dataset from HuggingFace and save as parquet
python download_dataset.py
```

This will create `data/frontierscience.parquet` with 160 tasks (~150-200 KB).

**Note**: The download script only needs to be run once. The parquet file will be used for all subsequent runs.

### Local Setup

```bash
# Install dependencies (if not already done above)
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the server
python server.py
```

The server will start on `http://0.0.0.0:8080`.

### Docker Setup

```bash
# Build the image
docker build -t frontierscience:latest .

# Run the container (mount local data directory)
docker run -p 8080:8080 \
  -v $(pwd)/data:/orwd_data/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  frontierscience:latest
```

**Note**: The `-v` flag mounts your local `data/` directory into the container at `/orwd_data/data/` so the server can access the parquet file.

## Usage

### Testing with OpenAI Agent

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run the test agent
python test_agent.py
```

The test agent will:
1. Connect to the local environment server
2. Load tasks from the test split
3. Run an agent on an Olympiad track question
4. Display the grading results

### Using with OpenReward SDK

```python
import asyncio
from openreward import AsyncOpenReward

async def main():
    client = AsyncOpenReward()

    # Connect to environment
    env = client.environments.get(name="EnvCommons/frontierscience")

    # List tasks
    tasks = await env.list_tasks(split="test")
    print(f"Found {len(tasks)} tasks")

    # Create a session
    async with env.session(
        task=tasks[0],
        secrets={"openai_api_key": "your-key"}
    ) as session:
        # Get the problem prompt
        prompt = await session.get_prompt()
        print(prompt)

        # Submit an answer
        result = await session.call_tool(
            "submit_answer",
            {"answer": "your answer here"}
        )

        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.blocks[0].text}")

asyncio.run(main())
```

## Grading Methodology

### Olympiad Track Grading

The Olympiad track uses **gpt-5.2** (with high reasoning effort) as a judge to determine answer equivalence:

1. Judge receives: problem + reference answer + submitted answer
2. Judge evaluates equivalence considering:
   - Algebraic expressions that simplify to the same result
   - Numerical values within 1 decimal place of rounding
   - Equivalent chemical compound names or formulas
   - Equivalent representations when accounting for units
3. Judge outputs: `VERDICT: CORRECT` or `VERDICT: INCORRECT`
4. Reward: 1.0 for correct, 0.0 for incorrect

### Research Track Grading

The Research track uses **gpt-5.2** (with high reasoning effort) for multi-criterion rubric evaluation:

1. Rubric is automatically parsed from the dataset's answer field
2. Each rubric criterion is evaluated independently in parallel
3. Judge provides:
   - Analysis of how well the response meets the criterion
   - Score from 0 to max_points for that criterion
4. Scores are aggregated across all criteria
5. Reward: normalized total score / total possible (0 to 1)
6. Success threshold: 7+ points out of 10 (0.7 reward)

### Implementation Notes

**Paper's Methodology** (reference baseline):
- Uses GPT-5 with high reasoning effort
- Averages across 20 trials (Olympiad) or 30 trials (Research)
- Production-grade benchmark evaluation

**Our Implementation** (practical for interactive use):
- Uses gpt-5.2 with high reasoning effort (via Responses API)
- Single trial per submission (cost and latency optimization)
- Suitable for agent development and interactive evaluation

**Note**: Scores may differ slightly from paper's reported baselines due to model and methodology differences.

## Environment Details

### Tools

**submit_answer**
- **Description**: Submit your final answer for grading
- **Parameters**:
  - `answer` (string): Your final answer to the problem
- **Returns**: ToolOutput with feedback, reward [0, 1], and grading details
- **Behavior**: Finishes episode after submission (single-turn)

### Secrets

Required secrets:
- `openai_api_key`: OpenAI API key for LLM-based grading (both tracks)

### Task Specification

Each task includes:
- `task_id`: Unique integer identifier (0-159)
- `problem`: The scientific problem statement
- `answer`: Reference answer (short answer for Olympiad, rubric for Research)
- `subject`: Physics, Chemistry, or Biology
- `task_group_id`: UUID for task grouping

## Performance Baselines

From the original paper (using GPT-5 with high reasoning effort):

**Olympiad Track**:
- GPT-5.2: 77%
- Gemini 3 Pro: 76%

**Research Track**:
- GPT-5.2: 25%
- GPT-5: 25%

These baselines demonstrate substantial progress on constrained reasoning while leaving significant room for improvement on open-ended research tasks.

## Development

### Running Tests

```bash
# Syntax check
python -m py_compile server.py

# Run local server
python server.py

# Test with agent (requires OPENAI_API_KEY)
python test_agent.py
```

### Docker Testing

```bash
# Build
docker build -t frontierscience:test .

# Run (with data volume mount)
docker run -p 8080:8080 \
  -v $(pwd)/data:/orwd_data/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  frontierscience:test

# Test against Docker container
python test_agent.py
```

## Citation

If you use this environment in your research, please cite the original Frontier Science paper:

```bibtex
@article{frontierscience2025,
  title={FrontierScience: Measuring Expert-Level Scientific Reasoning in AI},
  author={OpenAI},
  journal={arXiv preprint arXiv:2601.21165},
  year={2025}
}
```

## References

- **Paper**: [FrontierScience: Measuring Expert-Level Scientific Reasoning in AI](https://arxiv.org/html/2601.21165v1)
- **Dataset**: [openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience)
- **OpenReward**: [https://docs.openreward.org/](https://docs.openreward.org/)

## License

This implementation is provided under the Apache 2.0 license, consistent with the dataset license.

## Contributing

This environment is part of the EnvCommons organization. Contributions and feedback are welcome via issues and pull requests.

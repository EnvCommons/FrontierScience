# Data Upload Requirements for FrontierScience

## Overview
This environment requires the FrontierScience dataset to be uploaded to OpenReward cloud storage for ORS deployment.

## Directory Structure
The following structure should be created in your OpenReward namespace storage:

```
/orwd_data/
└── data/
    └── frontierscience.parquet
```

## File Description

### frontierscience.parquet
- **Size**: ~150-200 KB
- **Format**: Apache Parquet
- **Contents**: 160 expert-level science problems
  - **Olympiad Track**: 92 short-answer problems
  - **Research Track**: 68 rubric-based problems
  - **Subjects**:
    - Physics: 70 tasks
    - Chemistry: 60 tasks
    - Biology: 30 tasks

## Data Source

The dataset originates from:
- **HuggingFace**: https://huggingface.co/datasets/openai/frontierscience
- **Paper**: https://arxiv.org/html/2601.21165v1

## How to Generate the Parquet File

The parquet file can be generated using the included `download_dataset.py` script:

```bash
# Install dependencies (including datasets library)
pip install -r requirements.txt

# Run the download script
python download_dataset.py
```

This will create `data/frontierscience.parquet` in the local directory.

## Local Testing vs Production

### Local Development
When running locally, the environment will automatically use the local parquet file:
```
./data/frontierscience.parquet
```

### ORS Production Deployment
When deployed to ORS, the environment will automatically look for the file at:
```
/orwd_data/data/frontierscience.parquet
```

The path resolution is handled automatically by `constants.py`.

## Upload Instructions

1. **Generate the parquet file locally**:
   ```bash
   python download_dataset.py
   ```

2. **Upload to OpenReward namespace**:
   - Navigate to https://openreward.ai
   - Go to your namespace storage
   - Create the directory structure: `data/`
   - Upload `frontierscience.parquet` to this location

3. **Verify deployment**:
   - Deploy your environment to ORS
   - Check server logs for successful data loading
   - Run a test task to verify functionality

## Data Schema

The parquet file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `problem` | string | The scientific problem statement (markdown format) |
| `answer` | string | Reference answer (short answer for Olympiad, rubric for Research) |
| `subject` | string | Subject area (Physics, Chemistry, or Biology) |
| `task_group_id` | string | Identifier grouping related tasks |

## Security Notes

- The dataset is publicly available from HuggingFace
- No sensitive or private data is included
- Read-only access is sufficient for the environment
- Dataset is cached locally after first download

## Support

If you encounter issues with data upload or loading:
1. Verify the parquet file was generated correctly
2. Check the directory structure matches exactly
3. Confirm file permissions allow read access
4. Review server logs for path resolution errors

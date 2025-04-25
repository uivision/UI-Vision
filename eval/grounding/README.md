# Grounding Evaluation

This directory contains code and resources for evaluating model grounding capabilities, specifically focusing on UI vision element and layout grounding tasks.

## Directory Structure

```
eval/grounding/
├── README.md
├── eval_uivision.py   # Main evaluation script for UI vision tasks
├── models/            # Model implementations directory
│   ├── cogagent.py
│   ├── uitars.py
│   ├── seeclick.py
│   ├── qwen2vl.py
│   └── ... (other model implementations)
└── sample_script.sh         # Sample script with run commands
```
## Overview

This evaluation framework tests various vision-language models' abilities to understand and interact with UI elements in images. It supports two main types of evaluations:

1. **Element Grounding**: Tests the model's ability to locate and identify specific UI elements in images
2. **Layout Grounding**: Tests the model's ability to understand and predict UI regions

## Running Evaluations

The main evaluation script `eval_uivision.py` accepts the following arguments:

    --model_type MODEL_TYPE        # The type of model to evaluate (e.g., cogagent, uitars, gpt4v)
    --model_name_or_path PATH      # Optional: Path to model weights if required
    --uivision_imgs IMAGES_DIR     # Directory containing the UI test images
    --uivision_test_file TEST_FILE # JSON file containing test cases
    --task TASK_TYPE               # Either 'element' or 'layout'
    --log_path OUTPUT_PATH         # Where to save the evaluation results

## Supported Models

The framework currently supports these models:
- CogAgent
- UI-TARS (7B and 72B variants)
- SeeClick
- Qwen VL (2.0 and 2.5)
- MiniCPM-V
- InternVL
- GPT-4V
- Claude (3.5 and 3.7)
- Gemini
- OSAtlas (4B and 7B)
- UGround
- ShowUI
- AriaUI
- And more...

## How to run the evaluation

Please refer to the `sample_script.sh` for the commands to run the evaluation.
You can change the model type and the test file to evaluate different models on different datasets.
Please keep in mind to set the correct image directory and test file.

## Output Format

Results are saved in JSON format with the following structure:

    {
        "model_type": "model_name",
        "task_type": "element|layout",
        "results": {
            "details": [
                // Detailed results for each test case
            ],
            "metrics": {
                // Task-specific evaluation metrics
            },
            "input_files": "evaluation file name",
        },
    }

## Adding New Models

To add support for a new model:

1. Create a new model file in the `models/` directory
2. Implement the required interface methods:
   - `ground_only_positive()` for element evaluation
   - `layout_gen()` for layout evaluation
3. Add the model to the `build_model()` function in `eval_uivision.py`

Please refer to different models in the `models/` directory to understand the required implementation.
## Acknowledgments

This code is adapted from [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding). We thank the original authors for making their code available.


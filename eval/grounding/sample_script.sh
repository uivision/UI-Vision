#!/bin/bash

# Script to evaluate UI element grounding on UIVision dataset for basic element grounding
python eval_uivision.py \
    --model_type uitars \
    --uivision_imgs ./uivision_imgs \
    --uivision_test_file ./uivision_annotations/element_grounding/element_grounding_basic.json \
    --task element \
    --log_path results/save_results_basic.json

# Script to evaluate UI element grounding on UIVision dataset for functional element grounding
python eval_uivision.py \
    --model_type uitars \
    --uivision_imgs ./uivision_imgs \
    --uivision_test_file ./uivision_annotations/element_grounding/element_grounding_functional.json \
    --task element \
    --log_path results/save_results_functional.json

# Script to evaluate UI element grounding on UIVision dataset for spatial element grounding
python eval_uivision.py \
    --model_type uitars \
    --uivision_imgs ./uivision_imgs \
    --uivision_test_file ./uivision_annotations/element_grounding/element_grounding_spatial.json \
    --task element \
    --log_path results/save_results_spatial.json

# Script to evaluate UI layout grounding on UIVision dataset
python eval_uivision.py \
    --model_type qwen2vl \
    --uivision_imgs ./uivision_imgs \
    --uivision_test_file ./uivision_annotations/layout_grounding/layout_grounding.json \
    --task layout \
    --log_path results/save_results_layout_qwen.json
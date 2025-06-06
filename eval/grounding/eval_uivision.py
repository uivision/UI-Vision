import copy
import numpy as np
import torch
import json
import argparse
import os
from tqdm import tqdm
import logging

# os.environ['HF_HOME'] = '~/.cache/huggingface'

# For loading some models HF token needs to be set or you need to be logged in to HF
# os.environ["HF_TOKEN"] = "<HF_TOKEN>"

ANNOTATOR_IMAGE_SIZE = (800, 700)

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)
"""
    
python eval_uivision.py --model_type tongui --uivision_imgs ../../data/UI-Vision/images --uivision_test_file ../../data/UI-Vision/annotations/element_grounding/element_grounding_basic.json --task element --log_path ./element_grounding_basic_tongui.json
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--uivision_imgs', type=str, required=True)
    parser.add_argument('--uivision_test_file', type=str, required=True,
                        help="Path to the test file containing the evaluation data")
    parser.add_argument('--task', type=str, required=True, 
                        choices=['element', 'layout'],
                        help="Task type: 'element' or 'layout'")
    parser.add_argument('--log_path', type=str, required=True)

    args = parser.parse_args()
    return args

def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "cogagent":
        from models.cogagent import CogAgentModel
        model = CogAgentModel()
        model.load_model()
    elif model_type == "uitars": # 7b
        from models.uitars import UITARSModel
        model = UITARSModel()
        model.load_model()
    elif model_type == "uitars_72b":
        from models.uitars import UITARSModel
        model = UITARSModel()
        model.load_model(model_name_or_path="bytedance-research/UI-TARS-72B-DPO")
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel
        model = SeeClickModel()
        model.load_model()
    elif model_type == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel
        model = Qwen2VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen25vl":
        from models.qwen25vl import Qwen25VLModel
        model = Qwen25VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel
        model = MiniCPMVModel()
        model.load_model()
    elif model_type == "internvl":
        from models.internvl import InternVLModel
        model = InternVLModel()
        model.load_model()
    elif model_type == "internvl25_7b":
        from models.internvl25 import InternVLModel
        model = InternVLModel()
        model.load_model()
    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import ProprietaryModel
        model = ProprietaryModel(model_name="gpt-4o")
    elif model_type == "claude":
        from models.claude import claudeModel
        model = claudeModel(model_name="anthropic/claude-3.5-sonnet")
    elif model_type == "claude_37":
        from models.claude import claudeModel
        model = claudeModel(model_name="anthropic/claude-3.7-sonnet")
    elif model_type == "gemini":
        from models.gpt4x import ProprietaryModel
        model = ProprietaryModel(model_name="google/gemini-pro-1.5")
    elif model_type == "gemini_2":
        from models.gpt4x import ProprietaryModel
        model = ProprietaryModel(model_name="google/gemini-2.0-flash-001")
    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel
        model = OSAtlas4BModel()
        model.load_model()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel
        model = OSAtlas7BModel()
        model.load_model()
    elif model_type == "uground":
        from models.uground import UGroundModel
        model = UGroundModel()
        model.load_model()
    elif model_type == "uground-v1":
        from models.uground import UGroundV1Model
        model = UGroundV1Model(model_name_or_path="osunlp/UGround-V1-7B")
    elif model_type == "uground-v1_72b":
        from models.uground import UGroundV1Model
        model = UGroundV1Model(model_name_or_path="osunlp/UGround-V1-72B")
    elif model_type == "showui":
        from models.showui import ShowUIModel
        model = ShowUIModel()
        model.load_model()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIModel
        model = AriaUIModel()
        model.load_model()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model
        model = CogAgent24Model()
        model.load_model()
    elif model_type == "aguvis":
        from models.aguvis import AguvisModel
        model = AguvisModel()
        model.load_model()
    elif model_type == "tongui":
        from models.tongui import TongUIModel
        model = TongUIModel()
        model.load_model(model_name_or_path=model_name_or_path)
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model

# Element evaluation functions
def calc_metric_for_result_list_simple(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    return {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
    }

def evaluate_overall_simple(results):
    metrics = calc_metric_for_result_list_simple(results)
    return metrics

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    if bbox[0] > bbox[2]:
        bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[1] > bbox[3]:
        bbox[1], bbox[3] = bbox[3], bbox[1]
    img_size = sample["image_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    try:
        click_point = response["point"]  # may be none
    except:
        click_point = None
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def evaluate_element(results):
    """Collect results and calculate metrics for element evaluation."""
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    result_report["metrics"]["overall_simple"] = evaluate_overall_simple(results)
    result_report["details"] = results

    return result_report

def evaluate_single_bbox(pred_bbox, gt_bbox):
    """
    Evaluate a single predicted bounding box against a single ground truth bounding box.

    Args:
        pred_bbox (tuple): Predicted bounding box (x_min, y_min, x_max, y_max).
        gt_bbox (tuple): Ground truth bounding box (x_min, y_min, x_max, y_max).

    Returns:
        dict: A dictionary containing IoU, precision, and recall.
    """

    # Calculate intersection area and union area between two bounding boxes
    def calculate_intersection_and_union(box1, box2):
        """Calculate intersection area and union area between two bounding boxes."""
        if not box1:
            return 0, 0, 0

        # check if the boxes are valid
        if box1[0] >= box1[2] or box1[1] >= box1[3]:
            return 0, 0, 0

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Areas of individual boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection, box1_area, box2_area

    # Calculate intersection, predicted area, and ground truth area
    intersection, pred_area, gt_area = calculate_intersection_and_union(pred_bbox, gt_bbox)

    # Calculate IoU
    union = pred_area + gt_area - intersection
    iou = intersection / union if union > 0 else 0

    # Calculate precision and recall
    precision = intersection / gt_area if gt_area > 0 else 0
    recall = intersection / pred_area if pred_area > 0 else 0

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "intersection": intersection,
        "pred_area": pred_area,
        "gt_area": gt_area,
    }

def eval_layout(gt_bbox, pred_bbox):
    res = evaluate_single_bbox(pred_bbox, gt_bbox)
    return res

def evaluate_layout(results):
    """Calculate metrics based on layout evaluation results."""
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    iou_list = []
    precision_list = []
    recall_list = []
    for sample in results:
        iou_list.append(sample["eval_results"]["iou"])
        precision_list.append(sample["eval_results"]["precision"])
        recall_list.append(sample["eval_results"]["recall"])
    
    result_report["metrics"]["mean_iou"] = np.mean(iou_list)
    result_report["metrics"]["mean_precision"] = np.mean(precision_list)
    result_report["metrics"]["mean_recall"] = np.mean(recall_list)
    result_report["metrics"]["std_iou"] = np.std(iou_list)
    result_report["metrics"]["std_precision"] = np.std(precision_list)
    result_report["metrics"]["std_recall"] = np.std(recall_list)
    # Save detailed results
    result_report["details"] = results

    return result_report

def process_element_eval(args, model):
    """Process element evaluation tasks."""
    with open(args.uivision_test_file, 'r') as f:
        tasks_data = json.load(f)

    tasks_to_run = []
    for task_instance in tasks_data:
        task_instance = copy.deepcopy(task_instance)
        tasks_to_run.append(task_instance)
    
    print(f"Num of samples in test file: {len(tasks_to_run)}")
    print(f"Total tasks: {len(tasks_to_run)}")

    results = []
    for sample in tqdm(tasks_to_run):
        filename = sample["image_path"]
        img_path = os.path.join(args.uivision_imgs, filename)

        # Use ground_only_positive for all samples
        response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path)
        
        try:
            point = response["point"]
        except:
            point = None
        img_size = sample["image_size"]
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None

        try:
            raw_response = response["raw_response"] if "raw_response" in response else None
        except:
            raw_response = None
        
        sample_result = {
            "image_path": img_path, 
            "platform": sample["platform"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"], 
            "ui_type": sample["element_type"], 
            "pred": point_in_pixel, 
            "raw_response": raw_response,
            "ground_truth": sample["ground_truth"] if "ground_truth" in sample else None,
            "gt_bbox": sample["gt_bbox"] if "gt_bbox" in sample else None,
        }

        # Evaluate the response
        correctness = eval_sample_positive_gt(sample, response)
        sample_result.update({
            "bbox": sample["bbox"], 
            "correctness": correctness,
        })
        results.append(sample_result)
        
    result_report = evaluate_element(results)
    result_report["input_files"] = args.uivision_test_file
    result_report["task_type"] = "element"
    
    return result_report

def process_layout_eval(args, model):
    """Process layout evaluation tasks."""
    with open(args.uivision_test_file, 'r') as f:
        tasks_data = json.load(f)

    tasks_to_run = []
    for task_instance in tasks_data:
        task_instance = copy.deepcopy(task_instance)
        tasks_to_run.append(task_instance)

    print(f"Total layout tasks: {len(tasks_to_run)}")

    results = []
    for sample in tqdm(tasks_to_run):
        img_path = os.path.join(args.uivision_imgs, sample["image_path"])

        # Call layout generation
        response = model.layout_gen(name=sample["name"], explanation=sample["explanation"], image=img_path)

        img_size = sample["image_size"]
        try:
            box_in_pixel = [
                response["bbox"][0] * img_size[0], 
                response["bbox"][1] * img_size[1], 
                response["bbox"][2] * img_size[0], 
                response["bbox"][3] * img_size[1]
            ] if response["bbox"] else None
        except:
            print("No box found")    
            box_in_pixel = None

        try:
            raw_response = response["raw_response"]
        except:
            raw_response = None
        
        sample_result = {
            "image_path": img_path, 
            "platform": sample["platform"],
            "pred": box_in_pixel,
            "raw_response": raw_response,
            "bbox": sample["bbox"],
            'image_size': sample['image_size'],
            'name': sample['name'],
            'explanation': sample['explanation'],
        }

        # Evaluate layout
        eval_results = eval_layout(copy.deepcopy(sample["bbox"]), copy.deepcopy(box_in_pixel))
        
        sample_result.update({
            "eval_results": eval_results
        })
        results.append(sample_result)
        
    result_report = evaluate_layout(results)
    result_report["input_files"] = args.uivision_test_file
    result_report["task_type"] = "layout"
    
    return result_report

def main(args):
    model = build_model(args)
    print("Load model success")

    # Based on task type, run the appropriate evaluation
    if args.task == "element":
        print("Running element evaluation...")
        result_report = process_element_eval(args, model)
    elif args.task == "layout":
        print("Running layout evaluation...")
        result_report = process_layout_eval(args, model)
    
    # Prepare output report
    output_report = {
        "model_type": args.model_type,
        "task_type": args.task,
        "results": result_report
    }
    
    # Save to file
    with open(args.log_path, 'w') as f:
        json.dump(output_report, f, indent=4)
    
    logging.info(f"Evaluation of UIVision {args.task} completed.")

if __name__ == "__main__":
    main(parse_args())

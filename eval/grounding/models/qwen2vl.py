import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image
import tempfile
import pdb

from qwen_vl_utils import process_vision_info

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

def extract_bbox(s):
    # Regular expression to match the bounding box format within <|box_start|> and <|box_end|>
    pattern1 = r"<\|box_start\|\>\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]<\|box_end\|\>"
    pattern2 = r"<\|box_start\|\>\(\s*(\d+),\s*(\d+)\s*\),\(\s*(\d+),\s*(\d+)\s*\)<\|box_end\|\>"
    
    # Try to match the first pattern
    matches = re.findall(pattern1, s)
    if not matches:
        # If no match, try the second pattern
        matches = re.findall(pattern2, s)

    # pdb.set_trace()
    
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    
    return None  # Return None if no bbox is found

def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name

def extract_bbox_simple(s):
    # Regular expression to match bounding box in the format [x0, y0, x1, y1]
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    # Find all matches in the text
    matches = re.findall(pattern, s)
    
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    
    return None  # Return None if no bbox is found

class Qwen2VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2-VL-7B-Instruct"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."



        prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}" with grounding.'
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                
                result_dict["bbox"] = pred_bbox
                result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            click_point = [x / 1000 for x in click_point] if click_point else None
            result_dict["point"] = click_point  # can be none
        
        return result_dict
    



    def user_prompt_construct(self, name: str, explanation: str) -> str:
        """
        Generates a user prompt for GPT to identify the bounding box for a functional region.

        Args:
            name (str): The name of the functional region.
            explanation (str): A description of the functional region.

        Returns:
            str: A complete user prompt including requirements and input details.
        """
        return (
            f"A functional region is a specific part of the UI that groups tools or elements serving a particular purpose.\n"
            f"Your task is to identify the bounding box for the following functional region:\n\n"
            f"Name of the functional region: {name}\n"
            f"Explanation of it: {explanation}\n\n"
            f"Requirements:\n"
            f"- Return the bounding box as '<|box_start|>[[x0, y0, x1, y1]]<|box_end|>'. Do not use latex format, just plain text.\n"
            f"- The coordinates should be relative ranging from 0 to 1000 (i.e. all values (x and y) in a range [0, 1000]).\n"
            f"- Ensure the bounding box is tight around the region but fully captures all relevant elements.\n"
            # f"- please use your special token '<|box_start|>' and '<|box_end|>' to mark the bounding box.\n"
        )


    def layout_gen(self, name, explanation, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        user_prompt = self.user_prompt_construct(name, explanation)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # if '<|box_start|>' in response and '<|box_end|>' in response:
        pred_bbox = extract_bbox(response)
        if pred_bbox is None:
            pred_bbox = extract_bbox_simple(response)
        if pred_bbox is not None:
            (x1, y1), (x2, y2) = pred_bbox
            pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
            # click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
            
            result_dict["bbox"] = pred_bbox
            # result_dict["point"] = click_point
            

        else:
            print('---------------')
            print(response)
            print("no box found")
        # pdb.set_trace()
        return result_dict


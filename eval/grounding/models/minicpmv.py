import os
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, GenerationConfig


class MiniCPMVModel():
    def __init__(self):
        pass

    def load_model(self, model_name_or_path="openbmb/MiniCPM-V-2_6", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        # Setting default generation config
        self.override_generation_config = GenerationConfig.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            max_length=None,
        )
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.override_generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        # Prepare query
        grounding_prompt = f'What is the bounding box of the UI element corresponding to the user instruction "{instruction}"? Output in the format of x1 x2 y1 y2.'
        
        msgs = [{'role': 'user', 'content': [image, grounding_prompt]}]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        # Try getting groundings
        bbox = extract_first_bounding_box(response)
        click_point = extract_first_point(response)
        
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }
        
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
            f"- Return the bounding box as '[[x0, y0, x1, y1]]'. Please strictly follow that format.\n"
            f"- The coordinates should be relative ranging from 0 to 1000 (i.e. all values (x and y) in a range [0, 1000]).\n"
            f"- Ensure the bounding box is tight around the region but fully captures all relevant elements.\n"
        )
    
    def layout_gen(self, name, explanation, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        grounding_prompt = self.user_prompt_construct(name, explanation)
        
        msgs = [{'role': 'user', 'content': [image, grounding_prompt]}]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        # Try getting groundings
        bbox = extract_first_bounding_box(response)

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": None,
            "raw_response": response
        }
        
        return result_dict


def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    pattern = r"<box>(\d+) (\d+) (\d+) (\d+)</box>"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return [pos / 1000 for pos in bbox]
    
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    pattern = r"(\d+) (\d+)"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2))]
        return [pos / 1000 for pos in bbox]
    
    return None
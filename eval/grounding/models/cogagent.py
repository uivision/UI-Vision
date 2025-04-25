import os
import re

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class CogAgentModel():
    def __init__(self, model_path="THUDM/cogagent-chat-hf", tokenizer_path="lmsys/vicuna-7b-v1.5", quant=None, bf16=False):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def load_model(self, model_name_or_path="THUDM/cogagent-chat-hf", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        # Setting default generation config
        self.override_generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True).to_dict()
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
        elif image is None:
            raise ValueError("`image` should be provided.")
        
        # Prepare query
        grounding_prompt = f'What steps do I need to take to "{instruction}"?(with grounding)'
        # Build conversation input IDs
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=grounding_prompt,
            history=[], 
            images=[image] if image is not None else None
        )

        # Prepare model inputs
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[input_by_model['images'][0].to(self.model.device).to(torch.bfloat16)]] if image is not None else None,
        }

        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.model.device).to(torch.bfloat16)]]

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.override_generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(grounding_prompt):]

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


def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    pattern = r"\[\[(\d+),(\d+),(\d+),(\d+)\]\]"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return [pos / 1000 for pos in bbox]
    
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    pattern = r"\[\[(\d+),(\d+)\]\]"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2))]
        return [pos / 1000 for pos in bbox]
    
    return None

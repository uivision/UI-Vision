import os
import re

import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import openai
from openai import BadRequestError

from pathlib import Path
from typing import Literal, TypedDict

class Resolution(TypedDict):
    width: int
    height: int

# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class ClaudeModel():
    def __init__(self, model_name="anthropic/claude-3.5-sonnet", api_key=None, api_base_url=None):
        self.model_name = model_name
        self.api_key = api_key if api_key else os.environ.get("OPENROUTER_API_KEY")
        self.api_base_url = api_base_url if api_base_url else "https://openrouter.ai/api/v1"

        assert self.api_key, "OPENROUTER_API_KEY is not set"
        assert self.api_base_url, "base url is not set"

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url
        )

        self.scaling_enabled = False
        self.resize_enable = True

        self.override_generation_config = {
            "temperature": 0.0
        }

    def scale_coordinates(self, source, width, height, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self.scaling_enabled:
            return x, y
        ratio = width / height
        print(ratio)
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / width
        y_scaling_factor = target_dimension["height"] / height
        if source == "api":
            if x > self.width or y > self.height:
                # raise ToolError(f"Coordinates {x}, {y} are out of bounds")
                raise ValueError(f"Coordinates {x}, {y} are out of bounds")
            # scale up

            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def load_model(self):
        pass
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def image_resize(self,image_path):
        img = Image.open(image_path)
        img = img.convert("RGB")
        w, h = img.size
        # resize longest side to 1568 , keep aspect ratio
        # tgt_w = w
        # tgt_h = h
        if w >= h and w > 1568:
            tgt_w = 1568
            tgt_h = int(tgt_w * h / w)
            img = img.resize((tgt_w, tgt_h))
        elif h >= w and h > 1568:
            tgt_h = 1568
            tgt_w = int(tgt_h * w / h)
            img = img.resize((tgt_w, tgt_h))
        return img
    


    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            if self.resize_enable:
                image = self.image_resize(image_path = image_path)
            else:
                image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        base64_image = convert_pil_image_to_base64(image)
        response_text = None
        retry_times = 0
        retry_time_limit = 3
        while(response_text is None):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                    }
                                },
                                {
                                    "type": "text", 
                                    "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                                            "Don't output any analysis. Output your result in the format of [[x,y]]."
                                            "please be strict with the format [[x,y]]"
                                            "The instruction is:\n"
                                            f"{instruction}\n"

                                }
                            ],
                        }
                    ],
                    temperature=self.override_generation_config['temperature'],
                    max_tokens=2048,
                )
                response_text = response.choices[0].message.content
            except BadRequestError as e:
                if retry_times < retry_time_limit:
                    retry_times += 1
                    continue
                else:
                    response_text = "[[0,0]]"
                    break

        response_text = response_text.replace(" ","")
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)
    
        if click_point and self.scaling_enabled:
            x, y = self.scale_coordinates(source = "api", width = image.size[0],height = image.size[1], x = click_point[0], y = click_point[1])
            click_point = [x,y]
        if click_point and self.resize_enable:
            click_point[0] = click_point[0]/image.size[0]
            click_point[1] = click_point[1]/image.size[1]
            # print(click_point)
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text
        }
        
        return result_dict

def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    
    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Capture the bounding box coordinates as floats
        bbox = [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
        return bbox
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first point in the format [[x0,y0]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    
    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        point = [float(match.group(1)), float(match.group(2))]
        print(point)
        return point
    
    return None
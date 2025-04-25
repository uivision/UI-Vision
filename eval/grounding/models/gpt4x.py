import os
import re
import requests
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pdb

import openai
from openai import BadRequestError

model_name = "gpt-4o"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class ProprietaryModel:
    def __init__(self, model_name="gpt-4o"):
        self.api_key = OPENAI_KEY
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPEN_ROUTER_API_KEY environment variable")
        
        self.model = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        self.override_generation_config = {
            "temperature": 0.0,
            "max_tokens": 2048
        }
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def encode_image(self, image_path):
        """Convert image to base64 string."""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save to bytes
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'data:image/jpeg;base64,{img_str}'
    
    def ground_only_positive(self, instruction, image, system_prompt = None):
        if isinstance(image, str):
            image_input = self.encode_image(image)
        else:
            raise ValueError("Invalid input image.")

        if not system_prompt:
            system_prompt = "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."
        
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_input
                }
            },
            {
                "type": "text",
                "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                        "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1."
                        "The instruction is:\n"
                        f"{instruction}\n"
            }
        ]
        message = [{
            'role': 'user',
            'content': content
        }]
        try:
            payload = {
                'model': self.model,
                'messages': message,
                'system': system_prompt,
                'temperature': self.override_generation_config['temperature'],
                'max_tokens': 2048
            }
            # stuck this like
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            # pdb.set_trace()
            response.raise_for_status()
            response_text =  response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")   
            return None

        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)
        
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text
        }
        
        return result_dict
    
    SYSTEM_PROMPT = \
"""You are an assistant for analyzing UI layouts. Your task is to identify the exact bounding box of a functional region based on its name and explanation. \n 
Return the bounding box as [[x0, y0, x1, y1]] with values normalized to [0, 1], ensuring the box is tight and accurate."""


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
            f"- Return the bounding box as [[x0, y0, x1, y1]]. Do not use latex format, just plain text.\n"
            f"- The coordinates should be relative ranging from 0 to 1 (i.e. all values (x and y) in a range [0, 1]).\n"
            f"- Ensure the bounding box is tight around the region but fully captures all relevant elements.\n"
        )

    
    def layout_gen(self, name, explanation, image):
        if isinstance(image, str):
            image_input = self.encode_image(image)
        else:
            raise ValueError("Invalid input image.")
        
        user_prompt = self.user_prompt_construct(name, explanation)

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_input
                }
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        message = [{
            'role': 'user',
            'content': content
        }]
        try:
            payload = {
                'model': self.model,
                'messages': message,
                'temperature': self.override_generation_config['temperature'],
                'max_tokens': 2048
            }
            # stuck this like
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            # pdb.set_trace()
            response.raise_for_status()
            response_text =  response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")   
            return None

        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "raw_response": response_text
        }
        
        return result_dict

class GPT4XModel():
    def __init__(self, model_name="gpt-4o"):
        self.client = openai.OpenAI(
            api_key=OPENAI_KEY,
        )
        self.model_name = model_name
        self.override_generation_config = {
            "temperature": 0.0
        }

    def load_model(self):
        pass
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        base64_image = convert_pil_image_to_base64(image)

        try:
            message = [
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
                                        "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1."
                                        "The instruction is:\n"
                                        f"{instruction}\n"

                            }
                        ],
                    }
                ],
            if self.model_name == "google/gemini-pro-1.5":
                message = [
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
                                        "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. If it does not fall within this range, please normalize the coordinates.\n"
                                        "The instruction is:\n"
                                        f"{instruction}\n"

                            }
                        ],
                    }
                ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages= message,
                temperature=self.override_generation_config['temperature'],
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return None

        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)
        
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
    # Regular expression pattern to match the first bounding box in the format [[x0, y0, x1, y1]]
    pattern = r"\[\[\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*\]\]"

    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)

    if match:
        # Capture the bounding box coordinates as floats
        bbox = [float(match.group(i)) for i in range(1, 5)]
        return bbox
    
    return None  # Return None if no match is found

def extract_first_bounding_box_v1(text):
    # Regular expression to capture bounding box inside LaTeX-style \left[ \left[ ... \right] \right]
    pattern = r"\\left\[\s*\\left\[\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*\\right\]\s*\\right\]"

    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)

    if match:
        # Capture the bounding box coordinates as floats
        bbox = [float(match.group(i)) for i in range(1, 5)]
        return bbox
    
    return None  # Return None if no match is found

def extract_first_point(text):
    # Regular expression pattern to match the first point in the format [[x0,y0]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    
    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        point = [float(match.group(1)), float(match.group(2))]
        return point
    
    return None

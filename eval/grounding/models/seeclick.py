import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image

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
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    return click_point

# bbox (qwen str) -> bbox
def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    # Convert the tuples of strings into tuples of integers
    return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class SeeClickModel():
    def load_model(self, model_name_or_path="cckevinn/SeeClick"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            max_length=None,
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


        prompt_origin = "In this UI screenshot, what is the position of the element corresponding to the command \"{instruction}\" (with point)?"
        prompt = prompt_origin.format(instruction=instruction)
        query = self.tokenizer.from_list_format([
            {'image': image_path},  # Either a local path or an url
            {'text': prompt}, 
        ])
        # print(query)
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response
        }

        if '<box>' in response:
            pred_bbox = extract_bbox(response)
            (x1, y1), (x2, y2) = pred_bbox
            
            click_point = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            result_dict["bbox"] = pred_bbox
            result_dict["point"] = click_point
        else:
            click_point = pred_2_point(response)
            result_dict["point"] = click_point
        
        return result_dict
    
    def layout_gen(self, name, explanation, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        instruction = f"An functional region named \"{name}\" where {explanation}."
        prompt = f"In this UI screenshot, what is the position of the functional region corresponding to the instruction \"{instruction}\" (with bbox)?"
        query = self.tokenizer.from_list_format([
            {'image': image_path},  # Either a local path or an url
            {'text': prompt}, 
        ])
        # print(query)
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<box>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox:
                (x1, y1), (x2, y2) = pred_bbox
                result_dict["bbox"] = [x1, y1, x2, y2]
        else:

            pred_bbox = extract_bbox_simple(response)
            if pred_bbox:
                (x1, y1), (x2, y2) = pred_bbox
                result_dict["bbox"] = [x1, y1, x2, y2]
            else:
                print("No bbox found.")
        
        return result_dict

    
def extract_bbox_simple(s):
    # Regular expression to match the bounding box in the format (x0, y0, x1, y1) with floating-point numbers
    pattern = r'\(\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\)'
    
    # Find all matches in the text
    matches = re.findall(pattern, s)
    
    if matches:
        # Get the last match and return as a tuple of floats
        last_match = matches[-1]
        return (float(last_match[0]), float(last_match[1])), (float(last_match[2]), float(last_match[3]))
    
    return None  # Return None if no bbox is found
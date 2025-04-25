import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers.generation import GenerationConfig
import re
import os
from PIL import Image
import io
import base64
from openai import OpenAI

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, pre_resize_by_width
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

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

# bbox (qwen str) -> bbox
def extract_bbox(s):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    return None

def encode_image(image_path):
    """Encode image as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

    
def format_openai_template(description: str, base64_image):
    """Format OpenAI request template."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {description}

Answer:""",
                },
            ],
        },
    ]

class UGroundModel():
    def load_model(self, model_name_or_path="osunlp/UGround"):
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_name_or_path, None, model_name_or_path)
        self.model.eval()

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("osunlp/UGround", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        prompt_template = "In the screenshot, where are the pixel coordinates (x, y) of the element corresponding to \"{}\"?"
        full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt_template.format(instruction)
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = image.convert('RGB')
        # Resize image and prepare tensor for inference
        resized_image, pre_resize_scale = pre_resize_by_width(image)  # resize to 1344 * 672
        image_tensor, image_new_size = process_images([resized_image], self.image_processor, self.model.config)

        # Perform inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                image_sizes=[image_new_size],
                generation_config=GenerationConfig(**self.generation_config),
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=16384,
                use_cache=True
            )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        pred_bbox = extract_bbox(response)
        width, height = image.size
        if pred_bbox is not None:
            (x1, y1), (x2, y2) = pred_bbox
            # pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
            pred_bbox = tuple(x / pre_resize_scale for x in pred_bbox)
            pred_bbox = [pred_bbox[0] / width, pred_bbox[1] / height, pred_bbox[2] / width, pred_bbox[3] / height]
            click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
            
            result_dict["bbox"] = pred_bbox
            result_dict["point"] = click_point
        else:
            click_point = pred_2_point(response)
            # click_point = [click_point[0] / 1000, click_point[1] / 1000]
            click_point = tuple(x / pre_resize_scale for x in click_point)
            click_point = [click_point[0] / width, click_point[1] / height]

            result_dict["point"] = click_point  # can be none
        
        return result_dict

class UGroundV1Model():
    def __init__(self, model_name_or_path="osunlp/UGround-V1-7B"):
        self.model_name_or_path = model_name_or_path
        self.temperature = 0
        self.max_new_tokens = 256
        self.client = OpenAI(
                            base_url="http://localhost:8000/v1",  # vLLM service address
                            api_key="<token>",  # Must match the --api-key used in vLLM serve
                        )

    def set_generation_config(self, temperature=0, max_new_tokens=256):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def ground_only_positive(self, instruction, image):
        try:
            if isinstance(image, str):
                image_path = image
                assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
                image = Image.open(image_path).convert('RGB')
            assert isinstance(image, Image.Image), "Invalid input image."

            # Get base64 image
            if isinstance(image, str):
                base64_image = encode_image(image)
            else:
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                base64_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

            # Format request
            messages = format_openai_template(instruction, base64_image)

            # Call model API
            completion = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )

            # Parse model output
            response_text = completion.choices[0].message.content.strip()
            ratio_coords = eval(response_text)
            x_ratio, y_ratio = ratio_coords

            # Convert to normalized coordinates
            x_coord = x_ratio / 1000 
            y_coord = y_ratio / 1000

            result_dict = {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": response_text,
                "bbox": None,
                "point": [x_coord, y_coord]
            }

            return result_dict

        except Exception as e:
            print(f"Error processing instruction: {e}")
            return {
                "result": "error",
                "format": "x1y1x2y2",
                "raw_response": str(e),
                "bbox": None,
                "point": None
            }

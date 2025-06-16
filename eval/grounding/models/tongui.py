import os
import re
import tempfile
from PIL import Image
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig

from qwen_vl_utils import process_vision_info

def extract_point(response):
    numbers = [float(num) for num in re.findall(r"\d*\.\d+|\d+", response)]
    
    # Return the last two numbers if there are more than two
    if len(numbers) > 2:
        return numbers[:2]
    elif len(numbers) == 2:
        return numbers
    else:
        return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class TongUIModel():
    def load_model(self, model_name_or_path="Bofeee5675/TongUI-3B", device="cuda"):
        print("Loading model...", model_name_or_path)
        self.device = device
        self.min_pixels = 256*28*28
        self.max_pixels = 1344*28*28
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        # Setting default generation config
        self.generation_config = dict()
        self.set_generation_config(
            max_length=8192,
            do_sample=False,
            temperature=0.01
        )

    def set_generation_config(self, **kwargs):
        if kwargs["temperature"] < 0.01:
            kwargs["temperature"] = 0.01
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."},
                    {"type": "image", "image": image_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    {"type": "text", "text": instruction}
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        print("Response:",response)
        print("Response length:",len(response))
        click_xy = extract_point(response)
        print("Click XY:",click_xy)
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": click_xy
        }

        return result_dict


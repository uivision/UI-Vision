import json
import os
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch
import ast

def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class AriaUIModel():
    def load_model(self, model_name_or_path="Aria-UI/Aria-UI-base", device="cuda"):
        self.device = device
        self.model = model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Aria-UI/Aria-UI-base", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        # self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        prompt_origin = 'Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: '
        full_prompt = prompt_origin + instruction
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": None, "type": "image"},
                    {"text": full_prompt, "type": "text"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # clear unused gpu memory
        torch.cuda.empty_cache()

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                stop_strings=["<|im_end|>"],
                tokenizer=self.processor.tokenizer,
                do_sample=False,
                # temperature=0.9,
            )

        output_ids = output[0][inputs["input_ids"].shape[1] :]
        response = self.processor.decode(output_ids, skip_special_tokens=True)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        try:
            point = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
            x, y = point
            click_point = [x / 1000, y / 1000]
            if 0 <= click_point[0] <= 1 and 0 <= click_point[1] <= 1:
                result_dict["point"] = click_point
        except Exception as e:
            point = None

        del inputs, output, output_ids
        torch.cuda.empty_cache()

        return result_dict

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path,pre_resize_by_width

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["id"] if "id" in line else "unknown_id"
        image_file = line["image"]
        if "description" in line:
            qs=line["description"]
        elif "text" in line:
            qs = line["text"]


        qs = f"In the screenshot, where are the pixel coordinates (x, y) of the element corresponding to \"{qs}\"?"
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        image,pre_resize_scale=pre_resize_by_width(image)


        original_imae_size=image.size


        image_tensor, image_new_size = process_images([image], image_processor, model.config)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                image_sizes=[image_new_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=16384,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        metadata = {k: v for k, v in line.items() if k not in ["id", "image", "text",'description']}

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   'image': image_file,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": metadata}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="osunlp/UGround")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/fs/ess/PAS1576/boyu_gou/Benchmark/screenspot_imgs_resized/")
    parser.add_argument("--question-file", type=str, default="/fs/ess/PAS1576/boyu_gou/Benchmark/screenspot_web_text.jsonl")
    parser.add_argument("--answers-file", type=str, default="/fs/ess/PAS1576/boyu_gou/Benchmark/answer_screenspot_web.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)

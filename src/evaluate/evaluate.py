import argparse
from threading import Thread
from PIL import Image
from src.utils import (
    load_pretrained_model,
    get_model_name_from_path,
    disable_torch_init,
)
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info
import json


warnings.filterwarnings("ignore")


def run_inference(image_name):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"images/{image_name}",
                },
                {
                    "type": "text",
                    "text": "Locate key objects and display depth and bounding box info in JSON format.",
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def main(args):
    global processor, model, device

    device = args.device

    disable_torch_init()

    use_flash_attn = True

    model_name = get_model_name_from_path(args.model_path)

    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=args.device,
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=args.device,
        use_flash_attn=use_flash_attn,
    )
    print(f"Model successfully loaded: {args.model_base}")

    json_file_path = "validation_data.json"
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return []

    image_names = []
    if isinstance(data, list):
        for item in data:
            if "image" in item:
                image_names.append(item["image"])
    elif isinstance(data, dict):
        if "image" in data:
            image_names.append(data["image"])

    for image_name in image_names:
        result = run_inference(image_name)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

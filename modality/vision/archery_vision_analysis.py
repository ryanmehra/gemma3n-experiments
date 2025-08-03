
"""
Ownership: Ryan Mehra @ ArcheryPulse
Created: 2025-07-14

License: Proprietary and Confidential

This file contains confidential and proprietary information
of ArcheryPulse. Unauthorized copying, modification,
distribution, or disclosure of this file or its contents
is strictly prohibited without prior written consent.
"""

from transformers import pipeline
import torch
import os

# Use the correct model ID for vision support
MODEL_ID = "google/gemma-3n-e4b-it"

# Optionally authenticate with Hugging Face if token is set
if os.getenv("HUGGINGFACE_TOKEN"):
    from huggingface_hub import login
    login(os.getenv("HUGGINGFACE_TOKEN"))

def describe_image_with_pipeline(image_path, user_text=None, pipe=None):
    # Accepts a preloaded pipeline for efficiency
    if pipe is None:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_ID,
            torch_dtype=dtype,
            device=device,
        )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """You are a Archery Expert Coach who is deeper in understanding body for, as-in archer is holding the bow right, line and length of arm, positioning at face, back muscle engagement etc. You will be given images of archer from various perspectives and you will have to analyze on what's looking good and what is not. Explain in this format:

What I see:
Gender: <Male, Female, Cannot Determine>
Wearing Top: <Yes, No, Not Visible> And If Yes, <COLOR> and <TYPE like Full Arm Shirt, Sleeveless Short, Tang Top etc>
Wearing Bottom: <Yes, No, Not Visible> And If Yes, <COLOR> and <TYPE like Shorts, Jeans, Athletic Wear etc>
Face Visible: <Yes or No>
Back Visible: <Yes or No>
Back Muscle Group Visible: <Yes or No>
Arms Full Length Visible: <Yes or No>
Right Foot Forward: <Yes, No, Not Visible>
Left Foot Forward: <Yes, No, Not Visible>
Stance: <stance>
Archery Related: <Yes or No>
Image Useful For Archery Analysis: <Yes or No>

Expert Analysis: <Your Detailed Analysis>"""}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                # {"type": "text", "text": user_text or "Describe this image."}
            ]
        }
    ]
    output = pipe(text=messages, max_new_tokens=200)
    return output[0]["generated_text"][-1]["content"]


import glob
import time
import psutil
import sys
import os

def print_markdown_bold(text):
    import re
    # Replace **bold** with ANSI bold
    return re.sub(r'\*\*(.*?)\*\*', '\033[1m\\1\033[0m', text)

def get_model_size_gb(model_id):
    # Try to estimate model size by summing all files in the cache dir for this model
    from transformers.utils import cached_file, WEIGHTS_NAME, CONFIG_NAME
    import pathlib
    try:
        # Get the path to the main weights file
        weights_path = cached_file(model_id, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        if weights_path is None:
            return None
        model_dir = pathlib.Path(weights_path).parent
        total_bytes = sum(f.stat().st_size for f in model_dir.glob("**/*") if f.is_file())
        return total_bytes / (1024 ** 3)
    except Exception:
        return None

def print_memory_usage_gb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 3)

def main():
    # Find all image files in the current directory
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
    if not image_files:
        print("No image files found in the current directory.")
        return

    # Time model loading
    model_load_start = time.time()
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        torch_dtype=dtype,
        device=device,
    )
    model_load_end = time.time()
    model_load_time = model_load_end - model_load_start

    # Print model size
    model_size_gb = get_model_size_gb(MODEL_ID)
    if model_size_gb is not None:
        print(f"Model size on disk: {model_size_gb:.2f} GB")
    else:
        print("Model size on disk: Unknown")

    print(f"Model loaded in {model_load_time:.2f} seconds.")
    print(f"Initial memory usage: {print_memory_usage_gb():.2f} GB\n")

    for image_path in image_files:
        print(f"Processing image: {image_path}")
        infer_start = time.time()
        mem_before = print_memory_usage_gb()
        try:
            result = describe_image_with_pipeline(image_path, pipe=pipe)
            infer_end = time.time()
            infer_time = infer_end - infer_start
            mem_after = print_memory_usage_gb()
            print(f"---\nImage Name: {image_path}")
            print("\nLLM Inference:\n" + print_markdown_bold(result))
            print(f"\nInference Time: {infer_time:.2f} seconds")
            print(f"Memory usage before inference: {mem_before:.2f} GB")
            print(f"Memory usage after inference: {mem_after:.2f} GB\n---\n")
            print(f"\n\n---------------------------------------------------------------\n\n")
        except Exception as e:
            infer_end = time.time()
            infer_time = infer_end - infer_start
            mem_after = print_memory_usage_gb()
            print(f"Error processing {image_path}: {e} (in {infer_time:.2f} seconds)")
            print(f"Memory usage before inference: {mem_before:.2f} GB")
            print(f"Memory usage after inference: {mem_after:.2f} GB\n---\n")

if __name__ == "__main__":
    main()

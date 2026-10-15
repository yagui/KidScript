"""OCR a PDF using unsloth/Qwen3.5-0.8B + Ak137 handwritten-spanish-ocr LoRA adapter."""

import argparse
import sys

import torch
import numpy as np
from pdf2image import convert_from_path
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = "unsloth/Qwen3.5-0.8B"
ADAPTER = "qwen3.5-handwritten-spanish-adapter-v2"
# Must match the prompt used in train_qwen_lora.py.
PROMPT = "Transcribe the text in this image exactly. Do not add, correct, or infer text."


def detect_text_regions(
    image, background_threshold=200, padding=24, gap=45, min_width_ratio=0.25
):
    """Find horizontal handwriting bands using the page's dark-pixel projection."""
    gray = np.asarray(image.convert("L"))
    ink = gray < background_threshold
    row_counts = ink.sum(axis=1)
    active = row_counts > max(20, int(image.width * 0.06))
    active = np.convolve(active.astype(np.uint8), np.ones(gap, dtype=np.uint8), mode="same") > 0

    runs = []
    start = None
    for row, is_active in enumerate(active):
        if is_active and start is None:
            start = row
        elif not is_active and start is not None:
            runs.append((start, row))
            start = None
    if start is not None:
        runs.append((start, image.height))

    regions = []
    for top, bottom in runs:
        rows = np.flatnonzero(ink[top:bottom].any(axis=0))
        if rows.size == 0:
            continue
        left, right = rows[0], rows[-1] + 1
        if right - left < image.width * min_width_ratio:
            continue
        regions.append(
            image.crop(
                (
                    max(0, left - padding),
                    max(0, top - padding),
                    min(image.width, right + padding),
                    min(image.height, bottom + padding),
                )
            )
        )
    return regions or [image]


def load_model(adapter=ADAPTER, max_image_pixels=100352):
    device = "cpu"
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > 3 * 1024**3:  # need ~3GB free for the 0.8B model
            device = "cuda"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.image_processor.size.longest_edge = max_image_pixels  # same cap as training
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=dtype, device_map=device
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor, device


def ocr_image(model, processor, device, image, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default="1a.pdf", help="Path to the input PDF")
    parser.add_argument("-o", "--output", default="output.txt", help="Path to write the transcription")
    parser.add_argument("--dpi", type=int, default=300, help="Rendering DPI for PDF pages")
    parser.add_argument("--save-crops", help="Directory where detected text crops are saved")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--adapter", default=ADAPTER, help="LoRA adapter directory or Hugging Face id")
    parser.add_argument("--max-image-pixels", type=int, default=100352)
    args = parser.parse_args()

    pages = convert_from_path(args.pdf, dpi=args.dpi)
    model, processor, device = load_model(args.adapter, args.max_image_pixels)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, page in enumerate(pages, start=1):
            print(f"Processing page {i}/{len(pages)}...", file=sys.stderr)
            regions = detect_text_regions(page)
            if args.save_crops:
                from pathlib import Path

                crop_dir = Path(args.save_crops)
                crop_dir.mkdir(parents=True, exist_ok=True)
                for region_number, region in enumerate(regions, start=1):
                    region.save(crop_dir / f"page-{i:03d}-region-{region_number:02d}.png")
            f.write(f"========== PAGE {i} ==========\n\n")
            for region in regions:
                text = ocr_image(model, processor, device, region, args.max_new_tokens)
                f.write(text + "\n\n")


if __name__ == "__main__":
    main()

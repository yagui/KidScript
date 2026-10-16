"""Fine-tune a Qwen3.5 vision-language model with image/transcription pairs.

Each JSON record must contain an image path and its exact target transcription.
Relative image paths are resolved from the manifest file's directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-0.8B"
DEFAULT_ADAPTER = "Ak137/qwen3.5-0.8B-handwritten-spanish-ocr-lora"
PROMPT = "Transcribe the text in this image exactly. Do not add, correct, or infer text."


class OCRPairs(Dataset):
    def __init__(self, manifest: Path, processor: Any):
        self.manifest = manifest.resolve()
        self.processor = processor
        self.records: list[dict[str, str]] = []

        manifest_data = json.loads(self.manifest.read_text(encoding="utf-8"))
        if isinstance(manifest_data, dict):
            manifest_data = manifest_data.get("examples")
        if not isinstance(manifest_data, list):
            raise TypeError(f"{self.manifest}: expected a JSON array of examples")

        for line_number, record in enumerate(manifest_data, start=1):
            if not isinstance(record, dict):
                raise TypeError(f"{self.manifest}: example {line_number} must be an object")
            if not isinstance(record.get("image"), str) or not isinstance(record.get("text"), str):
                raise TypeError(
                    f"{self.manifest}:{line_number}: expected string fields 'image' and 'text'"
                )
            if not record["text"].strip():
                print(
                    f"Skipping example {line_number} with an empty transcription",
                    file=sys.stderr,
                )
                continue
            image_path = Path(record["image"])
            if not image_path.is_absolute():
                image_path = self.manifest.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{self.manifest}:{line_number}: image not found: {image_path}"
                )
            self.records.append({"image": str(image_path), "text": record["text"]})

        if not self.records:
            raise ValueError(f"No training examples found in {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")

        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
        full_messages = [
            user_message,
            {"role": "assistant", "content": record["text"]},
        ]
        encoded = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        prompt_length = prompt["input_ids"].shape[1]
        if not torch.equal(prompt["input_ids"][0], encoded["input_ids"][0, :prompt_length]):
            raise ValueError(
                "Prompt tokens are not a prefix of the full conversation; "
                "cannot safely mask the transcription labels."
            )

        encoded["labels"] = encoded["input_ids"].clone()
        encoded["labels"][:, :prompt_length] = -100
        return encoded


def single_example_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(features) != 1:
        raise ValueError("This collator expects per-device batch size 1 for variable image sizes")
    return features[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("crops/transcriptions.json"),
        help="Training JSON manifest (default: crops/transcriptions.json)",
    )
    parser.add_argument("--validation-data", type=Path, help="Optional validation JSON manifest")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Starting OCR LoRA adapter")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("qwen3.5-handwritten-spanish-adapter")
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Set to 1 for a training smoke test"
    )
    parser.add_argument("--max-image-pixels", type=int, default=100352)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be positive")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 (unlimited) or a positive integer")
    if args.max_image_pixels < 1:
        parser.error("--max-image-pixels must be positive")

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.image_processor.size.longest_edge = args.max_image_pixels
    train_dataset = OCRPairs(args.train_data, processor)
    validation_dataset = OCRPairs(args.validation_data, processor) if args.validation_data else None

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=model_dtype,
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
        eval_strategy="epoch" if validation_dataset is not None else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=single_example_collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()

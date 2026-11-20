"""Fine-tune Qwen3.5 with line, paragraph, and PDF transcription examples."""

import argparse
import json
import math
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from pdf2image import convert_from_path, pdfinfo_from_path
from peft import PeftModel
from PIL import Image
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-0.8B"
DEFAULT_ADAPTER = "Ak137/qwen3.5-0.8B-handwritten-spanish-ocr-lora"
PROMPT = "Transcribe the text in this image exactly. Do not add, correct, or infer text."
PARAGRAPH_PATTERN = re.compile(r"para(?:graph|ghrap)_(\d+)$", re.IGNORECASE)


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def paragraph_groups(
    line_images: list[Path], paragraph_images: list[Path]
) -> list[tuple[int, int]]:
    """Split numbered crops into contiguous groups matching paragraph aspect ratios."""
    if not line_images:
        return []
    if not paragraph_images:
        return []
    if len(paragraph_images) == 1:
        return [(0, len(line_images))]
    if len(paragraph_images) > len(line_images):
        raise ValueError(
            f"Cannot assign {len(line_images)} line crops to {len(paragraph_images)} paragraphs"
        )

    dimensions = []
    for image_path in line_images:
        with Image.open(image_path) as image:
            dimensions.append(image.size)
    paragraph_dimensions = []
    for image_path in paragraph_images:
        with Image.open(image_path) as image:
            paragraph_dimensions.append(image.size)

    best_score = math.inf
    best_groups: list[tuple[int, int]] = []
    for cuts in combinations(range(1, len(line_images)), len(paragraph_images) - 1):
        boundaries = (0, *cuts, len(line_images))
        groups = list(zip(boundaries[:-1], boundaries[1:], strict=True))
        score = 0.0
        for (start, end), (paragraph_width, paragraph_height) in zip(
            groups, paragraph_dimensions, strict=True
        ):
            group = dimensions[start:end]
            combined_width = max(width for width, _ in group)
            combined_height = sum(height for _, height in group)
            expected_ratio = combined_width / combined_height
            actual_ratio = paragraph_width / paragraph_height
            score += abs(math.log(actual_ratio / expected_ratio))
        if score < best_score:
            best_score = score
            best_groups = groups
    return best_groups


def split_dataset(
    dataset: "OCRPairs", validation_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    """Split whole PDF/crop folders so related images cannot leak across sets."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(dataset.records):
        if "pdf" in record:
            pdf_path = Path(record["pdf"]).resolve()
            group_path = pdf_path.parent / pdf_path.stem
        else:
            image_dir = Path(record["image"]).resolve().parent
            if image_dir.name.startswith("crops_"):
                group_path = image_dir.parent / image_dir.name.removeprefix("crops_")
            else:
                group_path = image_dir
        group = str(group_path)
        groups.setdefault(group, []).append(index)

    group_names = list(groups)
    if len(group_names) < 2:
        raise ValueError("A 70/30 split requires at least two document groups")
    random.Random(seed).shuffle(group_names)

    validation_group_count = round(len(group_names) * validation_fraction)
    validation_group_count = min(max(validation_group_count, 1), len(group_names) - 1)
    validation_groups = set(group_names[:validation_group_count])
    train_indices = [
        index
        for group, indices in groups.items()
        if group not in validation_groups
        for index in indices
    ]
    validation_indices = [
        index
        for group, indices in groups.items()
        if group in validation_groups
        for index in indices
    ]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


class OCRPairs(Dataset):
    def __init__(self, data_path: Path, processor: Any, pdf_dpi: int = 200):
        self.data_path = data_path.resolve()
        self.processor = processor
        self.pdf_dpi = pdf_dpi
        self.records: list[dict[str, Any]] = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.data_path}")
        if self.data_path.suffix.lower() == ".pdf":
            self._add_document(self.data_path)
        elif self.data_path.is_file():
            self.records.extend(self._read_manifest(self.data_path))
        else:
            manifests = sorted(self.data_path.rglob("transcriptions.json"), key=natural_key)
            pdfs = sorted(self.data_path.rglob("*.pdf"), key=natural_key)
            if pdfs:
                for pdf_path in pdfs:
                    self._add_document(pdf_path)
            else:
                for manifest in manifests:
                    self.records.extend(self._read_manifest(manifest))
                for image_path, text in self._paragraph_examples(manifests):
                    self.records.append({"image": str(image_path), "text": text})

        if not self.records:
            raise ValueError(f"No training examples found in {self.data_path}")

    def _read_manifest(self, manifest: Path) -> list[dict[str, str]]:
        manifest = manifest.resolve()
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(manifest_data, dict):
            manifest_data = manifest_data.get("examples")
        if not isinstance(manifest_data, list):
            raise TypeError(f"{manifest}: expected a JSON array of examples")

        records: list[dict[str, str]] = []
        for line_number, record in enumerate(manifest_data, start=1):
            if not isinstance(record, dict):
                raise TypeError(f"{manifest}: example {line_number} must be an object")
            if not isinstance(record.get("image"), str) or not isinstance(record.get("text"), str):
                raise TypeError(
                    f"{manifest}:{line_number}: expected string fields 'image' and 'text'"
                )
            if not record["text"].strip():
                print(
                    f"Skipping {manifest} example {line_number} with an empty transcription",
                    file=sys.stderr,
                )
                continue
            image_path = Path(record["image"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"{manifest}:{line_number}: image not found: {image_path}")
            records.append({"image": str(image_path), "text": record["text"]})
        return records

    def _paragraph_images(self, crop_dir: Path) -> list[Path]:
        numbered_images: list[tuple[int, Path]] = []
        for path in crop_dir.glob("*.png"):
            match = PARAGRAPH_PATTERN.fullmatch(path.stem)
            if match:
                numbered_images.append((int(match.group(1)), path))
        return [path for _, path in sorted(numbered_images)]

    def _paragraph_examples(self, manifests: list[Path]) -> list[tuple[Path, str]]:
        examples: list[tuple[Path, str]] = []
        for manifest in manifests:
            records = self._read_manifest(manifest)
            numbered = [record for record in records if Path(record["image"]).stem.isdigit()]
            numbered.sort(key=lambda record: int(Path(record["image"]).stem))
            images = self._paragraph_images(manifest.parent)
            line_images = [Path(record["image"]) for record in numbered]
            for image_path, (start, end) in zip(
                images, paragraph_groups(line_images, images), strict=True
            ):
                text = "\n".join(record["text"] for record in numbered[start:end])
                if text.strip():
                    examples.append((image_path, text))
        return examples

    def _add_document(self, pdf_path: Path) -> None:
        crop_dir = pdf_path.with_name(f"crops_{pdf_path.stem}")
        manifest = crop_dir / "transcriptions.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing transcription manifest for {pdf_path}: {manifest}")

        records = self._read_manifest(manifest)
        self.records.extend(records)
        numbered = [record for record in records if Path(record["image"]).stem.isdigit()]
        numbered.sort(key=lambda record: int(Path(record["image"]).stem))
        if not numbered:
            raise ValueError(f"{manifest}: no numbered crop transcriptions found")

        paragraph_images = self._paragraph_images(crop_dir)
        groups = paragraph_groups([Path(record["image"]) for record in numbered], paragraph_images)
        paragraph_texts: list[str] = []
        explicit_images = {Path(record["image"]).resolve() for record in records}
        for paragraph_image, (start, end) in zip(paragraph_images, groups, strict=True):
            text = "\n".join(record["text"] for record in numbered[start:end])
            paragraph_texts.append(text)
            if paragraph_image.resolve() not in explicit_images:
                self.records.append({"image": str(paragraph_image), "text": text})

        if not paragraph_texts:
            paragraph_texts = ["\n".join(record["text"] for record in numbered)]
        full_text = "\n\n".join(paragraph_texts)
        info = pdfinfo_from_path(str(pdf_path))
        for page_number in range(1, int(info["Pages"]) + 1):
            self.records.append({"pdf": str(pdf_path), "page": page_number, "text": full_text})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if "pdf" in record:
            pages = convert_from_path(
                record["pdf"],
                dpi=self.pdf_dpi,
                first_page=record["page"],
                last_page=record["page"],
            )
            if len(pages) != 1:
                raise RuntimeError(f"Failed to render {record['pdf']} page {record['page']}")
            image = pages[0].convert("RGB")
        else:
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
        default=Path("training_set/"),
        help="Training directory, PDF, or JSON manifest (default: training_set/)",
    )
    parser.add_argument(
        "--validation-data", type=Path, help="Optional validation directory, PDF, or JSON"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.3,
        help="Fraction reserved for validation when --validation-data is omitted",
    )
    parser.add_argument("--pdf-dpi", type=int, default=200, help="DPI for PDF training pages")
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
    if args.pdf_dpi < 1:
        parser.error("--pdf-dpi must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be greater than 0 and less than 1")

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.image_processor.size.longest_edge = args.max_image_pixels
    all_examples = OCRPairs(args.train_data, processor, args.pdf_dpi)
    if args.validation_data:
        train_dataset: Dataset = all_examples
        validation_dataset: Dataset = OCRPairs(args.validation_data, processor, args.pdf_dpi)
    else:
        train_dataset, validation_dataset = split_dataset(
            all_examples, args.validation_fraction, args.seed
        )
        print(
            f"Validation fraction: {len(validation_dataset) / len(all_examples):.1%} "
            f"({len(validation_dataset)}/{len(all_examples)} examples)",
            file=sys.stderr,
        )
    print(
        f"Loaded {len(train_dataset)} training examples"
        f" and {len(validation_dataset)} validation examples",
        file=sys.stderr,
    )

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

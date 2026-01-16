import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from big5_infer import Big5InferenceModel  # type: ignore[import]


logger = logging.getLogger("score_ocean_parquet")


def iter_json_array(path: str) -> Iterable[Dict[str, Any]]:
    """Stream JSON objects from a large on-disk array."""
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            yield obj


def normalize_dialogue(dialogue: Any) -> List[str]:
    """Ensure dialogue content is provided as a list of strings."""
    if isinstance(dialogue, list):
        return [str(x) for x in dialogue]
    if dialogue is None:
        return []
    if isinstance(dialogue, str):
        return [dialogue]
    return [str(dialogue)]


def concat_dialogue(dialogue: Sequence[str]) -> str:
    return "\n".join(dialogue)


def build_schema(include_dialogue: bool = True) -> pa.schema:
    traits = Big5InferenceModel.CANONICAL_TRAITS
    trait_struct = pa.struct([(trait, pa.float32()) for trait in traits])

    scores_fields = [
        pa.field("journal_entry1_scores", trait_struct),
        pa.field("journal_entry2_scores", trait_struct),
    ]
    if include_dialogue:
        scores_fields.append(pa.field("dialogue_scores", trait_struct))

    fields = [
        pa.field("split", pa.string()),
        pa.field("author_fullname1", pa.string()),
        pa.field("author_fullname2", pa.string()),
        pa.field("id1", pa.string()),
        pa.field("id2", pa.string()),
        pa.field("dialogue", pa.list_(pa.string())),
    ]
    fields.extend(scores_fields)
    return pa.schema(fields)


def score_batch(
    buffer: List[Dict[str, Any]],
    model: Big5InferenceModel,
    batch_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    if not buffer:
        return []

    traits = Big5InferenceModel.CANONICAL_TRAITS

    j1_texts = [str(item.get("journal_entry1", "")) for item in buffer]
    j2_texts = [str(item.get("journal_entry2", "")) for item in buffer]
    dialogue_lists = [normalize_dialogue(item.get("dialogue", [])) for item in buffer]

    journal_scores = model.score_texts(j1_texts + j2_texts, batch_size=batch_size)
    j1_scores = journal_scores[: len(buffer)]
    j2_scores = journal_scores[len(buffer) :]

    dialogue_texts = [concat_dialogue(dialogue) for dialogue in dialogue_lists]
    dialogue_scores = model.score_texts(dialogue_texts, batch_size=batch_size)

    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(buffer):
        j1 = {trait: float(j1_scores[idx][trait]) for trait in traits}
        j2 = {trait: float(j2_scores[idx][trait]) for trait in traits}
        ds = {trait: float(dialogue_scores[idx][trait]) for trait in traits}

        record: Dict[str, Any] = {
            "split": split,
            "author_fullname1": item.get("author_fullname1"),
            "author_fullname2": item.get("author_fullname2"),
            "id1": item.get("id1"),
            "id2": item.get("id2"),
            "dialogue": dialogue_lists[idx],
            "journal_entry1_scores": j1,
            "journal_entry2_scores": j2,
            "dialogue_scores": ds,
        }
        records.append(record)
    return records


def process(
    inputs: List[str],
    output: str,
    model_name: str,
    cache_dir: str,
    buffer_size: int,
    batch_size: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    logger.info("Loading Big Five model '%s' (cache_dir=%s)", model_name, cache_dir)
    model = Big5InferenceModel(model_name=model_name, cache_dir=cache_dir)
    schema = build_schema(include_dialogue=True)
    logger.info("Writing output Parquet to %s", output)
    writer = pq.ParquetWriter(output, schema)

    try:
        for path in inputs:
            logger.info("Processing input %s", path)
            
            # Extract split name from filename (e.g., "train" from "train.json")
            basename = os.path.basename(path)
            split = os.path.splitext(basename)[0]
            logger.info("Split label: %s", split)
            
            file_size = os.path.getsize(path)
            
            with open(path, "rb") as f:
                pbar = tqdm(
                    total=file_size,
                    desc=f"Scoring {os.path.basename(path)}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
                
                buffer: List[Dict[str, Any]] = []
                processed: List[Dict[str, Any]] = []
                file_processed = 0
                file_written = 0
                last_pos = 0

                for obj in ijson.items(f, "item"):
                    buffer.append(obj)
                    file_processed += 1
                    
                    # Update progress based on file position
                    current_pos = f.tell()
                    pbar.update(current_pos - last_pos)
                    last_pos = current_pos
                    pbar.set_postfix(records=file_processed, scored=file_written)
                    
                    if len(buffer) >= buffer_size:
                        processed = score_batch(buffer, model, batch_size, split)
                        if processed:
                            table = pa.Table.from_pylist(processed, schema=schema)
                            writer.write_table(table)
                            file_written += len(processed)
                            pbar.set_postfix(records=file_processed, scored=file_written)
                        buffer.clear()

                if buffer:
                    processed = score_batch(buffer, model, batch_size, split)
                    if processed:
                        table = pa.Table.from_pylist(processed, schema=schema)
                        writer.write_table(table)
                        file_written += len(processed)
                        pbar.set_postfix(records=file_processed, scored=file_written)
                    buffer.clear()
                
                # Ensure we reach 100%
                pbar.update(file_size - last_pos)
                pbar.close()
            
            logger.info(
                "Finished %s: streamed=%d records, scored=%d records (split=%s)",
                path,
                file_processed,
                file_written,
                split,
            )
    finally:
        writer.close()
        logger.info("Closed Parquet writer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score JIC dataset entries on all Big Five traits and write a Parquet file "
            "containing the original openness subset fields plus full OCEAN scores."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    config_dir = config_path.parent
    base_dir_value = config.get("base_dir")
    if base_dir_value:
        base_dir = Path(base_dir_value)
        if not base_dir.is_absolute():
            base_dir = (config_dir / base_dir).resolve()
    else:
        base_dir = config_dir

    def resolve_path(path_like: Any) -> Path:
        path = Path(str(path_like))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    log_level_name = str(config.get("log_level", "INFO")).upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    if log_level_name not in level_map:
        raise ValueError(f"Unknown log_level '{log_level_name}'.")

    logging.basicConfig(
        level=level_map[log_level_name],
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Loaded configuration from %s", config_path)

    inputs = config.get("inputs")
    if not inputs or not isinstance(inputs, list):
        raise ValueError("Config must include 'inputs' as a non-empty list of JSON file paths.")
    input_paths = [str(resolve_path(p)) for p in inputs]

    output = config.get("output")
    if not output or not isinstance(output, str):
        raise ValueError("Config must include 'output' as a destination Parquet file path.")
    output_path = str(resolve_path(output))

    model_name = config.get("model_name", "Minej/bert-base-personality")
    cache_dir = config.get(
        "cache_dir", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = str(resolve_path(cache_dir))
    buffer_size = int(config.get("buffer_size", 256))
    batch_size = int(config.get("batch_size", 64))

    process(
        inputs=input_paths,
        output=output_path,
        model_name=model_name,
        cache_dir=cache_dir,
        buffer_size=buffer_size,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()

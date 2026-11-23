"""Download the pinned semantic model into the deployment artifact."""

from huggingface_hub import snapshot_download

from app.services.semantic_matcher import MODEL_ID, MODEL_PATH, MODEL_REVISION


MODEL_FILES = [
    "added_tokens.json",
    "config.json",
    "config_sentence_transformers.json",
    "data_config.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
]


def main() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_PATH),
        allow_patterns=MODEL_FILES,
    )
    if not (MODEL_PATH / "modules.json").is_file():
        raise RuntimeError("Pinned semantic model download is incomplete.")
    print(f"Preloaded semantic model revision {MODEL_REVISION}.")


if __name__ == "__main__":
    main()

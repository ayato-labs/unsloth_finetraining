from src.common.logger import logger


def align_model_embeddings(model, tokenizer, pad_to_multiple_of: int = 8) -> None:
    """ベースモデルとトークナイザーの語彙数が一致しない場合、動的にリサイズを実行する共通ヘルパー関数"""
    target_vocab_size = len(tokenizer)
    current_vocab_size = model.get_input_embeddings().weight.shape[0]

    if current_vocab_size != target_vocab_size or (current_vocab_size % pad_to_multiple_of != 0):
        model.resize_token_embeddings(target_vocab_size, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False)
        new_vocab_size = model.get_input_embeddings().weight.shape[0]
        logger.info(
            f"token_embeddings_aligned: resized from {current_vocab_size} to {new_vocab_size} "
            f"(target len={target_vocab_size}, pad_multiple={pad_to_multiple_of})"
        )

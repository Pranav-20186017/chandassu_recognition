"""One architecture factory for training and artifact loading."""

from transformers import T5Config, T5EncoderModel

from .direct import ByteMetreEncoder, TeluguMetreCNN


def build_model(config, vocabulary, encoder_config=None, pretrained=True):
    if config["model_type"] == "cnn":
        return TeluguMetreCNN(
            len(vocabulary.tokens) + 2, config["width"], tuple(config["kernel_sizes"]), config["dropout"]
        )
    if encoder_config is not None:
        encoder = T5EncoderModel(T5Config.from_dict(encoder_config))
    elif pretrained:
        encoder, info = T5EncoderModel.from_pretrained(
            config["base_model"], revision=config["model_revision"], output_loading_info=True
        )
        if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
            raise RuntimeError(f"Incomplete pretrained encoder: {info}")
    else:
        raise ValueError("ByT5 artifact must supply its saved encoder configuration")
    if config.get("gradient_checkpointing"):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return ByteMetreEncoder(encoder, config["dropout"])

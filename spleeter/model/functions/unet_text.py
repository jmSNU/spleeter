from functools import partial
from typing import Any, Dict, Iterable, Optional

import tensorflow as tf  # type: ignore
from tensorflow.compat.v1 import logging  # type: ignore
from tensorflow.compat.v1.keras.initializers import he_uniform  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    ELU,
    BatchNormalization,
    Concatenate,
    Dropout,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax,
)
tf.compat.v1.enable_eager_execution() 
tf.executing_eagerly()

from . import apply
from transformers import TFBertModel, AutoTokenizer


__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def _get_conv_activation_layer(params: Dict) -> Any:
    """
    Parameters:
        params (Dict):
            Model parameters.

    Returns:
        Any:
            Required Activation function.
    """
    conv_activation: str = str(params.get("conv_activation"))
    if conv_activation == "ReLU":
        return ReLU()
    elif conv_activation == "ELU":
        return ELU()
    return LeakyReLU(0.2)


def _get_deconv_activation_layer(params: Dict) -> Any:
    """
    Parameters:
        params (Dict):
            Model parameters.

    Returns:
        Any:
            Required Activation function.
    """
    deconv_activation: str = str(params.get("deconv_activation"))
    if deconv_activation == "LeakyReLU":
        return LeakyReLU(0.2)
    elif deconv_activation == "ELU":
        return ELU()
    return ReLU()


def apply_unet(
    input_tensor: tf.Tensor, # (None, 512, 1024, 2)
    output_name: str = "output",
    params: Dict = {},
    output_mask_logit: bool = False,
) -> tf.Tensor:
    prompts = """
        This track represents ordinary media content situation that includes the dialogue with background musics.
        You should separate this into dialogue, lyrics, background musics, and noise.
    """
    # print(f"input tensor shape : {input_tensor.shape}")
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    text_input = tokenizer(
        prompts,
        padding="max_length",      # Pad to the max sequence length
        truncation=True,           # Truncate to fit the model's max input size
        max_length=128,            # Set max input length (adjust as needed)
        return_tensors="tf"        # Return as TensorFlow tensors
    )

    conv_n_filters = params.get("conv_n_filters", [16, 32, 64, 128, 256, 512])
    kernel_initializer = tf.keras.initializers.HeUniform(seed=50)

    # Encoder
    conv_layers = []
    x = input_tensor
    for n_filters in conv_n_filters:
        x = tf.keras.layers.Conv2D(
            filters=n_filters, kernel_size=(5, 5), strides=(2, 2),
            padding="same", kernel_initializer=kernel_initializer
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        conv_layers.append(x)

    # Pretrained Text Encoder
    bert_encoder = TFBertModel.from_pretrained("google-bert/bert-base-uncased")
    text_embeddings = bert_encoder(text_input)[1]  # CLS token output
    text_embeddings = tf.keras.layers.Dense(512, activation="relu")(text_embeddings)  # Project to match audio

    # Fusion at bottleneck
    audio_bottleneck = conv_layers[-1]
    text_bottleneck = tf.keras.layers.Dense(
        audio_bottleneck.shape[-1],
        activation="relu"
    )(text_embeddings)

    text_bottleneck = tf.expand_dims(tf.expand_dims(text_bottleneck, 1), 1)
    text_bottleneck_broadcast = tf.tile(text_bottleneck, [1, tf.shape(audio_bottleneck)[1], tf.shape(audio_bottleneck)[2], 1])
    fused_bottleneck = Concatenate(axis=-1)([audio_bottleneck, text_bottleneck_broadcast])

    # Decoder
    for conv_layer, n_filters in zip(reversed(conv_layers[:-1]), reversed(conv_n_filters[:-1])):
        fused_bottleneck = tf.keras.layers.Conv2DTranspose(
            filters=n_filters, kernel_size=(5, 5), strides=(2, 2),
            padding="same", kernel_initializer=kernel_initializer
        )(fused_bottleneck)
        fused_bottleneck = BatchNormalization()(fused_bottleneck)
        fused_bottleneck = Concatenate()([fused_bottleneck, conv_layer])
        fused_bottleneck = Dropout(0.5)(fused_bottleneck)
        fused_bottleneck = tf.keras.layers.ReLU()(fused_bottleneck)

    fused_bottleneck = tf.keras.layers.Conv2DTranspose(
        filters = 1, kernel_size = (5,5), strides = (2,2),
        padding = "same", kernel_initializer = kernel_initializer
    )(fused_bottleneck)

    # Final layer
    final_output = tf.keras.layers.Conv2D(
        filters=2, kernel_size=(4, 4), padding="same",
        activation="sigmoid" if not output_mask_logit else None,
        kernel_initializer=kernel_initializer,
    )(fused_bottleneck)

    if not output_mask_logit:
        final_output = Multiply(name=output_name)([final_output, input_tensor])

    return final_output


def unet(
    input_tensor: tf.Tensor, instruments: Iterable[str], params: Optional[Dict] = None
) -> Dict:
    return apply(apply_unet, input_tensor, instruments, params)


def softmax_unet(
    input_tensor: tf.Tensor, instruments: Iterable[str], params: Dict = {}
) -> Dict:
    """
    Apply softmax to multitrack unet in order to have mask suming to one.

    Parameters:
        input_tensor (tf.Tensor):
            Tensor to apply blstm to.
        instruments (Iterable[str]):
            Iterable that provides a collection of instruments.
        params (Dict):
            (Optional) dict of BLSTM parameters.

    Returns:
        Dict:
            Created output tensor dict.
    """
    logit_mask_list = []
    for instrument in instruments:
        out_name = f"{instrument}_spectrogram"
        logit_mask_list.append(
            apply_unet(
                input_tensor,
                output_name=out_name,
                params=params,
                output_mask_logit=True,
            )
        )
    masks = Softmax(axis=4)(tf.stack(logit_mask_list, axis=4))
    output_dict = {}
    for i, instrument in enumerate(instruments):
        out_name = f"{instrument}_spectrogram"
        output_dict[out_name] = Multiply(name=out_name)([masks[..., i], input_tensor])
    return output_dict

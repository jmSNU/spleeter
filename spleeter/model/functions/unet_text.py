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
from transformers import TFDistilBertModel, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

def apply_unet(
    input_tensor: tf.Tensor,  # (None, 512, 1024, 2)
    text_embeddings : tf.Tensor,
    output_name: str = "output",
    params: dict = {},
    output_mask_logit: bool = False,
) -> tf.Tensor:

    # Encoder
    conv_n_filters = params.get("conv_n_filters", [16, 32, 64, 128, 256, 512])
    kernel_initializer = tf.keras.initializers.HeUniform(seed=50)

    conv_layers = []
    x = input_tensor
    for n_filters in conv_n_filters:
        x = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        conv_layers.append(x)

    # Fusion at bottleneck
    audio_bottleneck = conv_layers[-1]  # Shape: (batch_size, width, height, channels)

    # Get batch_size, width, height from audio_bottleneck
    batch_size = tf.shape(audio_bottleneck)[0]
    width = tf.shape(audio_bottleneck)[1]
    height = tf.shape(audio_bottleneck)[2]

    # Tile text_embeddings to match the batch_size of audio_bottleneck
    text_embeddings_tiled = tf.tile(text_embeddings, [batch_size, 1, 1])  # Shape: (batch_size, seq_length, 512)

    # Project text_embeddings to match the channel dimension of audio_bottleneck
    text_bottleneck = tf.keras.layers.Dense(
        audio_bottleneck.shape[-1], activation="relu"
    )(text_embeddings_tiled)  # Shape: (batch_size, seq_length, channels)

    # Pool over the sequence length to get a fixed-size representation
    text_bottleneck_pooled = tf.reduce_mean(text_bottleneck, axis=1)  # Shape: (batch_size, channels)

    # Expand dimensions to align with the spatial dimensions of audio_bottleneck
    text_bottleneck_expanded = tf.expand_dims(tf.expand_dims(text_bottleneck_pooled, 1), 1)  # Shape: (batch_size, 1, 1, channels)

    # Broadcast to match the spatial dimensions of audio_bottleneck
    text_bottleneck_broadcast = tf.tile(
        text_bottleneck_expanded, [1, width, height, 1]
    )  # Shape: (batch_size, width, height, channels)

    # Concatenate along the channels axis
    fused_bottleneck = Concatenate(axis=-1)([audio_bottleneck, text_bottleneck_broadcast])

    # Decoder
    for conv_layer, n_filters in zip(
        reversed(conv_layers[:-1]), reversed(conv_n_filters[:-1])
    ):
        fused_bottleneck = tf.keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )(fused_bottleneck)
        fused_bottleneck = BatchNormalization()(fused_bottleneck)
        fused_bottleneck = Concatenate()([fused_bottleneck, conv_layer])
        fused_bottleneck = Dropout(0.5)(fused_bottleneck)
        fused_bottleneck = tf.keras.layers.ELU()(fused_bottleneck)

    fused_bottleneck = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(fused_bottleneck)

    # Final layer
    final_output = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=(4, 4),
        padding="same",
        activation="sigmoid" if not output_mask_logit else None,
        kernel_initializer=kernel_initializer,
    )(fused_bottleneck)

    if not output_mask_logit:
        final_output = tf.keras.layers.Multiply(name=output_name)([final_output, input_tensor])

    return final_output


def unet(
    input_tensor: tf.Tensor, instruments: Iterable[str], text_embed : tf.Tensor, params: Optional[Dict] = None
) -> Dict:
    output_dict: Dict = {}
    for instrument in instruments:
        out_name = f"{instrument}_spectrogram"
        output_dict[out_name] = apply_unet(
            input_tensor, text_embed, output_name=out_name, params=params or {}
        )
    return output_dict


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

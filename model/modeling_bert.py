import torch
import transformers as tf
import loralib as lora
from torch import nn
from transformers import BertModel, BertModelConfig
from loralib.layers import Embedding, Linear, MergedLinear, Conv2d
from loralib.utils import mark_only_lora_as_trainable, lora_state_dict

# ! deprecated use config dict rewrite
class LoraBertConfig(BertModelConfig):
    def __init__(self, apply_lora=False, lora_alpha=None, lora_r=None) -> None:
        super().__init__()
        self.apply_lora = apply_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha


_CONFIG_FOR_DOC = "LoraBertConfig"


class LoraBertModel(BertModel):
    def __init__(self, config, apply_lora, lora_r, lora_alpha):
        super().__init__()
        self.config = config
        self.config["apply_lora"] = apply_lora
        self.config["lora_r"] = lora_r
        self.config["lora_alpha"] = lora_alpha

        self.encoder = LoraBertEncoder(self.config)
        self.init_weights()


class LoraBertSelfAttention(tf.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        """Rewrite the Linear and other function use BertSelfAttention"""
        super().__init__()

        if config.apply_lora:
            self.query = lora.Linear(
                config.hidden_size,
                self.all_head_size,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.key = lora.Linear(
                config.hidden_size,
                self.all_head_size,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.value = lora.Linear(
                config.hidden_size,
                self.all_head_size,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            print("Use Lora Linear")
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)


class LoraBertAttention(tf.BertAttention):
    def __init__(self, config):
        super().__init__()
        self.self = LoraBertAttention(config)


class LoraBertLayer(tf.BertLayer):
    def __init__(self, config):
        super().__init__()
        self.attention = LoraBertAttention(config)
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LoraBertAttention(config)


class LoraBertEncoder(tf.BertEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [LoraBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

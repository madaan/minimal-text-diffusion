from transformers import AutoConfig

# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder
import torch

import torch as th
import torch.nn as nn
from src.modeling.diffusion.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class TransformerNetModel(nn.Module):
    """
    A transformer model to be used in Diffusion Model Training.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes. TODO for the next version
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        init_pretrained,
        freeze_embeddings,
        use_pretrained_embeddings,
        dropout=0,
        use_checkpoint=False,
        num_heads=1,
        config=None,
        config_name="bert-base-uncased",
        vocab_size=None,
        logits_mode=1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            # config.hidden_size = 512

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.logits_mode = logits_mode
        self.vocab_size = vocab_size
        self.init_pretrained = init_pretrained
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.config = config
        self.config_name = config_name

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        self.build_xstart_predictor()
        self.build_input_output_projections()
        self.build_embeddings()

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def build_xstart_predictor(self):
        if self.init_pretrained:
            from transformers.models.bert.modeling_bert import BertModel

            temp_bert = BertModel.from_pretrained(self.config_name, config=self.config)
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
        else:
            self.input_transformers = BertEncoder(self.config)

    def build_input_output_projections(self):
        if self.use_pretrained_embeddings:
            self.input_up_proj = nn.Identity()
            self.output_down_proj = nn.Identity()
        else:  # need to adapt the model to the embedding size
            self.input_up_proj = nn.Sequential(
                nn.Linear(self.in_channels, self.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )

            self.output_down_proj = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.config.hidden_size, self.out_channels),
            )

    def build_embeddings(self):
        if self.use_pretrained_embeddings:
            from transformers.models.bert.modeling_bert import BertModel

            temp_bert = BertModel.from_pretrained(self.config_name, config=self.config)
            self.word_embedding = temp_bert.embeddings.word_embeddings
            self.position_embeddings = temp_bert.embeddings.position_embeddings
        else:
            self.word_embedding = nn.Embedding(self.vocab_size, self.in_channels)
            self.position_embeddings = nn.Embedding(
                self.config.max_position_embeddings, self.config.hidden_size
            )
        
        self.lm_head = nn.Linear(self.in_channels, self.word_embedding.weight.shape[0])

        if self.freeze_embeddings:
            self.word_embedding.weight.requires_grad = False
            self.position_embeddings.weight.requires_grad = False

        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, timesteps, y=None, src_ids=None, src_mask=None, attention_mask=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb.unsqueeze(1).expand(-1, seq_length, -1)
        )
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # https://github.com/huggingface/transformers/blob/e95d433d77727a9babadf008dd621a2326d37303/src/transformers/modeling_utils.py#L700
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        input_trans_hidden_states = self.input_transformers(
            emb_inputs, attention_mask=attention_mask
        ).last_hidden_state

        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h

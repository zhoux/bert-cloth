# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import sys
sys.path.append('..')

from pytorch_pretrained_bert.file_utils import cached_path, PYTORCH_PRETRAINED_BERT_CACHE
from allennlp.modules.elmo import Elmo
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# ====== ELMo Predictions =======

class ELMoLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ELMoLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta



class ELMoPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act):
        super(ELMoPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act] \
            if isinstance(hidden_act, str) else hidden_act
        self.LayerNorm = ELMoLayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ELMoLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, hidden_act, embedding_size, vocab_size):
        super(ELMoLMPredictionHead, self).__init__()
        self.transform = ELMoPredictionHeadTransform(hidden_size, hidden_act)

        self.decoder = nn.Linear(embedding_size,
                                 vocab_size,
                                 bias=True)
        #self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ELMoOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size, hidden_act, embedding_size, vocab_size):
        super(ELMoOnlyMLMHead, self).__init__()
        self.predictions = ELMoLMPredictionHead(hidden_size, hidden_act,
                                                embedding_size, vocab_size)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

# ====== END ELMo Predictions =======


class ELMoForCLOTH(nn.Module):

    def __init__(self):
        super(ELMoForCLOTH, self).__init__()
        # TODO Define hidden_size, hidden_act, embedding_size, vocab_size
        # The following are BERT specific parameters
        #embedding_size = 768
        embedding_size = 1024   # ELMo output size??
        vocab_size = 27247
        hidden_size = 1024 # ELMo output is 1024
        #hidden_size = 512 # BERT output is 512
        char_embedding_size = 50
        ops_char_size = char_embedding_size * 1
        num_chars = 262
        hidden_act = 'gelu'

        # Use AllenNLP ELMo model here
        self.elmo = Elmo(options_file, weight_file, 1,
                         dropout=0, requires_grad=True)
        # Use ELMo task specific layers here
        self.cls = ELMoOnlyMLMHead(hidden_size, hidden_act,
                                   embedding_size, vocab_size)
        # Skip weight initialization
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.vocab_size = vocab_size
        self.ops_char_size = ops_char_size
        self.num_chars = num_chars
    
    def accuracy(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).float()
        
    def forward(self, inp, tgt):
        '''
        input: article -> bsz X alen, 
        option -> bsz X opnum X 4 X olen
        output: bsz X opnum 
        '''
        articles, articles_mask, ops, ops_mask, ops_id, ops_id_mask, question_pos, mask, high_mask = inp 
        #import pdb; pdb.set_trace()
        bsz = ops.size(0)
        opnum = ops.size(1)   

        #out_dict = self.elmo(articles, attention_mask = articles_mask)
        # We do not need articles_mask because AllenNLP ELMo handles this
        # by itself. Long sequences are 0.
        out_dict = self.elmo(articles)
        #import pdb; pdb.set_trace()

        #out_elmo_embeddings = tuple(out_dict['elmo_representations'])
        #out = torch.cat(out_elmo_embeddings, -1)
        out = out_dict['elmo_representations'][0]
        out_masks = out_dict['mask']        # should be same as articles mask

        question_pos = question_pos.unsqueeze(-1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))

        #import pdb; pdb.set_trace()

        out = torch.gather(out, 1, question_pos)
        out = self.cls(out)

        #import pdb; pdb.set_trace()
        #convert ops to one hot
        out = out.view(bsz, opnum, 1, self.vocab_size)    # (4 x 20 x 1 x 261)
        out = out.expand(bsz, opnum, 4, self.vocab_size)  # (4 x 20 x 4 x 261)
        ops = ops.view(bsz, opnum, 4, -1)               # (4 x 20 x 4 x 150)
        #out = torch.gather(out, 3, ops)
        out = torch.gather(out, 3, ops_id)

        #mask average pooling
        #ops_mask = ops_mask.view(bsz, opnum, 4, -1)
        ops_id_mask = ops_id_mask.view(bsz, opnum, 4, -1)
        #out = out * ops_mask
        out = out * ops_id_mask
        out = out.sum(-1)
        #out = out/(ops_mask.sum(-1))
        out = out/(ops_id_mask.sum(-1))
        
        out = out.view(-1, 4)
        tgt = tgt.view(-1,)
        loss = self.loss(out, tgt)
        acc = self.accuracy(out, tgt)
        loss = loss.view(bsz, opnum)
        acc = acc.view(bsz, opnum)
        loss = loss * mask
        acc = acc * mask
        acc = acc.sum(-1)
        acc_high = (acc * high_mask).sum()
        acc = acc.sum()
        acc_middle = acc - acc_high

        loss = loss.sum()/(mask.sum())
        return loss, acc, acc_high, acc_middle
                           
    def init_zero_weight(self, shape):
        weight = next(self.parameters())
        return weight.new_zeros(shape)    

#from .file_utils import      
if __name__ == '__main__':
    bsz = 32
    max_length = 50
    max_olen = 3
    articles = torch.zeros(bsz, max_length).long()
    articles_mask = torch.ones(articles.size())
    ops = torch.zeros(bsz, 4, max_olen).long()
    ops_mask = torch.ones(ops.size())
    question_id = torch.arange(bsz).long()
    question_pos = torch.arange(bsz).long()
    ans = torch.zeros(bsz).long()
    inp = [articles, articles_mask, ops, ops_mask, question_id, question_pos]
    tgt = ans
    model = BertForCloth.from_pretrained('bert-base-uncased',
          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
    loss, acc = model(inp, tgt)
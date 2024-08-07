import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

device = "cuda" if torch.cuda.is_available() else "cpu"


class MiniSeg(nn.Module):
    def __init__(self, sentence_encoder, doc_encoder):
        super(MiniSeg, self).__init__()
        self.num_labels = 2
        self.sentence_encoder = sentence_encoder
        self.doc_encoder = doc_encoder

    def forward(self, texts, labels_list):
        batch_sentence_embeddings = [
            self._get_sentence_encoding(
                self.sentence_encoder(**text), text["attention_mask"]
            )
            for text in texts
        ]

        lengths = torch.tensor([len(text) for text in batch_sentence_embeddings]).to(
            device
        )

        padded_embeddings = rnn_utils.pad_sequence(
            batch_sentence_embeddings, batch_first=True
        )

        masks = self._generate_masks(
            lengths,
            padded_embeddings.size(1),
            padded_embeddings.size(0),
        ).float()

        result = self.doc_encoder(
            inputs_embeds=padded_embeddings,
            attention_mask=masks,
            labels=labels_list,
            return_dict=True,
        )

        return result

    def _generate_masks(self, lengths, max_len, batch_size):
        return torch.arange(max_len).to(device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)

    def _get_sentence_encoding(self, encoder_output, attention_mask):
        # Perform pooling
        sentence_embeddings = self.mean_pooling(encoder_output, attention_mask)

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.sentence_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )
        self.doc_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

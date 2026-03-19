import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        max_len = lengths.max().item()
        embeddings = self.embedding(indices[:, :max_len])
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device

        # Encode prefix with BOS
        tokens = [self.dataset.bos_id]
        if prefix:
            tokens.extend(self.dataset.text2ids(prefix))

        # Pass prefix through RNN to build hidden state
        hidden = None
        for token in tokens:
            inp = torch.tensor([[token]], device=device)
            output, hidden = self.rnn(self.embedding(inp), hidden)

        # Generate new tokens one by one
        generated_ids = []
        for _ in range(self.max_length - len(tokens)):
            logits = self.linear(output.squeeze(1)) / temp
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            if next_token == self.dataset.eos_id:
                break
            generated_ids.append(next_token)
            inp = torch.tensor([[next_token]], device=device)
            output, hidden = self.rnn(self.embedding(inp), hidden)

        return prefix + self.dataset.ids2text(generated_ids)

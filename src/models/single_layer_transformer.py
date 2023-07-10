import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class BeamSplitterAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_len=1024):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_len = max_len

        self.chunk_self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.global_self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # determine the chunk size
        chunk_size = self.max_len
        num_chunks = seq_len // chunk_size

        # create chunks and perform attention
        x_chunks = x.view(batch_size, num_chunks, chunk_size, -1)  # (batch_size, num_chunks, chunk_size, d_model)
        x_chunks = x_chunks.transpose(1, 2)  # (batch_size, chunk_size, num_chunks, d_model)

        chunk_attn_outputs = []
        for i in range(chunk_size):
            chunk = x_chunks[:, i, :, :]  # (batch_size, num_chunks, d_model)
            chunk = chunk.transpose(0, 1)  # (num_chunks, batch_size, d_model)
            chunk_attn_output, _ = self.chunk_self_attn(chunk, chunk, chunk)  # (num_chunks, batch_size, d_model)
            chunk_attn_outputs.append(chunk_attn_output)

        x_chunks_attn = torch.stack(chunk_attn_outputs, dim=1)  # (batch_size, chunk_size, num_chunks, d_model)
        x_chunks_attn = x_chunks_attn.transpose(1, 2)  # (batch_size, num_chunks, chunk_size, d_model)
        x_chunks_attn = x_chunks_attn.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, d_model)

        # perform global attention
        x_chunks_attn = x_chunks_attn.transpose(0, 1)  # (seq_len, batch_size, d_model)
        global_attn_output, _ = self.global_self_attn(x_chunks_attn, x_chunks_attn, x_chunks_attn)  # (seq_len, batch_size, d_model)
        global_attn_output = global_attn_output.transpose(0, 1)  # (batch_size, seq_len, d_model)

        return global_attn_output

class SingleLayerTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, dropout=0.1, max_len=1024):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=embedding_dim*4, dropout=dropout, activation='relu'),
            num_layers=1
        )
        # Replace MultiheadAttention in TransformerEncoderLayer with BeamSplitterAttention
        self.transformer.encoder.layers[0].self_attn = BeamSplitterAttention(embedding_dim, num_heads, dropout, max_len)
        
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        embeddings = self.embedding(x) * np.sqrt(self.embedding_dim)
        embeddings = self.pos_encoder(embeddings)
        transformer_output = self.transformer(embeddings)
        output = self.fc_out(transformer_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
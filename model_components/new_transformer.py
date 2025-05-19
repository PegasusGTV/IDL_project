import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import TimeSeriesPositionalEncoding
from .masks import PadMask, CausalMask
from .time2vec_new import Time2Vec
import math

class FeatureAttentionPool(nn.Module):
    """
    Pools over the feature axis using a learned query vector.
    Returns:
      - pooled:   [B*T, D]    single vector per time step
      - weights:  [B*T, Fm1]  importance of each feature
    """
    def __init__(self, d_model: int):
        super().__init__()
        # one learnable query per “sentence” of features
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, feat_tokens: torch.Tensor):
        # feat_tokens: [B*T, Fm1, D]
        B_T, Fm1, D = feat_tokens.shape

        # expand query to match batch*time
        q = self.query.expand(B_T, -1, -1)               # [B*T, 1, D]
        # compute unnormalized scores: [B*T, 1, Fm1]
        scores = torch.bmm(q, feat_tokens.transpose(1, 2)) / math.sqrt(D)
        # normalize to get importance weights
        attn_weights = F.softmax(scores, dim=-1)          # [B*T, 1, Fm1]
        # weighted sum of features
        pooled = torch.bmm(attn_weights, feat_tokens)     # [B*T, 1, D]
        pooled = pooled.squeeze(1)                        # [B*T, D]
        weights = attn_weights.squeeze(1)                 # [B*T, Fm1]
        return pooled, weights

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_time:int,
        d_freq: int,  # unused but kept for API compatibility
        dropout: float,
        forecast_horizon: int
    ):

        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_features = input_features
        self.d_model = d_model

        # Combined input dim: (F-1) features + 1 time
        self.combined_dim = d_model * (input_features - 1) + d_time
        self.tgt_combined_dim = d_model

        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            for _ in range(input_features - 1)
        ])
        self.time_embedding = Time2Vec(d_model, periodic_activation=torch.sin)

        self.id_embedding = nn.Embedding(input_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.feature_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            # nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.feature_pool = FeatureAttentionPool(d_model)

        self.tgt_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    
    def add_id(x):
        idx = torch.arange(F, device=x.device)
        return x + self.id_embedding(idx)

    def forward(self, x: torch.Tensor, tgt_shifted: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_features)
        tgt_shifted: (batch_size, forecast_horizon, 1)
        """
        batch_size, seq_len, _ = x.size()
        features = x[:, :, :-1]
        time = x[:, :, -1:]
        tokens = torch.stack([
            feature_embedding(features[:, :, i])
            for i, feature_embedding in enumerate(self.feature_embeddings)
        ], dim=2)
        # time = self.time_embedding(time)
        tokens = tokens + self.id_embedding.weight[None, None, :-1, :]
        # tokens = torch.cat([tokens, time], dim=2)
        B, T, Fm1, D = tokens.shape
        tokens = tokens.view(B*T, Fm1, D)
        encoded = self.feature_encoder(tokens)

        # encoded: [B*T, Fm1, D]
        pooled, feat_weights = self.feature_pool(encoded) # [B*T, D], [B*T, Fm1]

        # reshape back to (B, T, D)
        z = pooled.view(B, T, D)
        
        # if you want per‑feature weights for analysis:
        importance = feat_weights.view(B, T, Fm1)            # e.g. log or visualize

        # add time embedding
        time = self.time_embedding(time)
        time = time.view(batch_size, seq_len, self.d_model)
        z = z + time
        z = z.view(batch_size, seq_len, self.d_model)
        z = self.dropout(z)


        B, H, _ = tgt_shifted.shape
        tgt_embedding = self.tgt_embedding(tgt_shifted)
        tgt_embedding = tgt_embedding.view(B, H, self.d_model)
        tgt_embedding = self.dropout(tgt_embedding)

        # causal_mask = CausalMask(tgt_embedding).to(dtype=torch.float32, device=memory.device)
        # causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0, float(0.0))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(H).to(x.device)

        decoded = self.decoder(tgt=tgt_embedding, memory=z, tgt_mask=causal_mask)
        decoded = self.norm(decoded)
        output = self.output_proj(decoded)  # [B, T, 1]
        return output
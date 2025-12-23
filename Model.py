import torch
import torch.nn as nn


class ChessModel(nn.Module):
    """
    Transformer-only model with embedding-based inputs.

    Inputs:
        piece_ids    – (B,64) int64 piece IDs (0=empty, 1..12=pieces)
        global_state – (side, castle_bits, ep, hmc, rep) categorical tensors
        mask         – (B,4672) bool legal action mask (optional)
    Returns:
        logits – (B,4672)
        value  – (B,)
    """
    def __init__(self,
                 token_dim: int = 512,
                 n_heads: int = 16,
                 n_attn_layers: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.token_dim = token_dim
        self.piece_embed = nn.Embedding(13, token_dim)
        self.pos_embed = nn.Embedding(64, token_dim)
        self.side_embed = nn.Embedding(2, token_dim)
        self.castle_wk_embed = nn.Embedding(2, token_dim)
        self.castle_wq_embed = nn.Embedding(2, token_dim)
        self.castle_bk_embed = nn.Embedding(2, token_dim)
        self.castle_bq_embed = nn.Embedding(2, token_dim)
        self.ep_embed = nn.Embedding(9, token_dim)
        self.hmc_embed = nn.Embedding(13, token_dim)
        self.rep_embed = nn.Embedding(4, token_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_attn_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, 4672),
        )
        self.value_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, 1),
        )

        self.register_buffer("pos_ids", torch.arange(64), persistent=False)

    def forward(self,
                piece_ids: torch.Tensor,
                global_state: tuple[torch.Tensor, ...] | None = None,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if piece_ids.dtype != torch.long:
            piece_ids = piece_ids.long()

        pos_emb = self.pos_embed(self.pos_ids).unsqueeze(0)
        square_tokens = self.piece_embed(piece_ids) + pos_emb

        if global_state is None:
            batch = piece_ids.size(0)
            side = torch.zeros(batch, dtype=torch.long, device=piece_ids.device)
            castle_bits = torch.zeros((batch, 4), dtype=torch.long, device=piece_ids.device)
            ep = torch.zeros(batch, dtype=torch.long, device=piece_ids.device)
            hmc = torch.zeros(batch, dtype=torch.long, device=piece_ids.device)
            rep = torch.zeros(batch, dtype=torch.long, device=piece_ids.device)
        else:
            side, castle_bits, ep, hmc, rep = global_state
            side = side.long()
            castle_bits = castle_bits.long()
            ep = ep.long()
            hmc = hmc.long()
            rep = rep.long()

        cls_tok = self.side_embed(side)
        cls_tok = cls_tok + self.castle_wk_embed(castle_bits[:, 0])
        cls_tok = cls_tok + self.castle_wq_embed(castle_bits[:, 1])
        cls_tok = cls_tok + self.castle_bk_embed(castle_bits[:, 2])
        cls_tok = cls_tok + self.castle_bq_embed(castle_bits[:, 3])
        cls_tok = cls_tok + self.ep_embed(ep)
        cls_tok = cls_tok + self.hmc_embed(hmc)
        cls_tok = cls_tok + self.rep_embed(rep)
        cls_tok = cls_tok.unsqueeze(1)

        seq = torch.cat([cls_tok, square_tokens], dim=1)
        seq = self.transformer(seq)

        cls_out = seq[:, 0, :]
        logits = self.policy_head(cls_out)
        value = self.value_head(cls_out).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits, value


if __name__ == "__main__":
    model = ChessModel()
    dummy_ids = torch.zeros((2, 64), dtype=torch.long)
    dummy_global = (
        torch.zeros((2,), dtype=torch.long),
        torch.zeros((2, 4), dtype=torch.long),
        torch.zeros((2,), dtype=torch.long),
        torch.zeros((2,), dtype=torch.long),
        torch.zeros((2,), dtype=torch.long),
    )
    logits, value = model(dummy_ids, dummy_global)
    print("logits shape:", logits.shape)    # (2,4672)
    print("value shape:", value.shape)      # (2,)
    print("≈ # params:", sum(p.numel() for p in model.parameters()))

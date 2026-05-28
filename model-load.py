from pathlib import Path
import torch
from safetensors.torch import load_file
from braindecode.models import LUNA
from braindecode.util import set_random_seeds

set_random_seeds(seed=42, cuda=torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_dir  = Path("checkpoints/LUNA")
weights_path = ckpt_dir / "LUNA_base.safetensors"

# --- Architettura ---
model = LUNA(
    n_outputs=2,
    n_chans=64,
    n_times=1000,
    embed_dim=64,
    num_queries=4,
    depth=8,
)

# --- Carica checkpoint grezzo ---
raw_sd = load_file(str(weights_path), device="cpu")

# --- Remapping chiavi incompatibili ---
KEY_MAP = {
    # typo nel checkpoint originale
    "cross_attn.temparature": "cross_attn.temperature",
    # rename channel_location_embedder: rimuove il ".0." e la norm
    "channel_location_embedder.0.fc1.weight": "channel_location_embedder.fc1.weight",
    "channel_location_embedder.0.fc1.bias":   "channel_location_embedder.fc1.bias",
    "channel_location_embedder.0.fc2.weight": "channel_location_embedder.fc2.weight",
    "channel_location_embedder.0.fc2.bias":   "channel_location_embedder.fc2.bias",
    # .norm non esiste in braindecode, skippiamo (rimane unexpected)
}

# Chiavi da ignorare (presenti nel ckpt ma inutili per classificazione)
SKIP_KEYS = {
    "channel_emb.embeddings.weight",          # lookup nominale canali
    "channel_location_embedder.0.norm.weight",
    "channel_location_embedder.0.norm.bias",
    "cross_attn.ffn.norm.weight",
    "cross_attn.ffn.norm.bias",
    "decoder_head.decoder_linear.fc1.bias",
    "decoder_head.decoder_linear.fc1.weight",
    "decoder_head.decoder_linear.fc2.bias",
    "decoder_head.decoder_linear.fc2.weight",
    "decoder_head.decoder_pred.layers.0.linear1.bias",
    "decoder_head.decoder_pred.layers.0.linear1.weight",
    "decoder_head.decoder_pred.layers.0.linear2.bias",
    "decoder_head.decoder_pred.layers.0.linear2.weight",
    "decoder_head.decoder_pred.layers.0.multihead_attn.in_proj_bias",
    "decoder_head.decoder_pred.layers.0.multihead_attn.in_proj_weight",
    "decoder_head.decoder_pred.layers.0.multihead_attn.out_proj.bias",
    "decoder_head.decoder_pred.layers.0.multihead_attn.out_proj.weight",
    "decoder_head.decoder_pred.layers.0.norm1.bias",
    "decoder_head.decoder_pred.layers.0.norm1.weight",
    "decoder_head.decoder_pred.layers.0.norm2.bias",
    "decoder_head.decoder_pred.layers.0.norm2.weight",
    "decoder_head.decoder_pred.layers.0.norm3.bias",
    "decoder_head.decoder_pred.layers.0.norm3.weight",
    "decoder_head.decoder_pred.layers.0.self_attn.in_proj_bias",
    "decoder_head.decoder_pred.layers.0.self_attn.in_proj_weight",
    "decoder_head.decoder_pred.layers.0.self_attn.out_proj.bias",
    "decoder_head.decoder_pred.layers.0.self_attn.out_proj.weight",
    "decoder_head.norm.bias",
    "decoder_head.norm.weight",
}

remapped_sd = {}
for k, v in raw_sd.items():
    if k in SKIP_KEYS:
        continue
    new_k = KEY_MAP.get(k, k)
    remapped_sd[new_k] = v

# --- Carica con strict=False (solo final_layer mancante, come atteso) ---
missing, unexpected = model.load_state_dict(remapped_sd, strict=False)

print(f"Missing   ({len(missing)}): {missing}")
print(f"Unexpected({len(unexpected)}): {unexpected}")

# Atteso:
# Missing: solo final_layer.* (9 chiavi) — head classificazione, OK
# Unexpected: [] — zero

model = model.to(device)
print(f"\nLUNA pronto su {device}. Parametri totali: {sum(p.numel() for p in model.parameters()):,}")
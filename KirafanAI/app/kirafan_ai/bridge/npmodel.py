"""The policy/value network in numpy, for the portable WindBot bridge (no PyTorch).

export() turns a training checkpoint into an .npz; NumpyPolicy computes the same logits and value
as model.net.PolicyNet for one observation (batch of 1, so no padding masks are needed).
"""
import json

import numpy as np

NOOP_MASK = -1e8
META = ("__vocab__", "__config__", "__update__")


def export(checkpoint, out_path):
    """Checkpoint (.pt) -> .npz with the weights, the card vocabulary order and the config."""
    import torch
    from ..cards import get_db
    from ..model.net import build, fit_state_dict
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    db = get_db()
    db.set_vocab_order(ck["vocab"])
    model = build(len(db.vocab_codes))
    model.load_state_dict(fit_state_dict(ck["model"], model))
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    arrays["__vocab__"] = np.asarray(db.vocab_codes, dtype=np.int64)
    arrays["__config__"] = np.frombuffer(json.dumps(model.config).encode("utf-8"), dtype=np.uint8)
    arrays["__update__"] = np.asarray([ck.get("update") or 0], dtype=np.int64)
    np.savez(out_path, **arrays)


def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _relu(x):
    return np.maximum(x, 0)


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.w = {k: z[k] for k in z.files if k not in META}
        self.vocab = [int(c) for c in z["__vocab__"]]
        self.config = json.loads(bytes(z["__config__"]).decode("utf-8"))
        self.update = int(z["__update__"][0])
        self.layers = self.config["layers"]
        self.heads = self.config["heads"]
        self._widen()

    def _widen(self):
        """Zero weights for option / global features added after this model was exported (as
        model.net.fit_state_dict), so older bots keep working with the current feature code."""
        from ..model.features import GLOBAL_DIM, OPTION_DIM
        code = self.config.get("code_dim", 64)
        for name, dim, rest in (("o_in.0.weight", OPTION_DIM, 2 * code + 32 + 16), ("g_in.0.weight", GLOBAL_DIM, code + 16)):
            w = self.w[name]
            extra = dim + rest - w.shape[1]
            if extra > 0:
                old = dim - extra
                self.w[name] = np.concatenate([w[:, :old], np.zeros((w.shape[0], extra), w.dtype), w[:, old:]], 1)

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _mlp(self, x, name):
        return self._lin(_relu(self._lin(x, name + ".0")), name + ".2")

    def _emb(self, name, idx):
        return self.w[name + ".weight"][idx]

    def _attention(self, x, p):
        n, d = x.shape
        h = self.heads
        qkv = x @ self.w[p + "in_proj_weight"].T + self.w[p + "in_proj_bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(n, h, d // h).transpose(1, 0, 2)
        k = k.reshape(n, h, d // h).transpose(1, 0, 2)
        v = v.reshape(n, h, d // h).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(d // h)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        out = (s @ v).transpose(1, 0, 2).reshape(n, d)
        return out @ self.w[p + "out_proj.weight"].T + self.w[p + "out_proj.bias"]

    def forward(self, x):
        """x: features.pack() / net.to_arrays()-style arrays of one observation -> (logits, value)."""
        hint = x["hint"]
        g = self._mlp(np.concatenate([x["g"], self._emb("code_emb", hint[0]),
                                      self._emb("hint_idx_emb", min(max(int(hint[1]), 0), 31))]), "g_in")
        c = self._mlp(np.concatenate([x["cards"], self._emb("code_emb", x["card_codes"])], -1), "c_in") \
            if len(x["card_codes"]) else np.zeros((0, g.shape[0]), np.float32)
        oids = x["opt_ids"]
        o = self._mlp(np.concatenate([x["opts"], self._emb("code_emb", oids[:, 1]), self._emb("code_emb", oids[:, 2]),
                                      self._emb("kind_emb", oids[:, 0]),
                                      self._emb("desc_idx_emb", np.clip(oids[:, 3], 0, 31))], -1), "o_in")
        c_pad = np.concatenate([np.zeros((1, c.shape[1] if len(c) else g.shape[0]), np.float32), c], 0)
        o = o + c_pad[np.clip(oids[:, 4], 0, len(c_pad) - 1)]
        t = self.w["type_emb.weight"]
        seq = np.concatenate([(g + t[0])[None], c + t[1], o + t[2]], 0).astype(np.float32)
        for i in range(self.layers):
            p = "encoder.layers.%d." % i
            seq = seq + self._attention(_layer_norm(seq, self.w[p + "norm1.weight"], self.w[p + "norm1.bias"]),
                                        p + "self_attn.")
            y = _layer_norm(seq, self.w[p + "norm2.weight"], self.w[p + "norm2.bias"])
            seq = seq + self._lin(_relu(self._lin(y, p + "linear1")), p + "linear2")
        h = _layer_norm(seq, self.w["norm.weight"], self.w["norm.bias"])
        value = float(self._mlp(h[0], "value")[0])
        logits = self._mlp(h[1 + len(c):], "logit")[:, 0]
        logits = np.where(x["noop"] > 0.5, NOOP_MASK, logits)
        return logits, value


def to_arrays(enc):
    """features.pack() bytes -> numpy arrays (same as model.net.to_arrays, without PyTorch)."""
    from ..model.features import CARD_DIM, OPTION_DIM
    f32 = lambda k: np.frombuffer(enc[k], dtype=np.float32)  # noqa: E731
    i32 = lambda k: np.frombuffer(enc[k], dtype=np.int32).astype(np.int64)  # noqa: E731
    return {"g": f32("g"), "cards": f32("cards").reshape(-1, CARD_DIM), "card_codes": i32("card_codes"),
            "opts": f32("opts").reshape(-1, OPTION_DIM), "opt_ids": i32("opt_ids").reshape(-1, 5),
            "noop": f32("noop"), "hint": i32("hint")}

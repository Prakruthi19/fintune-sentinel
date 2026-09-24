"""Toy re-implementation of llama.cpp Q4_0 / bitsandbytes NF4 and a GGUF header, for explaining quantization.

    python scripts/quant_demo.py
"""
import struct, numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(3072, 3072)).astype(np.float32)   # one Llama-3.2-3B q_proj-sized matrix
W[rng.random(W.shape) < 1e-4] *= 20                              # a few outliers, like real weights

# ---------------- Q4_0 (ggml reference: quantize_row_q4_0_ref) ----------------
QK = 32
def quantize_q4_0(x):
    x = x.reshape(-1, QK)
    idx = np.abs(x).argmax(1)
    maxv = x[np.arange(len(x)), idx]                 # signed value with largest |x|
    d = maxv / -8                                    # scale: maps that value to -8
    inv = np.where(d != 0, 1 / d, 0)
    q = np.minimum(15, (x * inv[:, None] + 8.5).astype(np.int8)).astype(np.uint8)  # 0..15
    qs = q[:, :16] | (q[:, 16:] << 4)                # pack 2 nibbles per byte: j and j+16
    return d.astype(np.float16), qs                  # block = 2 B scale + 16 B = 18 B / 32 weights

def dequantize_q4_0(d, qs):
    lo, hi = (qs & 0x0F).astype(np.int8) - 8, (qs >> 4).astype(np.int8) - 8
    return (np.concatenate([lo, hi], 1) * d.astype(np.float32)[:, None]).ravel()

d, qs = quantize_q4_0(W)
W_q40 = dequantize_q4_0(d, qs).reshape(W.shape)

# ---------------- NF4 (bitsandbytes, used during QLoRA training) ----------------
NF4 = np.array([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0], np.float32)
def nf4_roundtrip(x, block=64):
    x = x.reshape(-1, block)
    absmax = np.abs(x).max(1, keepdims=True)
    codes = np.abs((x / absmax)[..., None] - NF4).argmin(-1)   # nearest of 16 normal-quantile levels
    return (NF4[codes] * absmax).ravel()
W_nf4 = nf4_roundtrip(W).reshape(W.shape)

def report(name, Wq, bits):
    err = np.abs(W - Wq)
    x = rng.normal(size=3072).astype(np.float32)
    rel = np.linalg.norm(W @ x - Wq @ x) / np.linalg.norm(W @ x)
    print(f"{name:6} {bits:>4} bits/w  size={W.size*bits/8/2**20:6.2f} MiB  "
          f"mean|err|={err.mean():.2e}  matmul rel.err={rel:.2%}")

print(f"FP32   {W.size*4/2**20:6.2f} MiB")
report("FP16", W.astype(np.float16).astype(np.float32), 16)
report("NF4", W_nf4, 4 + 32 / 64)          # + fp32 absmax per 64 (≈4.127 with double-quant)
report("Q4_0", W_q40, 18 * 8 / 32)         # 4.5
print("first block raw bytes:", d[0].tobytes().hex(), qs[0].tobytes().hex())
print("first 8 weights:", W.ravel()[:8].round(4))
print("dequantized    :", W_q40.ravel()[:8].round(4))

# ---------------- integer dot product (ggml_vec_dot_q4_0_q8_0, scalar version) ----------------
def quantize_q8_0(x):
    x = x.reshape(-1, QK); dd = np.abs(x).max(1) / 127
    return dd.astype(np.float16), np.round(x / dd[:, None]).astype(np.int8)
a = rng.normal(size=QK * 4).astype(np.float32)
ad, aq = quantize_q8_0(a)
wd, wqs = quantize_q4_0(W[0, :QK * 4])
sumf = 0.0
for i in range(len(wd)):
    lo, hi = (wqs[i] & 0x0F).astype(np.int32) - 8, (wqs[i] >> 4).astype(np.int32) - 8
    sumi = int((lo * aq[i, :16]).sum() + (hi * aq[i, 16:]).sum())      # pure int math (AVX2 does 32 at once)
    sumf += sumi * float(wd[i]) * float(ad[i])                           # one float multiply per block
print(f"dot: fp32={W[0,:QK*4] @ a:.5f}  int4xint8={sumf:.5f}")

# ---------------- write + read a tiny GGUF (v3) ----------------
def gstr(s): b = s.encode(); return struct.pack("<Q", len(b)) + b
kv = [("general.architecture", 8, gstr("llama")), ("llama.context_length", 4, struct.pack("<I", 131072))]
hdr = struct.pack("<IIQQ", 0x46554747, 3, 1, len(kv))
hdr += b"".join(gstr(k) + struct.pack("<I", t) + v for k, t, v in kv)
hdr += gstr("blk.0.attn_q.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 3072, 3072) + struct.pack("<IQ", 2, 0)
pad = (-len(hdr)) % 32
blob = hdr + b"\0" * pad + b"".join(d[i].tobytes() + qs[i].tobytes() for i in range(len(d)))
open("toy.gguf", "wb").write(blob)

buf = open("toy.gguf", "rb").read(); off = 0
def rd(fmt):
    global off; v = struct.unpack_from(fmt, buf, off); off += struct.calcsize(fmt); return v
def rs():
    global off; (n,) = rd("<Q"); s = buf[off:off + n].decode(); off += n; return s
magic, ver, n_t, n_kv = rd("<IIQQ")
print(f"\nmagic={buf[:4]} version={ver} tensors={n_t} kv={n_kv}")
for _ in range(n_kv):
    k = rs(); (t,) = rd("<I"); print("  ", k, "=", rs() if t == 8 else rd("<I")[0])
name = rs(); (nd,) = rd("<I"); dims = rd("<" + "Q" * nd); ttype, toff = rd("<IQ")
print(f"   tensor {name} dims={dims} ggml_type={ttype} (2=Q4_0) data_offset={toff}")
print(f"   file size={len(buf)/2**20:.2f} MiB vs FP16 {W.size*2/2**20:.2f} MiB")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration & Hyperparameters ---
MODEL_ID = "state-spaces/mamba-130m-hf"

# Mamba-130m Architecture Constants
D_MODEL = 768
EXPAND = 2
D_INNER = D_MODEL * EXPAND  # 1536
D_STATE = 16
D_CONV = 4
DT_RANK = D_MODEL // 16
N_LAYERS = 24

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# --- Helper Functions ---

def rms_norm(x, weight, eps=1e-5):
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(mean_sq + eps)
    return normed * weight

# --- The Core Logic: Unified Mamba Block (Updated) ---

class UnifiedMambaBlock(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.params = nn.ParameterDict()

        # Metadata to track operations. Note the new split keys.
        # NEW: add "systolic" flag (default False) without changing existing "type" usage
        self.layer_meta = {k: {"type": "standard", "systolic": False} for k in [
            "in_proj", "dt_proj", "out_proj",
            "x_proj_dt", "x_proj_b", "x_proj_c"  # <--- Split keys
        ]}

    # ----------------------------
    # NEW: quant_method = "abs" or "average"
    # ----------------------------
    def _quantize(self, tensor, method="abs", eps=1e-12):
        """
        symmetric per-tensor int8 quantization
        method:
          - "abs":     scale = max(abs(W)) / 127   (abs-max)
          - "average": scale = mean(abs(W)) / 127
        """
        t_abs = tensor.abs().float()

        if method == "abs":
            threshold = t_abs.max()
        elif method == "average":
            threshold = t_abs.mean()
        else:
            raise ValueError(f"Unknown quant method: {method}")

        scale = threshold / 127.0
        scale = torch.clamp(scale, min=eps)

        int8_val = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return int8_val, scale

    # ----------------------------
    # NEW: systolic-style tiled GEMM (software simulation)
    # ----------------------------
    def _matmul2d_systolic(self, A, B, tile_m=64, tile_n=64, tile_k=64):
        """
        Compute C = A @ B using blocked/tiled loops (systolic-like dataflow simulation).
        A: [M, K], B: [K, N], C: [M, N]
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"matmul shape mismatch: A {A.shape}, B {B.shape}"

        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

        for m0 in range(0, M, tile_m):
            m1 = min(m0 + tile_m, M)
            for n0 in range(0, N, tile_n):
                n1 = min(n0 + tile_n, N)
                acc = torch.zeros((m1 - m0, n1 - n0), device=A.device, dtype=A.dtype)
                for k0 in range(0, K, tile_k):
                    k1 = min(k0 + tile_k, K)
                    acc += A[m0:m1, k0:k1] @ B[k0:k1, n0:n1]
                C[m0:m1, n0:n1] = acc
        return C

    def _matmul2d(self, A, B, systolic=False):
        if not systolic:
            return A @ B
        return self._matmul2d_systolic(A, B)

    def _linear(self, x, w, systolic=False):
        """
        x: [B, S, IN]
        w: [OUT, IN] (PyTorch F.linear weight layout)
        returns: [B, S, OUT]
        """
        if not systolic:
            return F.linear(x, w)

        B, S, IN = x.shape
        OUT, IN2 = w.shape
        assert IN == IN2, f"linear shape mismatch: x {x.shape}, w {w.shape}"

        x2 = x.reshape(B * S, IN)                 # [BS, IN]
        y2 = self._matmul2d(x2, w.t().contiguous(), systolic=True)  # [BS, OUT]
        return y2.reshape(B, S, OUT)

    def load_weights(self, hf_layer):
        mixer = hf_layer.mixer

        # --- 1. Load Static SSM Params ---
        self.params["conv_weight"] = nn.Parameter(mixer.conv1d.weight.data)
        if mixer.conv1d.bias is not None:
            self.params["conv_bias"] = nn.Parameter(mixer.conv1d.bias.data)
        self.params["A_log"] = nn.Parameter(mixer.A_log.data)
        self.params["D"] = nn.Parameter(mixer.D.data)

        # --- 2. Prepare Weights for Processing ---
        # We manually split x_proj here before the loop
        x_proj_w = mixer.x_proj.weight.data
        # x_proj maps to [dt, B, C]
        w_dt, w_b, w_c = torch.split(x_proj_w, [DT_RANK, D_STATE, D_STATE], dim=0)

        # Create a list of (name, weight, bias) to iterate over uniformly
        # Note: x_proj usually has no bias in Mamba, but we handle None safely
        weights_to_process = [
            ("in_proj", mixer.in_proj.weight.data, mixer.in_proj.bias),
            ("dt_proj", mixer.dt_proj.weight.data, mixer.dt_proj.bias),
            ("out_proj", mixer.out_proj.weight.data, mixer.out_proj.bias),
            # The split layers (bias is likely None for these)
            ("x_proj_dt", w_dt, None),
            ("x_proj_b",  w_b,  None),
            ("x_proj_c",  w_c,  None),
        ]

        # --- 3. Process Projections (SVD / Quant / Standard) ---
        for name, W, bias_data in weights_to_process:

            # Get config for this specific projection
            cfg = self.config.get(name, {})
            rank = cfg.get("svd_rank", None)
            do_quant = cfg.get("quant", False)

            # NEW: read quant_method + systolic flag (default keeps old behavior)
            quant_method = cfg.get("quant_method", "abs")          # "abs" or "average"
            self.layer_meta[name]["systolic"] = cfg.get("systolic", False)

            # --- SVD ENABLED ---
            if rank is not None:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]

                if do_quant:
                    u_int8, u_scale = self._quantize(U, method=quant_method)
                    vh_int8, vh_scale = self._quantize(Vh, method=quant_method)
                    self.params[f"{name}_U_int8"] = nn.Parameter(u_int8, requires_grad=False)
                    self.params[f"{name}_U_scale"] = nn.Parameter(u_scale, requires_grad=False)
                    self.params[f"{name}_Vh_int8"] = nn.Parameter(vh_int8, requires_grad=False)
                    self.params[f"{name}_Vh_scale"] = nn.Parameter(vh_scale, requires_grad=False)
                    self.params[f"{name}_S"] = nn.Parameter(S)
                    self.layer_meta[name]["type"] = "svd_quant"
                else:
                    self.params[f"{name}_U"] = nn.Parameter(U)
                    self.params[f"{name}_S"] = nn.Parameter(S)
                    self.params[f"{name}_Vh"] = nn.Parameter(Vh)
                    self.layer_meta[name]["type"] = "svd"

            # --- NO SVD ---
            else:
                if do_quant:
                    w_int8, w_scale = self._quantize(W, method=quant_method)
                    self.params[f"{name}_weight_int8"] = nn.Parameter(w_int8, requires_grad=False)
                    self.params[f"{name}_scale"] = nn.Parameter(w_scale, requires_grad=False)
                    self.layer_meta[name]["type"] = "quant"
                else:
                    self.params[f"{name}_weight"] = nn.Parameter(W)
                    self.layer_meta[name]["type"] = "standard"

            if bias_data is not None:
                self.params[f"{name}_bias"] = nn.Parameter(bias_data.data)

    def _apply_proj(self, x, name):
        """Dispatches computation based on type (Standard, Quant, SVD, etc)"""
        meta_type = self.layer_meta[name]["type"]
        use_systolic = self.layer_meta[name].get("systolic", False)
        bias = self.params.get(f"{name}_bias", 0)

        if meta_type == "standard":
            return self._linear(x, self.params[f"{name}_weight"], systolic=use_systolic) + bias

        elif meta_type == "quant":
            w = self.params[f"{name}_weight_int8"].float() * self.params[f"{name}_scale"]
            return self._linear(x, w, systolic=use_systolic) + bias

        elif meta_type == "svd":
            U, S, Vh = self.params[f"{name}_U"], self.params[f"{name}_S"], self.params[f"{name}_Vh"]
            # x: [..., in] -> reshape to 2D for matmul control
            x2 = x.reshape(-1, x.shape[-1])  # [BS, in]
            t1 = self._matmul2d(x2, Vh.t().contiguous(), systolic=use_systolic)  # [BS, r]
            t1 = t1 * S  # broadcast over last dim
            t2 = self._matmul2d(t1, U.t().contiguous(), systolic=use_systolic)   # [BS, out]
            out = t2.reshape(*x.shape[:-1], U.shape[0])
            return out + bias

        elif meta_type == "svd_quant":
            U = self.params[f"{name}_U_int8"].float() * self.params[f"{name}_U_scale"]
            Vh = self.params[f"{name}_Vh_int8"].float() * self.params[f"{name}_Vh_scale"]
            S = self.params[f"{name}_S"]
            x2 = x.reshape(-1, x.shape[-1])  # [BS, in]
            t1 = self._matmul2d(x2, Vh.t().contiguous(), systolic=use_systolic)  # [BS, r]
            t1 = t1 * S
            t2 = self._matmul2d(t1, U.t().contiguous(), systolic=use_systolic)  # [BS, out]
            out = t2.reshape(*x.shape[:-1], U.shape[0])
            return out + bias

        else:
            raise ValueError(f"Unknown meta_type: {meta_type}")

    def forward(self, u):
        batch, seq_len, _ = u.shape

        # 1. Input Projection
        xz = self._apply_proj(u, "in_proj")
        x, z = xz.chunk(2, dim=-1)

        # 2. Convolution
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (D_CONV - 1, 0))
        x_conv = F.conv1d(x_pad, self.params["conv_weight"], bias=self.params.get("conv_bias"), groups=D_INNER)
        x_conv = F.silu(x_conv).transpose(1, 2)

        # 3. SSM Data Dependent Steps (SPLIT VERSION)
        # Instead of one big projection, we do 3 small ones
        dt_rank = self._apply_proj(x_conv, "x_proj_dt")
        B = self._apply_proj(x_conv, "x_proj_b")
        C = self._apply_proj(x_conv, "x_proj_c")

        dt = F.softplus(self._apply_proj(dt_rank, "dt_proj"))

        A = -torch.exp(self.params["A_log"].float())
        D = self.params["D"].float()
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)

        # 4. Recurrence
        h = torch.zeros(batch, D_INNER, D_STATE, device=x.device)
        y_ssm = []
        for t in range(seq_len):
            h = h * dA[:, t] + x_conv[:, t].unsqueeze(-1) * dB[:, t]
            y_t = torch.sum(h * C[:, t].unsqueeze(-2), dim=-1)
            y_t = y_t + (D * x_conv[:, t])
            y_ssm.append(y_t)

        y = torch.stack(y_ssm, dim=1) * F.silu(z)

        # 5. Output Projection
        return self._apply_proj(y, "out_proj")

class MambaModelUnified(nn.Module):
    def __init__(self, hf_model, config, skip_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_layers = skip_layers
        self.embedding = hf_model.backbone.embeddings
        self.final_norm = hf_model.backbone.norm_f
        self.lm_head = hf_model.lm_head

        for i, hf_layer in enumerate(hf_model.backbone.layers):
            layer_cfg = config if i not in skip_layers else {}
            block = UnifiedMambaBlock(i, layer_cfg)
            block.load_weights(hf_layer)
            self.layers.append(block)
            self.norms.append(hf_layer.norm)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(rms_norm(x, norm.weight))
        x = rms_norm(x, self.final_norm.weight)
        return self.lm_head(x)

if __name__ == "__main__":
    from datasets import load_dataset
    import tqdm

    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading Wikitext-2 dataset...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

    # --- UPDATED CONFIGURATION (Your requested config) ---
    mix_config = {
      "in_proj":   {"svd_rank": None, "quant": False, "quant_method": "average", "systolic": True},
      "out_proj":  {"svd_rank": None, "quant": False, "quant_method": "average", "systolic": True},
      "dt_proj":   {"svd_rank": None, "quant": False, "quant_method": "average", "systolic": True},

      "x_proj_dt": {"svd_rank": None, "quant": False, "quant_method": "average", "systolic": True},
      "x_proj_b":  {"svd_rank": None, "quant": True,  "quant_method": "abs", "systolic": True},
      "x_proj_c":  {"svd_rank": 10,   "quant": False, "quant_method": "average", "systolic": True},
    }

    custom_model = MambaModelUnified(hf_model, mix_config, [0, 1, 22, 23]).to(DEVICE)

    def eval_model(model, name):
        nlls = []
        prev_end_loc = 0
        count = 0
        stride = 512
        seq_len = 1024
        max_len = encodings.input_ids.size(1)

        for begin_loc in tqdm.tqdm(range(0, max_len, stride), desc=f"Evaluating {name}"):
            if count > 20: break  # Short run for speed

            end_loc = min(begin_loc + seq_len, max_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                nlls.append(loss)

            prev_end_loc = end_loc
            count += 1
            if end_loc == max_len: break

        return torch.exp(torch.stack(nlls).mean())

    print("\nRunning Valid Comparison...")
    ppl_hf = eval_model(hf_model, "HF Original")
    ppl_custom = eval_model(custom_model, "Custom Split-Quant + Systolic")

    print(f"\nHF: {ppl_hf:.4f} | Custom: {ppl_custom:.4f}")
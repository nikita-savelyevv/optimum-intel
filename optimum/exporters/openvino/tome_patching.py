import math
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def bipartite_soft_matching_random2d(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - r: number of tokens to remove (by merging)
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_add(-2, dst_idx.expand(B, r, C), src)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_tokens = tome_info["size"]
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        r = int(x.shape[1] * args["ratio"])

        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        # use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = bipartite_soft_matching_random2d(x, r)
    else:
        m, u = (do_nothing, do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_m, u_a, u_m  # Okay this is probably not very good


def merge_wavg(merge: Callable, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    # print("Merge")
    size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x


def make_mllama_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a mllama model.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = None,
        ) -> torch.Tensor:
            m_a, m_m, u_a, u_m = compute_merge(hidden_state, self._tome_info)
            # Self Attention
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)

            hidden_state = merge_wavg(m_a, hidden_state)

            hidden_state, attn_weights = self.self_attn(hidden_state)  # , attention_mask=attention_mask)
            if self.is_gated:
                hidden_state = self.gate_attn.tanh() * hidden_state
            hidden_state = residual + u_a(hidden_state)

            # Feed forward
            residual = hidden_state
            hidden_state = self.post_attention_layernorm(hidden_state)
            hidden_state = merge_wavg(m_m, hidden_state)
            hidden_state = self.mlp(hidden_state)
            if self.is_gated:
                hidden_state = self.gate_ffn.tanh() * hidden_state

            hidden_state = residual + u_m(hidden_state)

            outputs = (hidden_state,)

            if output_attentions:
                outputs += (attn_weights,)

            return outputs

    return ToMeBlock


def make_clip_encoder_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.FloatTensor]:
            hidden_state = hidden_states

            m_a, m_m, u_a, u_m = compute_merge(hidden_state, self._tome_info)

            # Self Attention
            residual = hidden_state
            hidden_state = self.layer_norm1(hidden_state)

            hidden_state, _ = merge_wavg(m_a, hidden_state)

            hidden_state, attn_weights = self.self_attn(
                hidden_states=hidden_state,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_state = residual + u_a(hidden_state)

            # Feed forward
            residual = hidden_state
            hidden_state = self.layer_norm2(hidden_state)
            hidden_state, _ = merge_wavg(m_m, hidden_state)
            hidden_state = self.mlp(hidden_state)
            hidden_state = residual + u_m(hidden_state)

            outputs = (hidden_state,)

            if output_attentions:
                outputs += (attn_weights,)

            return outputs

    return ToMeBlock


def mllama_hook_tome_model(model: torch.nn.Module):
    """Adds a forward pre hook to get the image size. This hook can be removed with remove_patch."""

    def hook(module, args):
        module._tome_info["size"] = args[0].shape[1]
        print("shape", args[0].shape)
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def llava_hook_tome_model(model: torch.nn.Module):
    """Adds a forward pre hook to get the image size. This hook can be removed with remove_patch."""

    def hook(module, args, kwargs):
        module._tome_info["size"] = kwargs["inputs_embeds"].shape[1]
        print("shape", kwargs["inputs_embeds"].shape)
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook, with_kwargs=True))


def remove_patch(model: torch.nn.Module):
    """Removes a patch from a ToMe Diffusion module if it was already patched."""
    # For mllama
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent

    return model


def patch_mmlama_vit(
    model: torch.nn.Module,
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_mlp: bool = False,
):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_mllama = isinstance_str(model, "MllamaVisionModel")

    if not is_mllama:
        raise RuntimeError("Provided model was not a Mllama model.")
    else:
        # Supports "pipe.unet" and "unet"
        mmlama_vit1 = model.transformer
        mmlama_vit2 = model.global_transformer

    mmlama_vit1._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp,
        },
    }
    mllama_hook_tome_model(mmlama_vit1)

    for _, module in mmlama_vit1.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "MllamaVisionEncoderLayer"):
            print("Patched module in transformer")
            make_tome_block_fn = make_mllama_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = mmlama_vit1._tome_info

    mmlama_vit2._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp,
        },
    }
    mllama_hook_tome_model(mmlama_vit2)

    for _, module in mmlama_vit2.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "MllamaVisionEncoderLayer"):
            print("Patched module in global_transformer")
            make_tome_block_fn = make_mllama_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = mmlama_vit2._tome_info


def patch_llava_vit(
    model: torch.nn.Module,
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_mlp: bool = False,
):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_llava = isinstance_str(model, "LlavaNextForConditionalGeneration") or isinstance_str(
        model, "LlavaForConditionalGeneration"
    )

    if not is_llava:
        raise RuntimeError("Provided model was not a Llava model.")
    else:
        # Supports "pipe.unet" and "unet"
        llava_vit = model.vision_tower.vision_model.encoder

    llava_vit._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp,
        },
    }
    llava_hook_tome_model(llava_vit)

    for _, module in llava_vit.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "CLIPEncoderLayer"):
            print("Patched module in transformer")
            make_tome_block_fn = make_clip_encoder_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = llava_vit._tome_info

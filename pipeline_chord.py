from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import BaseOutput
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel
import torch.nn.functional as F
DEFAULT_SEED = 42
DEFAULT_COMPUTE_DTYPE = torch.float32
DEFAULT_SAFETY_CHECKER_ID = "CompVis/stable-diffusion-safety-checker"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------


@dataclass
class ChordEditPipelineOutput(BaseOutput):
    images: List[Image.Image] | torch.Tensor
    latents: torch.Tensor
    masks: Optional[torch.Tensor] = None  # ← 新增，仅 return_mask=True 时非 None



class _CenterSquareCropTransform:
    """Center-crop the shorter image dimension before resizing."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        target = min(width, height)
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            resample = Image.LANCZOS
        return ImageOps.fit(
            image,
            (target, target),
            method=resample,
            centering=(0.5, 0.5),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# ChordEdit Pipeline
# ---------------------------------------------------------------------------


class ChordEditPipeline(DiffusionPipeline):
    """Standalone pipeline that wires up diffusers modules with the Chord editor."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        vae: AutoencoderKL,
        tokenizer,
        text_encoder: CLIPTextModel,
        default_edit_config: Optional[Dict[str, Any]] = None,
        image_size: int = 512,
        device: Optional[str | torch.device] = None,
        compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
        use_attention_mask: bool = False,
        use_center_crop: bool = True,
        use_safety_checker: bool = False,
        safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._compute_dtype = compute_dtype
        self._use_attention_mask = bool(use_attention_mask)
        self.to(self._device)
        self._set_compute_precision()

        self.default_edit_config = default_edit_config or {}
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)
        self._vae_transform = self._build_vae_transform()
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self._max_unet_timestep = self.scheduler.config.num_train_timesteps - 1
        self._use_safety_checker = bool(use_safety_checker)
        self._safety_checker_id = safety_checker_id
        self._safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self._safety_feature_extractor: Optional[CLIPImageProcessor] = None
        if self._use_safety_checker:
            self._init_safety_checker()

    def _set_compute_precision(self) -> None:
        modules = (self.unet, self.vae, self.text_encoder)
        for module in modules:
            if module is not None:
                module.to(device=self._device, dtype=self._compute_dtype)
    def _init_safety_checker(self) -> None:
        if not self._safety_checker_id:
            LOGGER.warning("Safety checker requested but no identifier provided; disabling safety checks.")
            self._use_safety_checker = False
            return
        try:
            self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                self._safety_checker_id,
                torch_dtype=self._compute_dtype,
            ).to(self._device)
            self._safety_feature_extractor = CLIPImageProcessor.from_pretrained(self._safety_checker_id)
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("Failed to initialize safety checker (%s). Safety checks disabled.", exc)
            self._safety_checker = None
            self._safety_feature_extractor = None
            self._use_safety_checker = False

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_local_weights(
        cls,
        component_paths: Dict[str, str],
        *,
        default_edit_config: Optional[Dict[str, Any]] = None,
        device: Optional[str | torch.device] = None,
        torch_dtype: torch.dtype = torch.float32,
        image_size: int = 512,
        use_center_crop: bool = True,
        compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
        use_attention_mask: bool = False,
        use_safety_checker: bool = False,
        safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> "ChordEditPipeline":
        """Instantiate the pipeline from individual component checkpoints."""

        unet = UNet2DConditionModel.from_pretrained(
            component_paths["unet_path"],
            torch_dtype=torch_dtype,
        )
        scheduler = DDPMScheduler.from_pretrained(component_paths["scheduler_path"])
        vae = AutoencoderKL.from_pretrained(component_paths["vae_path"], torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(component_paths["tokenizer_path"])
        text_encoder = CLIPTextModel.from_pretrained(
            component_paths["text_encoder_path"],
            torch_dtype=torch_dtype,
        )
        return cls(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            default_edit_config=default_edit_config,
            image_size=image_size,
            device=device,
            compute_dtype=compute_dtype,
            use_attention_mask=use_attention_mask,
            use_center_crop=use_center_crop,
            use_safety_checker=use_safety_checker,
            safety_checker_id=safety_checker_id,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image | torch.Tensor,
        *,
        source_prompt: str,
        target_prompt: str,
        edit_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_type: str = "pil",
        return_mask: bool = False,
    ) -> ChordEditPipelineOutput:
        """Run ChordEdit once on a single image."""

        cfg = dict(self.default_edit_config)
        if edit_config:
            cfg.update(edit_config)
        required_keys = ["noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"edit_config is missing required keys: {missing}")

        pixel_values = self._prepare_image_tensor(image)
        latents = self._encode_image_to_latent(pixel_values)
        src_embed = self.encode_prompt([source_prompt])
        tgt_embed = self.encode_prompt([target_prompt])

        output_latents: List[torch.Tensor] = []
        decoded_batches: List[torch.Tensor] = []

        edit_params = self._prepare_edit_params(cfg)
        seed_value = int(seed) if seed is not None else DEFAULT_SEED

        noise_list = self._prepare_noise_list(
            latents=latents,
            seed_value=seed_value,
            num_noises=edit_params["noise_samples"],
        )

        x0_pred, final_mask = self._run_edit(
            x_src=latents,
            src_embed=src_embed,
            edit_embed=tgt_embed,
            noise=noise_list,
            params=edit_params,
        )

        decoded = self._decode_latent_to_image(x0_pred)
        decoded, _ = self._apply_safety_checker(decoded)
        output_latents.append(x0_pred.detach().cpu())
        decoded_batches.append(decoded.detach().cpu())

        images_tensor = torch.cat(decoded_batches, dim=0)
        latents_tensor = torch.cat(output_latents, dim=0)
        images = self._tensor_to_pil(images_tensor) if output_type == "pil" else images_tensor

        return ChordEditPipelineOutput(
            images=images,
            latents=latents_tensor,
            masks=final_mask.detach().cpu() if (return_mask and final_mask is not None) else None,
        )

    def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
        """Public helper mirroring diffusers pipelines for text encoding."""
        return self._encode_text(prompts)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_image_tensor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            vae_tensor = self._vae_transform(image)
        elif torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = tensor * 2.0 - 1.0
            vae_tensor = tensor
        else:
            raise TypeError("image must be a PIL.Image or a torch.Tensor.")

        if vae_tensor.ndim == 3:
            vae_tensor = vae_tensor.unsqueeze(0)
        if self._use_center_crop and vae_tensor.ndim == 4:
            _, _, height, width = vae_tensor.shape
            if height != width:
                side = min(height, width)
                top = (height - side) // 2
                left = (width - side) // 2
                vae_tensor = vae_tensor[:, :, top : top + side, left : left + side]
        return vae_tensor.to(device=self._device, dtype=self._compute_dtype)

    def _encode_image_to_latent(self, pixel_values: torch.Tensor) -> torch.Tensor:
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        pixel_values = pixel_values.to(device=self._device, dtype=self._compute_dtype)
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        latents = latents * scaling_factor
        return latents.to(device=self._device, dtype=self._compute_dtype)

    def _decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents.to(device=self._device, dtype=self._compute_dtype)
        decoded = self.vae.decode(latents / scaling_factor).sample
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
        return decoded.to(dtype=self._compute_dtype)

    def _apply_safety_checker(self, images: torch.Tensor) -> tuple[torch.Tensor, List[bool]]:
        batch = images.shape[0]
        if (
            not self._use_safety_checker
            or self._safety_checker is None
            or self._safety_feature_extractor is None
            or batch == 0
        ):
            return images, [False] * batch

        images_clamped = images.detach().clamp(0.0, 1.0)
        pil_images = self._tensor_to_pil(images_clamped)
        try:
            clip_input = self._safety_feature_extractor(images=pil_images, return_tensors="pt").to(self._device)
            images_np = np.stack([np.array(img).astype(np.float32) / 255.0 for img in pil_images], axis=0)
            images_np = images_np * 2.0 - 1.0
            _, has_nsfw_concept = self._safety_checker(
                images=images_np,
                clip_input=clip_input.pixel_values.to(self._device),
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Safety checker failed (%s). Skipping safety checks.", exc)
            return images, [False] * batch

        if isinstance(has_nsfw_concept, torch.Tensor):
            has_nsfw = has_nsfw_concept.detach().cpu().to(dtype=torch.bool).tolist()
        else:
            has_nsfw = [bool(flag) for flag in has_nsfw_concept]

        if any(has_nsfw):
            for idx, flagged in enumerate(has_nsfw):
                if flagged:
                    images[idx] = torch.zeros_like(images[idx])
        return images, has_nsfw

    def _encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._device)
        attn_mask = inputs.attention_mask.to(self._device) if self._use_attention_mask else None
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
        return hidden.to(device=self._device, dtype=self._compute_dtype)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        to_pil = transforms.ToPILImage()
        return [to_pil(sample) for sample in tensor]

    def _build_vae_transform(self) -> transforms.Compose:
        """Create image->latent preprocessing transform."""
        ops: List[Any] = []
        if self._use_center_crop:
            ops.append(_CenterSquareCropTransform())
            resize_interp = InterpolationMode.LANCZOS
        else:
            resize_interp = InterpolationMode.BILINEAR
        ops.append(
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=resize_interp,
            )
        )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return transforms.Compose(ops)

    def _prepare_edit_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(cfg)
        params["noise_samples"] = int(max(1, params["noise_samples"]))
        params["n_steps"] = int(max(1, params["n_steps"]))
        params["t_start"] = float(max(0.0, min(1.0, params["t_start"])))
        params["t_end"] = float(max(0.0, min(params["t_start"], params["t_end"])))
        # ---- 时间自适应（能量驱动 δ）----
        params["use_adaptive_delta"] = bool(params.get("use_adaptive_delta", False))
        params["adaptive_alpha"] = float(
            max(0.0, min(1.0, float(params.get("adaptive_alpha", 1.0))))
        )
        params["adaptive_floor"] = float(
            max(0.0, min(1.0, float(params.get("adaptive_floor", 0.2))))
        )
        t_delta = float(max(0.0, min(1.0, params["t_delta"])))
        if t_delta >= params["t_start"]:
            safe_max = max(1, self._max_unet_timestep)
            t_delta = max(0.0, params["t_start"] - 1.0 / safe_max)
        params["t_delta"] = t_delta
        params["step_scale"] = float(params["step_scale"])
        params["cleanup"] = bool(params.get("cleanup", False))
        # ---- 新增：局部 Chord 参数 ----
        params["use_local_chord"] = bool(params.get("use_local_chord", False))
        mode = str(params.get("local_mode", "B")).upper()
        params["local_mode"] = mode if mode in ("A", "B") else "B"
        params["local_beta"] = float(
            max(0.0, min(1.0, float(params.get("local_beta", 0.2))))
        )
        tau_lo = float(max(0.0, float(params.get("mask_tau_low", 0.1))))
        tau_hi = float(max(tau_lo + 1e-3, float(params.get("mask_tau_high", 0.4))))
        params["mask_tau_low"] = tau_lo
        params["mask_tau_high"] = tau_hi
        params["mask_sigma"] = float(max(0.0, float(params.get("mask_sigma", 2.0))))

        # ---- 新增：高频增强/清晰度模块 ----
        params["use_sharpen_u"] = bool(params.get("use_sharpen_u", False))
        params["sharpen_gamma"] = float(max(0.0, float(params.get("sharpen_gamma", 0.5))))

        params["use_latent_sharp"] = bool(params.get("use_latent_sharp", False))
        params["latent_sharp_bg"] = float(max(0.0, float(params.get("latent_sharp_bg", 0.5))))
        params["latent_sharp_fg"] = float(max(0.0, float(params.get("latent_sharp_fg", 0.2))))
        # -------------------------------------

        return params

    def _prepare_noise_list(
        self,
        latents: torch.Tensor,
        seed_value: int,
        num_noises: int,
    ) -> List[torch.Tensor]:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        noise_list = [
            torch.randn_like(latents, device=latents.device, dtype=self._compute_dtype)
            for _ in range(num_noises)
        ]
        return noise_list

    def _gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """Separable-free 2D Gaussian blur on a (B, 1, H, W) mask."""
        if sigma <= 0:
            return mask
        import torch.nn.functional as F

        radius = max(1, int(round(2.5 * sigma)))  # 覆盖 ±2.5σ
        kernel_size = 2 * radius + 1
        coords = torch.arange(kernel_size, device=mask.device, dtype=torch.float32) - radius
        g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g = g / g.sum()
        kernel2d = (g[:, None] * g[None, :]).to(dtype=mask.dtype)
        channels = mask.shape[1]
        kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        return F.conv2d(mask, kernel2d, padding=radius, groups=channels)

    def _energy_mask(
            self,
            u_hat: torch.Tensor,
            tau_low: float = 0.1,
            tau_high: float = 0.4,
            sigma: float = 2.0,
            gamma: float = 0.5,  # ← 新增：mask 增益（<1 抬高 M）
            dilate: int = 2,  # ← 新增：最大池化膨胀核半径
            quantile: float = 0.95,  # ← 新增：归一化分位（0.99 → 0.95）
    ) -> torch.Tensor:
        """Derive a soft spatial mask from the control-field energy map.

        u_hat: (B, C, H, W) — 单步编辑方向
        返回:  (B, 1, H, W) ∈ [0, 1]
        """
        B = u_hat.shape[0]
        energy = (u_hat ** 2).sum(dim=1, keepdim=True)  # (B,1,H,W)
        flat = energy.reshape(B, -1).float()  # quantile 需 float32
        if flat.shape[1] >= 2:
            scale = torch.quantile(flat, 0.99, dim=1)
        else:
            scale = flat.max(dim=1).values
        scale = scale.clamp_min(1e-8).view(B, 1, 1, 1).to(dtype=u_hat.dtype)
        e_norm = energy / scale
        denom = max(tau_high - tau_low, 1e-6)
        mask = torch.clamp((e_norm - tau_low) / denom, 0.0, 1.0)
        # ① 膨胀：把强编辑点的邻域也纳入（M>0 的区域向外扩 dilate 像素）
        if dilate > 0:
            mask = F.max_pool2d(mask, kernel_size=2 * dilate + 1, stride=1, padding=dilate)

        # ② gamma 增益：M ← M^γ，γ<1 时中低值被显著抬高
        if gamma != 1.0:
            mask = mask.clamp_min(1e-8) ** gamma

        # ③ 高斯羽化（接缝 + 理论需要）
        mask = self._gaussian_blur(mask, sigma=sigma)
        return mask


    def _time_to_index(self, batch: int, t_scalar: float, device, dtype=torch.long):
        idx = round(self._max_unet_timestep * float(t_scalar))
        idx = max(0, min(self._max_unet_timestep, idx))
        return torch.full((batch,), idx, device=device, dtype=dtype)

    def _get_alpha_sigma(self, tensor: torch.Tensor, timesteps: torch.Tensor):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=torch.float32, device=tensor.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(dtype=tensor.dtype, device=tensor.device)
        sigma_t = sigma_t.to(dtype=tensor.dtype, device=tensor.device)
        eps = torch.finfo(alpha_t.dtype).eps
        alpha_t = alpha_t.clamp(min=eps)
        return alpha_t, sigma_t

    def _pred_x0(self, x_anchor, timesteps, cond, noise):
        alpha_t, sigma_t = self._get_alpha_sigma(x_anchor, timesteps)
        z_t = alpha_t * x_anchor + sigma_t * noise
        noise_pred = self.unet(
            sample=z_t,
            timestep=timesteps,
            encoder_hidden_states=cond,
            return_dict=False,
        )[0]
        x0_pred = (z_t - sigma_t * noise_pred) / alpha_t
        return x0_pred

    def _u_estimate(self, x_anchor, src_embed, edit_embed, noise, t_s: float, delta: float,adaptive_cfg ):
        batch, device = x_anchor.shape[0], x_anchor.device
        t_idx_s = self._time_to_index(batch, t_s, device=device)
        t_idx_s0 = self._time_to_index(batch, max(0.0, t_s - delta), device=device)

        noises = noise if isinstance(noise, (list, tuple)) else [noise]

        alpha_s, sigma_s = self._get_alpha_sigma(x_anchor, t_idx_s)
        alpha_prev, sigma_prev = self._get_alpha_sigma(x_anchor, t_idx_s0)

        num_noises = len(noises)
        noise_stack = torch.stack(noises, dim=0)

        x_anchor_b = x_anchor.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_s_b = alpha_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_prev_b = alpha_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_s_b = sigma_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_prev_b = sigma_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)

        z_s = alpha_s_b * x_anchor_b + sigma_s_b * noise_stack
        z_prev = alpha_prev_b * x_anchor_b + sigma_prev_b * noise_stack

        samples = torch.stack([z_s, z_s, z_prev, z_prev], dim=1)
        samples = samples.reshape(num_noises * 4 * batch, *x_anchor.shape[1:])

        conds = torch.cat([src_embed, edit_embed, src_embed, edit_embed], dim=0)
        repeat_dims = [num_noises] + [1] * (conds.dim() - 1)
        conds = conds.repeat(*repeat_dims)

        timesteps = torch.cat([t_idx_s, t_idx_s, t_idx_s0, t_idx_s0], dim=0)
        timesteps = timesteps.repeat(num_noises)

        alpha_cat = torch.stack(
            [alpha_s_b, alpha_s_b, alpha_prev_b, alpha_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)
        sigma_cat = torch.stack(
            [sigma_s_b, sigma_s_b, sigma_prev_b, sigma_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)

        noise_pred = self.unet(
            sample=samples,
            timestep=timesteps,
            encoder_hidden_states=conds,
            return_dict=False,
        )[0]

        x0_all = (samples - sigma_cat * noise_pred) / alpha_cat
        x0_all = x0_all.reshape(num_noises, 4, batch, *x_anchor.shape[1:])
        x_src_p_s, x_tar_p_s, x_src_p_s0, x_tar_p_s0 = x0_all.unbind(dim=1)

        dv_s = (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises)
        dv_s0 = (x_tar_p_s0 - x_src_p_s0).sum(dim=0) / float(num_noises)

        if adaptive_cfg is None or not adaptive_cfg.get("enable", False):
            denom = (t_s + delta)
            if denom <= 1e-6:
                return dv_s
            return (delta * dv_s + t_s * dv_s0) / denom

        # ---- 时间自适应：逐位置有效 δ ----
        alpha = float(adaptive_cfg.get("alpha", 1.0))
        floor = float(adaptive_cfg.get("floor", 0.2))
        eps = torch.finfo(dv_s.dtype).eps

        # 复用 _energy_mask 的归一化：tau_low=0, tau_high=1, sigma=0 时
        # 返回 clip(E/E_99, 0, 1)，正是所需的 E_norm ∈ [0,1]
        E_norm = self._energy_mask(
            dv_s0, tau_low=0.0, tau_high=1.0, sigma=0.0,
            gamma=1.0,    # <--- 1.0，取消增益
            dilate=0,     # <--- 0.0，取消膨胀
        )  # (B,1,H,W)

        E_mixed = floor + (1.0 - floor) * E_norm  # (B,1,H,W) ∈ [floor, 1]
        delta_eff = delta * ((1.0 - alpha) + alpha * E_mixed)  # (B,1,H,W) ∈ [δ·(1-α), δ]

        # 防御：确保 δ_eff < t_s（理论上已满足，兜底防浮点误差）
        max_delta = max(t_s - 1e-6, 0.0)
        delta_eff = delta_eff.clamp(min=0.0, max=max_delta)

        denom = (t_s + delta_eff).clamp_min(eps)  # (B,1,H,W)
        return (delta_eff * dv_s + t_s * dv_s0) / denom

    def _run_edit(
            self,
            x_src: torch.Tensor,
            src_embed: torch.Tensor,
            edit_embed: torch.Tensor,
            noise: List[torch.Tensor],
            params: Dict[str, Any],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = x_src.device
        if params["n_steps"] == 1:
            t_grid = [params["t_start"]]
        else:
            t_grid = torch.linspace(
                params["t_start"], params["t_end"],
                steps=params["n_steps"], device=device,
            ).tolist()

        use_local = params["use_local_chord"]
        mode = params["local_mode"]
        beta = params["local_beta"]
        tau_lo = params["mask_tau_low"]
        tau_hi = params["mask_tau_high"]
        sigma = params["mask_sigma"]
        # 新增背景清晰参数解包
        use_sharpen_u = params.get("use_sharpen_u", False)
        sharpen_gamma = params.get("sharpen_gamma", 0.5)

        x_curr = x_src
        last_mask: Optional[torch.Tensor] = None
        adaptive_cfg = None
        if params["use_adaptive_delta"]:
            adaptive_cfg = {
                "enable": True,
                "alpha": params["adaptive_alpha"],
                "floor": params["adaptive_floor"],
            }
        adaptive_cfg = None
        if params["use_adaptive_delta"]:
            adaptive_cfg = {
                "enable": True,
                "alpha": params["adaptive_alpha"],
                "floor": params["adaptive_floor"],
            }
        for t_s in t_grid:
            u_hat = self._u_estimate(
                x_curr, src_embed, edit_embed, noise,
                float(t_s), params["t_delta"],adaptive_cfg=adaptive_cfg,

            )
            # 路线一：控制场高频锐化 (Unsharp Masking)
            if use_sharpen_u:
                # 提取 u_hat 的高频（u_hat - 模糊）
                # 注意：_gaussian_blur 支持多通道输入
                hf_u = u_hat - self._gaussian_blur(u_hat, sigma=1.0)
                u_hat = u_hat + sharpen_gamma * hf_u

            if use_local:
                mask = self._energy_mask(u_hat, tau_lo, tau_hi, sigma)  # (B,1,H,W)
                last_mask = mask
                if params.get("local_apply_transport", True):
                    u_local = mask * u_hat + (1.0 - mask) * beta * u_hat
                    x_curr = x_curr + params["step_scale"] * u_local
                else:
                    # 原版 transport，力度完全保留
                    x_curr = x_curr + params["step_scale"] * u_hat
            else:
                x_curr = x_curr + params["step_scale"] * u_hat


        if params["cleanup"]:
            t_end_idx = self._time_to_index(
                x_src.shape[0], params["t_end"], device=device
            )
            if use_local and last_mask is not None:
                if mode == "B":
                    # 背景锚定 → prox → 后混合
                    x_pred_reg = last_mask * x_curr + (1.0 - last_mask) * x_src
                    x_prox = self._pred_x0(x_pred_reg, t_end_idx, edit_embed, noise[0])
                    x_curr = last_mask * x_prox + (1.0 - last_mask) * x_src
                else:  # mode == "A"
                    x_prox = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0])
                    x_curr = last_mask * x_prox + (1.0 - last_mask) * x_src
            else:
                x_curr = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0])

        # 路线二：潜空间高频回填 (Latent HF Compensation)
        if params.get("use_latent_sharp", False):
            hf_curr = x_curr - self._gaussian_blur(x_curr, sigma=1.2)
            hf_src = x_src - self._gaussian_blur(x_src, sigma=1.2)

            if use_local and last_mask is not None:
                # 背景把源图的高频补回来，前景做适当提锐
                x_curr = x_curr + (1.0 - last_mask) * hf_src * params["latent_sharp_bg"]
                x_curr = x_curr + last_mask * hf_curr * params["latent_sharp_fg"]
            else:
                # 如果关闭了局部 Mask，就只做整体提锐
                x_curr = x_curr + hf_curr * params["latent_sharp_fg"]

        return x_curr, last_mask

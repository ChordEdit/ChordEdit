from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image

from pipeline_chord import ChordEditPipeline
from utils import DEFAULT_DATA_ROOT

LOGGER = logging.getLogger("chord_app")

COMPONENT_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder",
    "tokenizer_path": "tokenizer",
    "vae_path": "vae",
}

DEFAULT_MODEL_ROOT = "/sd-turbo"
DEFAULT_COMPONENT_PATHS: Dict[str, str] = {
    key: str(Path(DEFAULT_MODEL_ROOT) / subdir) for key, subdir in COMPONENT_SUBDIRS.items()
}

DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
    "use_local_chord": False,
    "local_mode": "B",
    "local_beta": 0.2,
    "mask_tau_low": 0.1,
    "mask_tau_high": 0.4,
    "mask_sigma": 2.0,
    "use_adaptive_delta": False,
    "adaptive_alpha": 1.0,
    "adaptive_floor": 0.2,
    # ---- 清晰度增强模块 ----
    "use_sharpen_u": False,
    "sharpen_gamma": 0.5,
    "use_latent_sharp": False,
    "latent_sharp_bg": 0.5,
    "latent_sharp_fg": 0.2,
}

DEFAULT_SEED = 42
DEFAULT_PRECISION = "fp32"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_MAX_EXAMPLES = 24
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7860
DEFAULT_CENTER_CROP = True
DEFAULT_USE_ATTENTION_MASK = False
DEFAULT_USE_SAFETY_CHECKER = False
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SQUARE_PREVIEW_CSS = """
#source-image-input { width: 100% !important; }
#source-image-input .image-container,
#source-image-input [data-testid="image"] {
    aspect-ratio: 1 / 1 !important;
    overflow: hidden !important;
    background: #00000008 !important;
}
#source-image-input img, #source-image-input canvas {
    width: 100% !important; height: 100% !important;
    object-fit: cover !important; object-position: center center !important;
    display: block !important; background: transparent !important;
}
#editor-main-row { align-items: center !important; }
#source-prompt textarea, #target-prompt textarea {
    height: 112px !important; max-height: 112px !important;
    overflow-y: auto !important; resize: none !important;
}
.panel-note p {
    margin: 0 0 12px 0 !important; line-height: 1.4 !important;
    font-size: 0.95rem !important; color: #666 !important;
}
.param-guide table {
    font-size: 0.88rem !important; line-height: 1.35 !important;
}
.param-guide code {
    background: #f2f2f2 !important; padding: 0 3px !important; border-radius: 3px !important;
}
"""


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _mask_tensor_to_pil(mask_tensor: torch.Tensor, size: int = 512) -> Image.Image:
    """Convert pipeline-returned mask (B,1,H,W) ∈ [0,1] into a grayscale PIL image."""
    m = mask_tensor.detach().float().cpu()
    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        m = m[0]
    m = m.clamp(0.0, 1.0)
    arr = (m.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((size, size), Image.BILINEAR)


def _overlay_mask_on_source(
    source: Optional[Image.Image],
    mask_pil: Image.Image,
    alpha: float = 0.55,
) -> Optional[Image.Image]:
    """Red-tint the source image by the mask; center-crops like the pipeline."""
    if source is None:
        return None
    size = mask_pil.size[0]
    w, h = source.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = source.crop((left, top, left + side, top + side))
    src = cropped.convert("RGB").resize((size, size), Image.BILINEAR)
    src_arr = np.asarray(src).astype(np.float32)
    mask_arr = np.asarray(mask_pil).astype(np.float32) / 255.0
    red = np.array([255.0, 60.0, 60.0], dtype=np.float32)
    wgt = (mask_arr * alpha)[..., None]
    out = src_arr * (1.0 - wgt) + red * wgt
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Original helpers (unchanged)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ChordEdit web app.")
    parser.add_argument(
        "--model-root", type=str, default=DEFAULT_MODEL_ROOT,
        help="Root folder containing unet/scheduler/text_encoder/tokenizer/vae subfolders.",
    )
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT, help="Web server port.")
    return parser.parse_args()


def _dtype_from_precision(value: Optional[str]) -> torch.dtype:
    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def _paths_from_model_root(model_root: str | Path) -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    return {key: str((root / subdir).resolve()) for key, subdir in COMPONENT_SUBDIRS.items()}


def _expand_paths(path_map: Dict[str, str | None]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    missing: List[str] = []
    for key in COMPONENT_SUBDIRS:
        value = path_map.get(key)
        final_value = value if value is not None else DEFAULT_COMPONENT_PATHS.get(key)
        if final_value is None:
            missing.append(key)
            continue
        expanded[key] = str(Path(final_value).expanduser().resolve())
    if missing:
        raise ValueError(
            f"Missing required component paths for: {missing}. "
            "Set --model-root or provide per-component paths."
        )
    return expanded


def _resolve_component_paths(model_root: str | Path) -> Dict[str, str]:
    return _expand_paths(_paths_from_model_root(model_root))


def _select_image_file(folder: Path) -> Path:
    candidates = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No RGB image found inside {folder}")
    preferred = sorted(
        (p for p in candidates if p.stem.lower() in {"i", "image", "original"}),
        key=lambda p: p.name,
    )
    if preferred:
        return preferred[0]
    return sorted(candidates, key=lambda p: p.name)[0]


def load_examples(dataset_root: Path, max_examples: Optional[int]) -> List[List[Any]]:
    examples: List[List[Any]] = []
    if not dataset_root.exists():
        LOGGER.warning("Example dataset does not exist: %s", dataset_root)
        return examples
    for subdir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        meta_file = subdir / "meta.jsonl"
        if not meta_file.exists():
            continue
        try:
            image_path = _select_image_file(subdir)
        except FileNotFoundError:
            LOGGER.warning("No image found in %s", subdir)
            continue
        with meta_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping invalid JSON in %s:%d (%s)", meta_file, line_number, exc)
                    continue
                src_prompt = str(record.get("original_prompt", "")).strip()
                tgt_prompt = str(record.get("edited_prompt", "")).strip()
                examples.append([str(image_path), src_prompt, tgt_prompt])
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
    return examples


def _validate_inputs(
    image: Optional[Image.Image], source_prompt: str, target_prompt: str,
    t_start: float, t_end: float, t_delta: float,
) -> None:
    if image is None:
        raise gr.Error("Please upload a source image first.")
    if not source_prompt or not source_prompt.strip():
        raise gr.Error("Please provide the source image prompt.")
    if not target_prompt or not target_prompt.strip():
        raise gr.Error("Please provide the target image prompt.")
    if t_start <= t_end:
        raise gr.Error("Invalid parameters: t_start must be greater than t_end.")
    if t_delta < 0:
        raise gr.Error("Invalid parameters: t_delta must be greater than or equal to 0.")


# ---------------------------------------------------------------------------
# Demo builder
# ---------------------------------------------------------------------------
def build_demo(
    pipeline: ChordEditPipeline,
    default_seed: int,
    default_edit_config: Dict[str, Any],
    examples: List[List[Any]],
) -> gr.Blocks:
    def run_edit(
        image: Optional[Image.Image],
        source_prompt: str, target_prompt: str,
        seed: float, n_samples: float,
        t_start: float, t_end: float, t_delta: float, step_scale: float,
        use_local_chord, local_beta, mask_tau_low, mask_tau_high, mask_sigma,
        use_adaptive_delta, adaptive_alpha, adaptive_floor,use_sharpen_u, sharpen_gamma, use_latent_sharp, latent_sharp_bg, latent_sharp_fg,
    ):
        _validate_inputs(image, source_prompt, target_prompt, t_start, t_end, t_delta)

        edit_config = {
            "noise_samples": int(n_samples),
            "n_steps": int(default_edit_config.get("n_steps", 1)),
            "t_start": float(t_start),
            "t_end": float(t_end),
            "t_delta": float(t_delta),
            "step_scale": float(step_scale),
            "cleanup": bool(default_edit_config.get("cleanup", True)),
            "use_local_chord": bool(use_local_chord),
            "local_mode": default_edit_config.get("local_mode", "B"),
            "local_beta": float(local_beta),
            "mask_tau_low": float(mask_tau_low),
            "mask_tau_high": float(mask_tau_high),
            "mask_sigma": float(mask_sigma),
            "use_adaptive_delta": bool(use_adaptive_delta),
            "adaptive_alpha": float(adaptive_alpha),
            "adaptive_floor": float(adaptive_floor),
            # ---- 清晰度增强模块 ----
            "use_sharpen_u": bool(use_sharpen_u),
            "sharpen_gamma": float(sharpen_gamma),
            "use_latent_sharp": bool(use_latent_sharp),
            "latent_sharp_bg": float(latent_sharp_bg),
            "latent_sharp_fg": float(latent_sharp_fg),
        }

        try:
            result = pipeline(
                image=image,
                source_prompt=source_prompt.strip(),
                target_prompt=target_prompt.strip(),
                edit_config=edit_config,
                seed=int(seed),
                return_mask=True,
            )
        except Exception as exc:
            LOGGER.exception("Editing failed.")
            raise gr.Error(f"Editing failed: {exc}") from exc

        images = result.images
        if not isinstance(images, list) or not images:
            raise gr.Error("The pipeline returned no output image.")

        mask_pil = None
        overlay_pil = None
        if getattr(result, "masks", None) is not None:
            try:
                mask_pil = _mask_tensor_to_pil(result.masks, size=512)
                overlay_pil = _overlay_mask_on_source(image, mask_pil)
            except Exception as exc:
                LOGGER.warning("Mask visualization failed: %s", exc)

        return images[0], mask_pil, overlay_pil

    with gr.Blocks(title="ChordEdit App", css=SQUARE_PREVIEW_CSS) as demo:
        gr.Markdown("# ChordEdit App")
        gr.Markdown(
            'To study artifacts and background leakage of the one-step editor without Chord Control, set `t_delta` to `0`.\n'
            'Images shown in the paper are available in the "Examples" list below.',
            elem_classes=["panel-note"],
        )

        # ---------------- 参数调整指南（对应论文图 17 / 图 8） ----------------
        with gr.Accordion("📖 参数调整指南", open=False, elem_classes=["param-guide"]):
            gr.Markdown(
                """
**全局参数（作用于原版 ChordEdit，关闭本地开关时同样生效）**

| 参数 | 往 **大** 调 | 往 **小** 调 | 推荐 | 论文依据 |
|---|---|---|---|---|
| `t_start` | 语义编辑更强，背景漂移更多 | 更保守，可能欠编辑 | 0.85–0.95 | 论文默认 0.90 |
| `t_end` | 语义更强，**背景 LPIPS↑**（prox over-edit） | 背景保留更好，但可能编辑不足 | 0.20–0.30 | 论文图 17 的 \(t_c\) trade-off |
| `t_delta` | 融合更多**噪声端**，更激进 | 更保守；`=0` 退化为单点估计（研究无 Chord 控制的伪影） | 0.10–0.20 | 论文默认 0.15 |
| `step_scale` (λ) | 编辑更强，可能过编辑/伪影 | 编辑不足 | 0.8–1.2 | 论文默认 1.0 |
| `n_samples` | 方向估计方差↓，**显存/耗时线性↑** | 快但方向更抖 | 1–4 | `N*4*B` 决定 UNet 前向 batch |

**空间自适应（`use_local_chord`）**

| 参数 | 往 **大** 调 | 往 **小** 调 | 推荐 |
|---|---|---|---|
| `local_beta` (β) | 趋近原版（`1.0` = 完全关闭） | 背景更冻结，**`0` 易在边界出现接缝/贴图感** | 0.1–0.3 |
| `mask_sigma` (σ) | 边界更平滑，**过大缩编辑区** | `0` = 硬 mask，接缝明显 | 1.5–2.5 |
| `mask_tau_low` | mask 收缩，可能**漏掉弱语义编辑区** | mask 扩大，背景也变 | 0.05–0.20 |
| `mask_tau_high` | **编辑区被削弱**（CLIP-Edited↓） | mask 更满，接近原版 | 0.30–0.60 |

**时间自适应（`use_adaptive_delta`）**

| 参数 | 往 **大** 调 | 往 **小** 调 | 推荐 |
|---|---|---|---|
| `adaptive_alpha` (α) | 能量调制更强、更激进；**`0` 精确等价原版** | 退化为固定 δ | 0.5–0.75 |
| `adaptive_floor` | 背景 δ 更大，更接近原版（更稳） | **`0` 时背景融合几乎全取 `dv_s0`，易抖动** | 0.20–0.40 |

> 提示：mask 面板为空说明当前 `use_local_chord=False`，属正常情况。

#### 🛠 清晰度增强模块（High-Frequency Enhancement）

| 参数 | 原理与效果 | 推荐范围 | 论文解释 |
|---|---|---|---|
| **`use_sharpen_u`** | **路线一：控制场锐化**。对提取出的编辑方向 `û` 做反锐化掩蔽。让编辑边界更清晰，不拖泥带水。 | 开启 | 在 Transport 阶段干预能量场，增加高频。 |
| `sharpen_gamma` | 锐化强度。↑ 边缘更锐，但太高会导致边缘出现光晕（Halos）。 | 0.3 – 0.8 | Unsharp masking 的增益系数。 |
| **`use_latent_sharp`** | **路线二：潜空间高频回填**。由于 VAE 解码必定丢失高频，该操作在 VAE 解码前，把原图的纹理高频“塞回”背景，同时给前景提锐。 | 开启 | 解决单步编辑和 VAE 重建导致的“发糊”问题。 |
| `latent_sharp_bg` | 背景清晰度回填力度。↑ 背景树叶/发丝等纹理越清晰（越接近原图）；↓ 背景可能会显得像被磨皮过一样。 | 0.3 – 0.7 | 用原图高频 `hf_src` 修补背景。 |
| `latent_sharp_fg` | 前景（编辑区）细节强化。↑ 胡子/衣服纹理更锐利；↓ 更柔和。**注意：太高容易导致噪点或过曝。** | 0.1 – 0.3 | 用生成图高频 `hf_curr` 提锐前景。 |

                """
            )

        with gr.Row(elem_id="editor-main-row"):
            with gr.Column(scale=5, elem_id="left-input-panel"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=280):
                            input_image = gr.Image(
                                type="pil", label="Source Image",
                                sources=["upload", "clipboard"],
                                elem_id="source-image-input", height=320,
                            )
                        with gr.Column(scale=1, min_width=280):
                            source_prompt = gr.Textbox(
                                label="Source Prompt", lines=4, max_lines=4,
                                placeholder="Example: A cat on a sofa", elem_id="source-prompt",
                            )
                            target_prompt = gr.Textbox(
                                label="Target Prompt", lines=4, max_lines=4,
                                placeholder="Example: A cat wearing sunglasses", elem_id="target-prompt",
                            )

                gr.Markdown("### Parameters")
                n_samples_default = int(
                    default_edit_config.get("n_samples", default_edit_config.get("noise_samples", 1))
                )
                with gr.Row():
                    seed_input = gr.Number(
                        label="Seed", value=int(default_seed), precision=0,
                        info="固定噪声；同 seed 结果可复现。数值本身无大小含义。",
                    )
                    n_samples_input = gr.Slider(
                        label="n_samples (方向平均噪声数)",
                        minimum=1, maximum=16, step=1, value=n_samples_default,
                        info="↑ 方向估计更稳定（方差↓），但显存/耗时线性增长（UNet batch = N×4×B）；↓ 更快但更抖。推荐 1–4。",
                    )
                    step_scale_input = gr.Slider(
                        label="step_scale (λ, transport 步长)",
                        minimum=0.1, maximum=5.0, step=0.1,
                        value=float(default_edit_config.get("step_scale", 1.0)),
                        info="↑ 编辑更强但可能过编辑/伪影；↓ 编辑不足。论文默认 1.0，推荐 0.8–1.2。",
                    )
                with gr.Row():
                    t_start_input = gr.Slider(
                        label="t_start (编辑起始时间比例)",
                        minimum=0.01, maximum=1.0, step=0.01,
                        value=float(default_edit_config.get("t_start", 0.90)),
                        info="↑ 语义更强、背景漂移更多；↓ 更保守、可能欠编辑。论文默认 0.90，推荐 0.85–0.95。",
                    )
                    t_end_input = gr.Slider(
                        label="t_end (cleanup 加噪时刻 / 网格终点)",
                        minimum=0.0, maximum=0.99, step=0.01,
                        value=float(default_edit_config.get("t_end", 0.30)),
                        info="↑ 语义更强但背景 LPIPS 上升（论文图 17 的 prox over-edit）；↓ 背景保留更好。论文默认 0.30，推荐 0.20–0.30。",
                    )
                    t_delta_input = gr.Slider(
                        label="t_delta (Chord 双点估计间隔 δ)",
                        minimum=0.0, maximum=0.5, step=0.01,
                        value=float(default_edit_config.get("t_delta", 0.15)),
                        info="=0 退化为单点估计（研究无 Chord Control 的伪影）；↑ 融合更多噪声端、更激进；↓ 更保守。论文默认 0.15，推荐 0.10–0.20。",
                    )

                use_local_chord = gr.Checkbox(
                    label="Use Local Chord (Energy-Mask Localization)",
                    value=False,
                    info="关闭 = 原版 ChordEdit 基线；开启 = 用控制场能量导出 soft mask：编辑区强传输、背景弱传输，prox 前后双重背景锚定。NFE 不变。",
                )
                with gr.Accordion("Local Chord Parameters", open=False):
                    with gr.Row():
                        local_beta_input = gr.Slider(
                            label="local_beta (β, 背景 transport 衰减)",
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=float(default_edit_config.get("local_beta", 0.2)),
                            info="1.0 = 退化为原版；0 = 背景完全冻结（可能边界接缝/贴图感）；推荐 0.1–0.3，保留轻微背景响应更自然。",
                        )
                        mask_sigma_input = gr.Slider(
                            label="mask_sigma (σ, 高斯羽化)",
                            minimum=0.0, maximum=4.0, step=0.1,
                            value=float(default_edit_config.get("mask_sigma", 2.0)),
                            info="0 = 硬 mask（有接缝）；2 ≈ 平滑过渡；>4 过度模糊会缩小编辑区。推荐 1.5–2.5。",
                        )
                    with gr.Row():
                        mask_tau_low_input = gr.Slider(
                            label="mask_tau_low (soft mask 下阈值)",
                            minimum=0.0, maximum=1.0, step=0.01,
                            value=float(default_edit_config.get("mask_tau_low", 0.1)),
                            info="↑ mask 收缩、只保留强编辑区域（可能漏掉弱语义）；↓ mask 扩大、背景也变。推荐 0.05–0.20。",
                        )
                        mask_tau_high_input = gr.Slider(
                            label="mask_tau_high (soft mask 上阈值)",
                            minimum=0.0, maximum=2.0, step=0.01,
                            value=float(default_edit_config.get("mask_tau_high", 0.4)),
                            info="↑ 编辑区可能也被削弱（CLIP-Edited↓）；↓ mask 更满、接近原版。推荐 0.30–0.60。",
                        )

                use_adaptive_delta = gr.Checkbox(
                    label="Use Adaptive Delta (Energy-Driven Time Bandwidth)",
                    value=False,
                    info="关闭 = 固定 δ；开启 = 用 dv_s0 逐位置能量调制 δ：编辑区信任噪声端（激进）、背景信任干净端（保守）。α=0 时精确退化为原版。",
                )
                with gr.Accordion("Adaptive Delta Parameters", open=False):
                    with gr.Row():
                        adaptive_alpha_input = gr.Slider(
                            label="adaptive_alpha (α, 能量调制强度)",
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=float(default_edit_config.get("adaptive_alpha", 1.0)),
                            info="0 = 精确退化为固定 δ（用于等价性验证）；1 = 完全能量驱动，较激进。推荐 0.5–0.75。",
                        )
                        adaptive_floor_input = gr.Slider(
                            label="adaptive_floor (背景最小 δ 比例)",
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=float(default_edit_config.get("adaptive_floor", 0.2)),
                            info="0 时背景 δ→0，融合几乎全取 dv_s0，而背景 dv_s0 是噪声 → 易抖动；推荐 0.20–0.40 保留最小噪声端信息。",
                        )
                # ---------------- 清晰度增强 ----------------
                use_sharpen_u = gr.Checkbox(
                    label="1. Sharpen Control Field (u_hat High-Freq)",
                    value=False,
                    info="锐化编辑方向场，修复边缘模糊（零 NFE）。",
                )
                with gr.Accordion("Control Field Sharpen Parameters", open=False):
                    sharpen_gamma_input = gr.Slider(
                        label="sharpen_gamma (边缘锐化强度)",
                        minimum=0.0, maximum=2.0, step=0.1,
                        value=float(default_edit_config.get("sharpen_gamma", 0.5)),
                        info="↑ 边缘更锐利但可能出现伪影；↓ 边缘较平滑。推荐 0.3–0.8。",
                    )

                use_latent_sharp = gr.Checkbox(
                    label="2. Latent HF Compensation (Background Clarity)",
                    value=False,
                    info="在解码前注入原图高频，修复背景/纹理发糊（零 NFE）。",
                )
                with gr.Accordion("Latent HF Compensation Parameters", open=False):
                    with gr.Row():
                        latent_sharp_bg_input = gr.Slider(
                            label="latent_sharp_bg (背景清晰度回填)",
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=float(default_edit_config.get("latent_sharp_bg", 0.5)),
                            info="↑ 背景越接近源图原生清晰度；↓ 背景更平滑（可能会糊）。推荐 0.3–0.7。",
                        )
                        latent_sharp_fg_input = gr.Slider(
                            label="latent_sharp_fg (前景细节强化)",
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=float(default_edit_config.get("latent_sharp_fg", 0.2)),
                            info="↑ 生成的编辑对象细节更锐利；↓ 更柔和。推荐 0.1–0.3，太高容易过曝。",
                        )
                run_button = gr.Button("Run Edit", variant="primary")

            # ---------------- 右栏：结果 + mask 可视化（Tabs） ----------------
            with gr.Column(scale=5, elem_id="right-output-panel"):
                with gr.Tabs():
                    with gr.Tab("Editing Result"):
                        output_image = gr.Image(
                            type="pil", label="Editing Result",
                            elem_id="result-image-output", height=440,
                        )
                    with gr.Tab("Localization Mask"):
                        gr.Markdown(
                            "**Energy-Derived Soft Mask**：颜色越白 = 编辑强度越高。"
                            "对应论文图 8 的能量图定位证据；仅在 `Use Local Chord` 开启时生成。",
                            elem_classes=["panel-note"],
                        )
                        mask_output = gr.Image(
                            type="pil", label="Soft Mask M ∈ [0,1]", height=250,
                        )
                        overlay_output = gr.Image(
                            type="pil", label="Mask Overlay on Source (red = 编辑区域)",
                            height=250,
                        )

        run_inputs = [
            input_image, source_prompt, target_prompt,
            seed_input, n_samples_input,
            t_start_input, t_end_input, t_delta_input, step_scale_input,
            use_local_chord, local_beta_input,
            mask_tau_low_input, mask_tau_high_input, mask_sigma_input,
            use_adaptive_delta, adaptive_alpha_input, adaptive_floor_input,
            use_sharpen_u, sharpen_gamma_input,
            use_latent_sharp, latent_sharp_bg_input, latent_sharp_fg_input,
        ]
        run_outputs = [output_image, mask_output, overlay_output]

        run_button.click(fn=run_edit, inputs=run_inputs, outputs=run_outputs)
        target_prompt.submit(fn=run_edit, inputs=run_inputs, outputs=run_outputs)

        # 联动：未开启时禁用对应参数组，避免误触
        use_local_chord.change(
            fn=lambda v: [gr.update(interactive=bool(v)) for _ in
                          [local_beta_input, mask_sigma_input, mask_tau_low_input, mask_tau_high_input]],
            inputs=use_local_chord,
            outputs=[local_beta_input, mask_sigma_input, mask_tau_low_input, mask_tau_high_input],
        )
        use_adaptive_delta.change(
            fn=lambda v: [gr.update(interactive=bool(v)) for _ in
                          [adaptive_alpha_input, adaptive_floor_input]],
            inputs=use_adaptive_delta,
            outputs=[adaptive_alpha_input, adaptive_floor_input],
        )
        use_sharpen_u.change(
            fn=lambda v: gr.update(interactive=bool(v)),
            inputs=use_sharpen_u,
            outputs=sharpen_gamma_input,
        )
        use_latent_sharp.change(
            fn=lambda v: [gr.update(interactive=bool(v)) for _ in [latent_sharp_bg_input, latent_sharp_fg_input]],
            inputs=use_latent_sharp,
            outputs=[latent_sharp_bg_input, latent_sharp_fg_input],
        )
        if examples:
            gr.Markdown("## Examples")
            gr.Examples(
                examples=examples,
                inputs=[input_image, source_prompt, target_prompt],
                label="Click an example to auto-fill the left-side inputs.",
            )
        else:
            gr.Markdown("## Examples")
            gr.Markdown("No valid examples were found under the current dataset path.")

    return demo


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    component_paths = _resolve_component_paths(model_root=args.model_root)
    edit_config = dict(DEFAULT_EDIT_CONFIG)
    seed = DEFAULT_SEED
    torch_dtype = _dtype_from_precision(DEFAULT_PRECISION)
    compute_dtype = torch.float32

    dataset_root = DEFAULT_DATA_ROOT
    examples = load_examples(dataset_root=dataset_root, max_examples=DEFAULT_MAX_EXAMPLES)

    LOGGER.info("Loaded %d example records from %s", len(examples), dataset_root)
    LOGGER.info("Seed: %s | Default edit config: %s", seed, edit_config)
    LOGGER.info("Component paths: %s", component_paths)

    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device=None,
        torch_dtype=torch_dtype,
        image_size=DEFAULT_IMAGE_SIZE,
        use_center_crop=DEFAULT_CENTER_CROP,
        compute_dtype=compute_dtype,
        use_attention_mask=DEFAULT_USE_ATTENTION_MASK,
        use_safety_checker=DEFAULT_USE_SAFETY_CHECKER,
    )

    demo = build_demo(
        pipeline=pipeline, default_seed=seed,
        default_edit_config=edit_config, examples=examples,
    )
    demo.queue(api_open=False)
    demo.launch(server_name=DEFAULT_SERVER_NAME, server_port=args.server_port)


if __name__ == "__main__":
    main()
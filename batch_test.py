import os
import pandas as pd
import xlsxwriter
from PIL import Image
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import glob

# 导入你的主程序和 Pipeline
from pipeline_chord import ChordEditPipeline
from app import (
    _resolve_component_paths, DEFAULT_MODEL_ROOT, DEFAULT_PRECISION,
    DEFAULT_IMAGE_SIZE, DEFAULT_CENTER_CROP, DEFAULT_USE_ATTENTION_MASK,
    DEFAULT_USE_SAFETY_CHECKER, _dtype_from_precision,
    _mask_tensor_to_pil, DEFAULT_EDIT_CONFIG
)

# ================= 配置区 =================
EXCEL_INPUT = "temp_data/batch_cases1.xlsx"
OUTPUT_DIR = "batch_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU 推理锁
gpu_lock = threading.Lock()

# ================= 实验矩阵生成 (全参数密集网格搜索) =================
EXPERIMENTS = []

# 1. 基线 (全关)
EXPERIMENTS.append({"name": "00_baseline", "config": {}, "tag_bg": "关", "tag_time": "关", "tag_sharp": "关"})

# ==========================================
# 2. 空间自适应 (背景保护) 参数网格
# ==========================================
# 2.1 local_beta (0.0 ~ 1.0)
for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    EXPERIMENTS.append({
        "name": f"bg_beta_{str(v).replace('.', '_')}",
        "config": {"use_local_chord": True, "local_apply_transport": False, "local_beta": v},
        "tag_bg": f"开(β={v})", "tag_time": "关", "tag_sharp": "关"
    })
# 2.2 mask_sigma (0.0 ~ 4.0)
for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
    EXPERIMENTS.append({
        "name": f"bg_sigma_{str(v).replace('.', '_')}",
        "config": {"use_local_chord": True, "local_apply_transport": False, "mask_sigma": v},
        "tag_bg": f"开(σ={v})", "tag_time": "关", "tag_sharp": "关"
    })
# 2.3 mask_tau_low (0.0 ~ 0.5)
for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    EXPERIMENTS.append({
        "name": f"bg_taulow_{str(v).replace('.', '_')}",
        "config": {"use_local_chord": True, "local_apply_transport": False, "mask_tau_low": v},
        "tag_bg": f"开(τ_lo={v})", "tag_time": "关", "tag_sharp": "关"
    })
# 2.4 mask_tau_high (0.1 ~ 1.0)
for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    EXPERIMENTS.append({
        "name": f"bg_tauhigh_{str(v).replace('.', '_')}",
        "config": {"use_local_chord": True, "local_apply_transport": False, "mask_tau_high": v},
        "tag_bg": f"开(τ_hi={v})", "tag_time": "关", "tag_sharp": "关"
    })

# ==========================================
# 3. 时间自适应 参数网格
# ==========================================
# 3.1 adaptive_alpha (0.0 ~ 1.0)
for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    EXPERIMENTS.append({
        "name": f"time_alpha_{str(v).replace('.', '_')}",
        "config": {"use_adaptive_delta": True, "adaptive_alpha": v},
        "tag_bg": "关", "tag_time": f"开(α={v})", "tag_sharp": "关"
    })
# 3.2 adaptive_floor (0.0 ~ 1.0)
for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    EXPERIMENTS.append({
        "name": f"time_floor_{str(v).replace('.', '_')}",
        "config": {"use_adaptive_delta": True, "adaptive_floor": v},
        "tag_bg": "关", "tag_time": f"开(floor={v})", "tag_sharp": "关"
    })

# ==========================================
# 4. 清晰度增强 参数网格
# ==========================================
# 4.1 sharpen_gamma (0.0 ~ 1.5)
for v in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
    EXPERIMENTS.append({
        "name": f"sharp_gamma_{str(v).replace('.', '_')}",
        "config": {"use_sharpen_u": True, "sharpen_gamma": v},
        "tag_bg": "关", "tag_time": "关", "tag_sharp": f"开(γ={v})"
    })
# 4.2 latent_sharp_bg (0.0 ~ 1.0)
for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    EXPERIMENTS.append({
        "name": f"sharp_bg_{str(v).replace('.', '_')}",
        "config": {"use_latent_sharp": True, "latent_sharp_bg": v},
        "tag_bg": "关", "tag_time": "关", "tag_sharp": f"开(bg={v})"
    })
# 4.3 latent_sharp_fg (0.0 ~ 0.5)
for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    EXPERIMENTS.append({
        "name": f"sharp_fg_{str(v).replace('.', '_')}",
        "config": {"use_latent_sharp": True, "latent_sharp_fg": v},
        "tag_bg": "关", "tag_time": "关", "tag_sharp": f"开(fg={v})"
    })

# ==========================================
# 5. 协同组合实验 (全开 / 两两组合)
# ==========================================
EXPERIMENTS.extend([
    {"name": "99_combo_bg_time", "config": {"use_local_chord": True, "local_apply_transport": False, "use_adaptive_delta": True}, "tag_bg": "开", "tag_time": "开", "tag_sharp": "关"},
    {"name": "99_combo_bg_sharp", "config": {"use_local_chord": True, "local_apply_transport": False, "use_sharpen_u": True, "use_latent_sharp": True}, "tag_bg": "开", "tag_time": "关", "tag_sharp": "开"},
    {"name": "99_combo_time_sharp", "config": {"use_adaptive_delta": True, "use_sharpen_u": True, "use_latent_sharp": True}, "tag_bg": "关", "tag_time": "开", "tag_sharp": "开"},
    {"name": "99_combo_full", "config": {"use_local_chord": True, "local_apply_transport": False, "use_adaptive_delta": True, "use_sharpen_u": True, "use_latent_sharp": True}, "tag_bg": "开", "tag_time": "开", "tag_sharp": "开"},
])


def load_pipeline():
    print("正在加载模型，请稍候...")
    component_paths = _resolve_component_paths(DEFAULT_MODEL_ROOT)
    edit_config = dict(DEFAULT_EDIT_CONFIG)
    torch_dtype = _dtype_from_precision(DEFAULT_PRECISION)
    compute_dtype = torch.float32

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
    print("模型加载完成！")
    return pipeline


def run_batch():
    if not os.path.exists(EXCEL_INPUT):
        print(f"错误：找不到数据集 {EXCEL_INPUT}，请检查！")
        return

    df_cases = pd.read_excel(EXCEL_INPUT)
    if "tar_image_path" not in df_cases.columns:
        df_cases["tar_image_path"] = ""

    pipeline = load_pipeline()
    results = []
    tasks = []

    # 构建所有任务组合
    for idx, row in df_cases.iterrows():
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            print(f"警告：图片不存在 {img_path}")
            continue

        case_name = Path(img_path).stem
        for exp in EXPERIMENTS:
            tasks.append({
                "case_name": case_name,
                "img_path": img_path,
                "src_prompt": row["src_prompt"],
                "tar_prompt": row["tar_prompt"],
                "tar_img_path": row["tar_image_path"],
                "exp": exp
            })

    print(f"总共有 {len(df_cases)} 张图，生成了 {len(tasks)} 个推理任务，开始执行...")

    def process_task(task):
        case_name = task["case_name"]
        exp = task["exp"]
        exp_name = exp["name"]

        # 合并默认参数与当前实验参数
        edit_config = dict(DEFAULT_EDIT_CONFIG)
        edit_config.update(exp["config"])

        try:
            image = Image.open(task["img_path"]).convert("RGB")
        except Exception as e:
            print(f"读图失败 {task['img_path']}: {e}")
            return None

        # 加锁保护 GPU 前向
        with gpu_lock:
            try:
                result = pipeline(
                    image=image,
                    source_prompt=task["src_prompt"],
                    target_prompt=task["tar_prompt"],
                    edit_config=edit_config,
                    seed=42,
                    return_mask=True,
                )
            except Exception as e:
                print(f"推理失败 {case_name} - {exp_name}: {e}")
                return None

        # 保存生成的图片
        out_img_name = f"{case_name}_{exp_name}.png"
        out_img_path = os.path.join(OUTPUT_DIR, out_img_name)
        result.images[0].save(out_img_path)

        # 保存 Mask (如果背景保护没开，这里会返回 None，跳过)
        mask_path = ""
        if result.masks is not None:
            mask_pil = _mask_tensor_to_pil(result.masks)
            mask_name = f"{case_name}_{exp_name}_mask.png"
            mask_path = os.path.join(OUTPUT_DIR, mask_name)
            mask_pil.save(mask_path)

        return {
            "scr 图": task["img_path"],
            "scr prompt": task["src_prompt"],
            "tar prompt": task["tar_prompt"],
            "tar 图": task["tar_img_path"],
            "创新tar图": out_img_path,
            "背景增强": exp["tag_bg"],
            "时间自适应": exp["tag_time"],
            "图像清晰": exp["tag_sharp"],
            "mask 图": mask_path if mask_path else "无"
        }

    # 多线程执行 (线程池调度)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, res in enumerate(executor.map(process_task, tasks)):
            if res is not None:
                results.append(res)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                print(f"进度: {i + 1}/{len(tasks)}")

    # 导出带图片的 Excel
    if not results:
        print("没有有效结果生成，流程结束。")
        return

    df_results = pd.DataFrame(results)
    output_excel = os.path.join(OUTPUT_DIR, "batch_report.xlsx")

    print("正在写入 Excel 并插入缩略图...")
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, sheet_name='Results', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Results']

        # 调整列宽
        worksheet.set_column('A:A', 35)
        worksheet.set_column('B:C', 45)
        worksheet.set_column('D:E', 35)
        worksheet.set_column('F:H', 12)
        worksheet.set_column('I:I', 35)

        # 插入缩略图
        def insert_thumbnail(row, col, img_path, max_size=(120, 120)):
            if pd.isna(img_path) or not os.path.exists(img_path) or img_path == "无":
                return
            try:
                img = Image.open(img_path)
                img.thumbnail(max_size)
                tmp_path = f"{img_path}.thumb.jpg"
                img.convert("RGB").save(tmp_path)
                # 插入并留白
                worksheet.insert_image(row, col, tmp_path, {'x_offset': 5, 'y_offset': 5})
                worksheet.set_row(row, 100) # 行高
            except Exception as e:
                print(f"插入图片出错 {img_path}: {e}")

        for i, row in df_results.iterrows():
            excel_row = i + 1  # 表头占0行
            insert_thumbnail(excel_row, 0, row["scr 图"])
            insert_thumbnail(excel_row, 3, row["tar 图"])
            insert_thumbnail(excel_row, 4, row["创新tar图"])
            insert_thumbnail(excel_row, 8, row["mask 图"])

    # 清理临时缩略图
    for thumb in glob.glob(os.path.join(OUTPUT_DIR, "*.thumb.jpg")):
        try:
            os.remove(thumb)
        except:
            pass

    print(f"全部测试完成！结果表格保存为: {output_excel}")


if __name__ == "__main__":
    run_batch()
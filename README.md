<div align="center">
  <h1>[CVPR 2026] ChordEdit: One-Step Low-Energy Transport for Image Editing</h1>
  <div class="authors">
    <span><a href="https://luliangsi.github.io">Liangsi Lu</a><sup>1</sup>, <a href="https://cxh.netlify.app/">Xuhang Chen</a><sup>2</sup>, <a href="#">Minzhe Guo</a><sup>1</sup>, <a href="#">Shichu Li</a><sup>3</sup>, <a href="#">Jingchao Wang</a><sup>4</sup>, <a href="https://cnshiyang.github.io">Yang Shi</a><sup>1†</sup></span><br>
    <span style="color: #666; font-size: 0.9em;"><sup>1</sup> Guangdong University of Technology, <sup>2</sup> Huizhou University, <sup>3</sup> Shenzhen University, <sup>4</sup> Peking University <br> <sup>†</sup> Corresponding author</span>
  </div>

  <a href="https://chordedit.github.io"><img src="https://img.shields.io/badge/Project-Page-2b7de9"></a>
  <a href="https://arxiv.org/pdf/2602.19083"><img src="https://img.shields.io/badge/arXiv-2602.19083-b31b1b.svg"></a>

  <img src="chord_show.gif" alt="ChordEdit demo" width="100%" />
</div>

## 1. Environment
- Python 3.12
- PyTorch 2.5.0
- This repository requires the `sd-turbo` weights: https://huggingface.co/stabilityai/sd-turbo
- Model root should contain:
  - `unet/`
  - `scheduler/`
  - `text_encoder/`
  - `tokenizer/`
  - `vae/`

## 2. Install Dependencies
```bash
pip install -r requirement.txt
```

## 3. Run the Web Demo
Launch the interactive demo:
```bash
python app.py --model-root /path/to/sd-turbo --server-port 7860
```

Running `python app.py` now launches a local Gradio web app.
- Left panel: upload the original image, set source prompt, target prompt, and tuning parameters.
- Right panel: view the edited output image.
- Bottom section: click built-in examples (image + source prompt + target prompt) to auto-fill inputs.

<img src="chord_app.png" alt="ChordEdit app" width="100%" />

## 4. Run PIE Benchmark Export
Run PIE-Bench export with:
```bash
python run_pie_bench.py --model-root /path/to/sd-turbo --pie-root /path/to/pie_bench
```
`--pie-root` should point to a PIE-Bench folder containing at least:

1. `annotation_images/` — original PIE-Bench images (subfolders keep the official naming).
2. `mapping_file.json` — the mapping metadata describing prompts, instructions, and masks.

Example layout:
```
pie_bench
|-annotation_images
|-mapping_file.json
```

For PIE-Bench data preparation and protocol details, please refer to:
https://github.com/cure-lab/PnPInversion

# Citation
If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!
```
@misc{lu2026chordedit,
      title={ChordEdit: One-Step Low-Energy Transport for Image Editing}, 
      author={Liangsi Lu and Xuhang Chen and Minzhe Guo and Shichu Li and Jingchao Wang and Yang Shi},
      year={2026},
      eprint={2602.19083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.19083}, 
}
```

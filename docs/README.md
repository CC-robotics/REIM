# REIM 中文讲解资料

本目录包含面向项目答辩、组会和代码交接的中文技术材料：

- `REIM_technical_explanation_zh.md`：权威文字源文件；
- `REIM_technical_explanation_zh.docx`：可编辑技术详解；
- `REIM_technical_explanation_zh.pdf`：技术详解 PDF；
- `REIM_presentation_zh.pptx`：22 页可编辑演示稿；
- `REIM_presentation_zh.pdf`：演示稿 PDF；
- `assets/`：由脚本生成的讲解图、论文图裁剪和流程图。

重新生成可编辑文档和演示稿：

```bash
python -m venv .tools/docs_venv
.tools/docs_venv/bin/pip install -r requirements-docs.txt
.tools/docs_venv/bin/python scripts/generate_explanation_assets.py
```

PDF 使用 LibreOffice 的无界面模式从 DOCX/PPTX 导出：

```bash
libreoffice --headless --convert-to pdf --outdir docs \
  docs/REIM_technical_explanation_zh.docx \
  docs/REIM_presentation_zh.pptx
```

讲解中的正式实验数值以 `results/tables/`、`results/run_manifest.json` 和 `paper_assets/reim_results.pdf` 为证据源。

科研口径：当前恢复策略是触发对齐的监督模仿 actor，不是 PPO；当前结果来自 Meta-World/MuJoCo 模拟 Sawyer，不是实体机器人实验。

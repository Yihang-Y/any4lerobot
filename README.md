# any4lerobot

将多个机器人数据源转换到 LeRobot 格式的工具集合。目前重点包含 OpenX 与 AgiBot 的快速转换流程，以及 DROID 的后处理脚本。

## 安装

```bash
uv sync
```

## 数据下载以及转换 

### DROID数据集

先从 Hugging Face 拉取 DROID 元数据（`KarlP/droid`）：

```bash
uv run hf download \
  KarlP/droid \
  --local-dir ./info/droid
```

```bash
mkdir -p <your_local_path>/droid
uv run gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_local_path>/droid
```

```bash
uv run python openx2lerobot/openx_rlds.py \
  --raw-dir <your_local_path>/droid/1.0.1 \
  --local-dir <your_local_path>/droid_lerobot \
  --use-videos \
  --lang-ann ./info/droid/droid_language_annotations.json \
  --keep-ranges ./info/droid/keep_ranges_1_0_1.json
```

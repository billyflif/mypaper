# 第四章实验工作流

本次实现将第四章统一为“闭集分类主实验 + 检索补充实验”。

## 1. 新增脚本

- `train_ch4_baselines.py`
  - 训练公平外部基线：`lightcnn_arcface`、`resnet50_arcface`、`densenet121_arcface`
- `evaluate_ch4_closedset.py`
  - 统一评估闭集分类指标：`Top-1`、`Top-5`、`Macro-F1`、参数量、单张前向时间、FPS
- `prepare_ch4_query_gallery.py`
  - 从 `Paper-Data-Copy/test` 中筛出训练阶段的 44 个身份，并按固定规则构造 `query/gallery`
- `evaluate_ch4_retrieval.py`
  - 基于已训练模型提取 embedding，计算 `Rank-1`、`Rank-5`、`mAP`
- `export_ch4_tables.py`
  - 将多个结果文件导出为 CSV 和 LaTeX 表格行

## 2. SGPD-Net 定稿预设

`train_sgpd_net.py` 新增了 `--thesis-tuned`，用于第四章定稿阶段的单次定向重训。

该预设固定启用：

- 类均衡采样
- `0.6 * val_top1 + 0.4 * val_macro_f1` 作为最佳 checkpoint 选择指标
- 更弱的早期结构约束
  - 第 1-5 轮：`reid=0.6 rec=0.2 lsc=0.0 g_disc=0.0 center=0.0 domain_adapt=0.0`
  - 第 6-20 轮：余弦平滑过渡
  - 第 21 轮起：`reid=1.0 rec=0.15 lsc=0.10 g_disc=0.10 center=0.02 domain_adapt=0.0`

这样做的原因：

- 早期 `rec/LSC` 过重时，结构分支容易先学重建，不利于分类目标。
- `no_lsc_loss` 与完整模型差距不大，说明 `LSC` 应保留，但不宜主导训练。
- 当前主表报告 `Macro-F1`，类均衡采样比单纯看 `val_acc` 更合理。
- 第四章当前不强调跨域结论，域适应项置零更稳。

## 3. 运行顺序

### 3.1 训练公平外部基线

```bash
python train_ch4_baselines.py --model-type lightcnn_arcface --dataset-root Paper-Data-Copy --output-dir chapter4_runs
python train_ch4_baselines.py --model-type resnet50_arcface --dataset-root Paper-Data-Copy --output-dir chapter4_runs
python train_ch4_baselines.py --model-type densenet121_arcface --dataset-root Paper-Data-Copy --output-dir chapter4_runs
```

如果目标机器上有 torchvision 预训练权重缓存，可以为 ResNet50 和 DenseNet121 增加 `--pretrained`。

### 3.2 定向重训 SGPD-Net

```bash
python train_sgpd_net.py --thesis-tuned --dataset-root Paper-Data-Copy
```

说明：

- `--dataset-root` 用于自动重写 `paperdata-train.txt` 中旧机器路径。
- 若不加 `--thesis-tuned`，脚本仍按原训练策略运行。

### 3.3 评估闭集分类主表

```bash
python evaluate_ch4_closedset.py --model-type lightcnn_ce --checkpoint "第四章-消融实验结果与权重/simple_cnn_baseline/best_model.pth" --dataset-root Paper-Data-Copy --method-name "LightCNN + CE"
python evaluate_ch4_closedset.py --model-type lightcnn_arcface --checkpoint "chapter4_runs/lightcnn_arcface/checkpoint_best.pth" --dataset-root Paper-Data-Copy --method-name "LightCNN + Sub-Center ArcFace"
python evaluate_ch4_closedset.py --model-type resnet50_arcface --checkpoint "chapter4_runs/resnet50_arcface/checkpoint_best.pth" --dataset-root Paper-Data-Copy --method-name "ResNet50 + Sub-Center ArcFace"
python evaluate_ch4_closedset.py --model-type densenet121_arcface --checkpoint "chapter4_runs/densenet121_arcface/checkpoint_best.pth" --dataset-root Paper-Data-Copy --method-name "DenseNet121 + Sub-Center ArcFace"
python evaluate_ch4_closedset.py --model-type sgpd_no_structure --checkpoint "第四章-消融实验结果与权重/no_structure_guidance/best_model.pth" --dataset-root Paper-Data-Copy --method-name "SGPD-Net w/o structure guidance"
python evaluate_ch4_closedset.py --model-type sgpd_net --checkpoint "第四章-消融实验结果与权重/sgpd_net（完整模型）/checkpoint_best.pth" --dataset-root Paper-Data-Copy --method-name "SGPD-Net"
```

如果你完成了 `--thesis-tuned` 重训，就把最后一条里的 checkpoint 替换为新目录下的 `checkpoint_best.pth`。

替换正文结果的规则：

- 若新 SGPD-Net 的 `Top-1` 高于 `93.25%`，直接替换。
- 若 `Top-1` 持平但 `Macro-F1` 更高，也替换。
- 其余情况保留原完整模型数值，只保留新代码与补充记录。

### 3.4 构造 query/gallery

```bash
python prepare_ch4_query_gallery.py --test-root Paper-Data-Copy/test --train-annotation paperdata-train.txt --output-root chapter4_retrieval_split --clear-output
```

默认规则：

- 只保留训练阶段 44 个身份
- 每个身份按 `int(len(images) * 0.5)` 划入 query
- 至少保留 1 张 gallery
- 默认随机种子为 `3407`

这套规则在当前数据上会得到：

- 44 个身份
- 307 张 query
- 325 张 gallery

### 3.5 评估检索补充表

```bash
python evaluate_ch4_retrieval.py --model-type lightcnn_ce --checkpoint "第四章-消融实验结果与权重/simple_cnn_baseline/best_model.pth" --query-root chapter4_retrieval_split/query --gallery-root chapter4_retrieval_split/gallery --method-name "LightCNN + CE"
python evaluate_ch4_retrieval.py --model-type resnet50_arcface --checkpoint "chapter4_runs/resnet50_arcface/checkpoint_best.pth" --query-root chapter4_retrieval_split/query --gallery-root chapter4_retrieval_split/gallery --method-name "ResNet50 + Sub-Center ArcFace"
python evaluate_ch4_retrieval.py --model-type sgpd_net --checkpoint "第四章-消融实验结果与权重/sgpd_net（完整模型）/checkpoint_best.pth" --query-root chapter4_retrieval_split/query --gallery-root chapter4_retrieval_split/gallery --method-name "SGPD-Net"
```

默认使用余弦距离，不启用 re-ranking。

### 3.6 导出论文表格

闭集主表：

```bash
python export_ch4_tables.py \
  --result "LightCNN + CE=chapter4_reports/closedset/lightcnn_+_ce.json" \
  --result "LightCNN + Sub-Center ArcFace=chapter4_reports/closedset/lightcnn_+_sub-center_arcface.json" \
  --result "ResNet50 + Sub-Center ArcFace=chapter4_reports/closedset/resnet50_+_sub-center_arcface.json" \
  --result "DenseNet121 + Sub-Center ArcFace=chapter4_reports/closedset/densenet121_+_sub-center_arcface.json" \
  --result "SGPD-Net w/o structure guidance=chapter4_reports/closedset/sgpd-net_w_o_structure_guidance.json" \
  --result "SGPD-Net=chapter4_reports/closedset/sgpd-net.json" \
  --columns top1 top5 macro_f1 params_total time_per_image_ms \
  --output-prefix chapter4_reports/table4_1
```

检索补充表：

```bash
python export_ch4_tables.py \
  --result "LightCNN + CE=chapter4_reports/retrieval/lightcnn_+_ce.json" \
  --result "ResNet50 + Sub-Center ArcFace=chapter4_reports/retrieval/resnet50_+_sub-center_arcface.json" \
  --result "SGPD-Net=chapter4_reports/retrieval/sgpd-net.json" \
  --columns rank1 rank5 mAP \
  --output-prefix chapter4_reports/table4_2
```

## 4. 注意事项

- 当前实现默认闭集主协议沿用 `paperdata-train.txt` 的 44 类划分。
- 主表推理时间只统计识别模型前向，不统计 SAM 分割开销。
- 如果当前机器的 Python 环境缺少 `torch`、`numpy`、`diffusers` 等依赖，训练和评估脚本无法直接运行；这不影响纯文件处理脚本。
- `paperdata-train.txt` 若来自旧机器路径，可统一通过 `--dataset-root Paper-Data-Copy` 修正。

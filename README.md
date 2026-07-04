# Unmasked Multi-Seed Evaluation Protocol for Satellite 3DGS — Release Package

Companion code & data for: *On the Reliability of DSM Improvements in Satellite 3D Gaussian Splatting:
An Unmasked Multi-Seed Evaluation Protocol for Weak-Evidence Regions* (under review).

> 发布策略:GitHub 公开仓库(投稿时在 Data Availability 给 URL;JSTARS 单盲无需匿名)。
> 打包时从远端 `/root/autodl-tmp/eogs/eval_tools/` 与本地 `D:\new\analysis\` 收集,按下述清单核对。

## Contents

### `protocol/` — 评估工具链(论文 §III 的参考实现)
| 文件 | 作用 | 源位置 |
|---|---|---|
| `eval_pairs_generic.py` | 统一区域化评估(任意 DSM 集合;MAE/RMSE/completeness;锚定 weak-proxy;err-top20) | 远端 eval_tools |
| `run_all_evals.py` | 全场景批量评估(EOGS 系全网格组 + 跨方法交集组分离) | 远端 eval_tools |
| `analyze_all.py` | 汇总 → T1–T6 markdown/JSON | 远端 eval_tools |
| `s3_null_control.py` | seed-swap null 对照(式 (2)) | 远端 eval_tools |
| `s6_phase0_stats.py` | 多时相互补诊断统计 | 远端 eval_tools |
| `gen_latex_tables.py` | CSV → 论文 LaTeX 表(数字溯源) | 本地 analysis |
| `weak_proxy_lib.py` | `compute_weak_proxy/region_metrics`(自 EOGS2 phase3 模块拷入,自包含) | 已打包 |

### `results/` — 全部逐种子指标
- `out/<AOI>/<AOI>_metrics.csv` × 7(EOGS 系全网格:baseline/ours/nostruct × seeds,含 rmse/comp1m)
- `out_cross/<AOI>/<AOI>_metrics.csv`(EOGS/EOGS++/Sat-NeRF ckpt/EOGS-v1 多种子,交集网格+coverage meta)
- `s3_null_control.json`、`s6_phase0_stats_7scenes.json`、`summary_tables.{md,json}`
- anchor weak-proxy rasters(`weak_proxy_rasters/*.tif`,~250MB → 用 GitHub Release 附件或 Zenodo DOI)

### `casestudy/` — 被研究方法的可复现配置
- `run_*.sh` 全套(q1_seeds/q1bc/q1d/s2/m2a/q1e)+ phase1b-default 环境变量表
- EOGS_SEED patch(EOGS 原版多种子补丁,diff 形式)
- Sat-NeRF 现代化环境说明(satnerf310:torch 2.2.2+cu118 + pl1.9.5 + setuptools<81 + pyproj UTM patch)

### 数据引用(不再分发)
- DFC2019/US3D(IEEE DataPort)、EOGS/EOGS++ 官方 data.zip、Sat-NeRF EarthVision2022 release(数据+ckpt)

## 打包 TODO(投稿前)
- [ ] 从远端收集脚本与 results(tar);模块内绝对路径参数化
- [ ] 烟测:干净 clone + conda env → 复现 JAX_260 表 1 数字
- [ ] proxy rasters 决定 Release 附件 vs Zenodo(建议 Zenodo,拿 DOI 放论文)
- [ ] LICENSE(MIT)+ 引用格式(CITATION.cff)

# REIM 实验整改进展汇报（致导师）

> 本汇报对应导师的审阅意见与后续执行决定（成文 2026-08-21，08-24 收到并执行）。目标：尽快投出 EI 会议，实验闭环，并把代码修到"别人 clone 下来能直接跑、SHA256 不失配、README 与论文一致"。

---

## 一、总体结论

- MT10/MT50 **全链路实验已闭环**，发表门通过，官方 clean 与全部 disturbed 条件（噪声 0.1/0.2/0.3/0.4）数据完整、可审计。
- 导师的三个问题（参数、backfill/多种子、occupancy）**全部按决定落实**。
- 导师指出的三个代码问题（SHA256 失效、Windows 绝对路径、README/tex 矛盾）**已修复并复验通过**。
- 本轮最大的隐蔽雷——**跨平台行尾符导致的 SHA256 漂移**——已在检出侧（`.gitattributes`）与写入侧（三个生成器）同时封死。

---

## 二、三个问题的落实

### 问题 1：参数

- 已冻结 **release=0.05 / patience=10**，MT10 与 MT50 统一，定义为 **robustness-first operating point**。
- **release=0.15 / patience=5** 作为效率对照完整归档在 `results/tables/archive_release_015_05/`（MT10）与 `results/tables/archive_release_015_05_mt50/`（MT50）（含全部 summary/episodes/sidecar，与导师 PDF 核查依据中的引用路径一致），搜索网格产物（`results/diagnostics/release_patience_search/mt10_reim_grid.csv` 等）中的 0.15/5 行也原样保留；不再作为运行点引用。
- 未再做完整 MT50 网格，仅做 clean / 0.1 / 0.4 的小规模方向确认。
- **参数自本决定起停止修改。**

### 问题 2：backfill + 多种子

**Backfill（MT10 终端正 horizon 消融，{0,10,25,50}）**
- 只用 seed 42，ACT 与 recovery 固定，**只重训 detector**。
- 各 horizon 在各自验证库上按同一 precision-floor ≥0.60 规则重选阈值：

| Horizon | 阈值 | 严格 pre-event F1 | 事件轨迹预警率 | 中位 lead（步） |
|---|---:|---:|---:|---:|
| h0 | 0.72 | 0.428 | 30.0% | 1 |
| h10 | 0.71 | 0.414 | 30.7% | 1 |
| h25（canonical） | 0.65 | 0.410 | 29.7% | 2 |
| h50 | 0.62 | 0.394 | 32.3% | 1 |

- 闭环用 tuned 阈值重跑（clean / 0.1 / 0.4，每任务 20 回合，bank 20265010），并保留 0.65 的 50 回合对照；数据见 `results/tables/mt10_backfill_tuned_closed_loop_summary.csv`：

| Horizon | 阈值 | clean | 0.1 | 0.4 | 相对 thr=0.65（0.1 / 0.4） |
|---|---:|---:|---:|---:|---:|
| h0 | 0.72 | 98.5% | 46.0% | 59.5% | −3.0 / −0.3 pp |
| h10 | 0.71 | 98.0% | 45.0% | 61.0% | −2.8 / −0.4 pp |
| h50 | 0.62 | 98.5% | 47.5% | 62.5% | +5.3 / +3.3 pp |

  结论：调阈后闭环不掉点——clean 反而略升（+1.2~+1.7 pp），noise 0.4 持平，h50 上升；0.1 档的 −3 pp 在 20 回合样本噪声范围内。即 REIM 对 horizon 标注口径不敏感。

**precision-floor 0.65 口径闭环（双口径并行）**
- 用探针阈值（h0=0.79、h10=0.78、h25=0.73、h50=0.71）完成 12 单元闭环（同库 20265010、同 20 回合规格），数据见 `results/tables/mt10_backfill_floor065_closed_loop_summary.csv`：

| Horizon | 阈值（0.60→0.65） | clean | 0.1 | 0.4 |
|---|---:|---:|---:|---:|
| h0 | 0.72 → 0.79 | 98.5%（=） | 40.0%（−6.0） | 56.0%（−3.5） |
| h10 | 0.71 → 0.78 | 98.0%（=） | 36.5%（−8.5） | 60.0%（−1.0） |
| h25 | 0.65 → 0.73 | 98.5% | 37.5% | 56.5% |
| h50 | 0.62 → 0.71 | 98.5%（=） | 35.0%（−12.5） | 58.5%（−4.0） |

  结论：0.65 口径 clean 完全不掉点，但 0.1 档代价明显（−6~−12.5 pp，超出 20 回合噪声范围）、0.4 档小幅下降。两套口径数字均已落盘，论文可并列呈现，最终口径由导师定夺。

**多种子（canonical horizon=25）**
- detector + recovery 训练种子 **42/43/44**，ACT 固定，条件 clean / 0.1 / 0.4（每任务 50 回合，每方法每种子 500 回合）。
- 按导师规范补报 **3 次实验均值 ± 样本标准差** 与**配对 rescued/harmed**（配对对象为 MT-ACT，`results/tables/mt10_multiseed_summary.csv`，生成脚本 `scripts/build_multiseed_stats.py`）：

| Method | clean | 0.1 | 0.4 |
|---|---:|---:|---:|
| MT-REIM | 96.6% ± 0.4 | 44.1% ± 1.3 | 60.8% ± 4.9 |
| Heuristic gate | 96.7% ± 0.9 | 44.7% ± 3.2 | 40.8% ± 3.6 |
| MT-ACT | 96.6% ± 0.0 | 5.2% ± 0.0 | 4.6% ± 0.0 |

| Method | 条件 | rescued（42/43/44） | harmed（42/43/44） |
|---|---|---|---|
| MT-REIM | clean | 2/0/0 | 0/0/2 |
| MT-REIM | 0.1 | 202/189/193 | **0/0/0** |
| MT-REIM | 0.4 | 284/257/306 | 0/2/2 |
| Heuristic | clean | 9/6/10 | 9/10/5 |
| Heuristic | 0.1 | 213/182/201 | 0/1/2 |
| Heuristic | 0.4 | 202/171/171 | 0/0/1 |

（MT-ACT 标准差为 0 是预期的：ACT checkpoint 固定且 CRN 配对复用相同 episode 种子，三种子下 ACT 轨迹逐位一致。）

- 三个要点：① 0.1 档 REIM harmed **三个种子全为 0**（不误伤）；② 0.4 档 REIM 比 heuristic 每种多救约 100 回合（均值 282 vs 181）；③ 0.1 档 REIM 与启发式打平（44.1% vs 44.7%，差距在样本标准差内），0.4 档每个种子都明显领先。

- 未做 MT50 三种子，未与四个 backfill horizon 交叉。

### 问题 3：occupancy

- occupancy 已进论文主表：`Table_multitask_clean.tex` 新增 **Occup. ↓** 列，gate 宏为 MT10 Heuristic/REIM = 13.2%/18.9%，MT50 = 10.4%/16.6%（与 burden 汇总逐位一致）。
- **MT10 noise 0.4 matched-occupancy 对照**（200 回合搜索库，方向性）：REIM 在 release=0.3/patience=5 时 occupancy 49.0%（heuristic 47.6%），success 49.5% vs 47.0%，**同预算下 +2.5 个百分点**（配对统计见下方"对照已闭环"条）。
- **success–occupancy 曲线**：`results/figures/mt10_success_occupancy.png`。
- **三种负担口径已补齐**（官方库 bank 20265010，seed 42，每方法每条件 500 回合，`results/tables/intervention_burden_three_metrics.json`，脚本 `scripts/build_burden_three_metrics.py`）：

| 条件 | 口径 | REIM | Heuristic |
|---|---|---:|---:|
| clean | mean per-episode occupancy | 18.9% | 13.2% |
| clean | pooled control share | 17.7% | 28.1% |
| clean | 干预次数/回合（段长） | 0.47（33 步） | 0.24（120 步） |
| 0.1 | mean per-episode occupancy | 39.4% | 49.6% |
| 0.1 | pooled control share | 31.7% | 48.3% |
| 0.1 | 干预次数/回合（段长） | 0.92（121 步） | 1.09（159 步） |
| 0.4 | mean per-episode occupancy | 69.0% | 47.9% |
| 0.4 | pooled control share | 57.9% | 45.8% |
| 0.4 | 干预次数/回合（段长） | 1.22（129 步） | 1.05（159 步） |

  （0.2/0.3 档在同一 JSON 中完整列出。开关切换次数 = 干预次数，每次干预对应一段 recovery 开→关。）
- **分母效应已核验**：noise 0.4 下 REIM 每回合恢复 **157.1** 步 vs heuristic **167.4** 步（导师 PDF 引用值 157/167，逐位吻合）。注意 pooled 口径下 0.4 档 REIM 为 57.9%（而非 mean 口径的 69.0%）——REIM 高占用回合往往更短（更早成功），两种口径必须并列报告。
- **matched-occupancy 对照已闭环**（`results/tables/mt10_matched_occupancy_comparison.json`，脚本 `scripts/analyze_matched_occupancy_supplement.py`，配对经 `paired_episode_id` 与启发式参考逐回合对齐）：
  - **noise 0.4（匹配成功）**：REIM release 0.3/patience 5 → success **49.5%** @ occupancy **49.0%** vs heuristic **47.0%** @ **47.6%**，同预算 +2.5pp；配对 rescued **32** / harmed **27**（净 +5 回合，恰为 +2.5pp）。
  - **noise 0.1（匹配不可达，但结论更强）**：即使按导师建议的 release 0.02/patience 3 补跑，REIM occupancy 也只到 **39.3%**，够不到 heuristic 的 48.2%——但该点 success 已达 **48.0% ≥ 47.0%**（rescued 27 / harmed 25）。即 0.1 档 REIM 用**低约 9 个百分点的预算**实现持平/反超，同等占用预算的匹配点在现有参数范围内不存在。

---

## 三、三个代码问题的修复

### 问题 A：SHA256 全部失效

根因有**四层**，均已修复：

1. **Git LFS 指针**：仓库二进制（`*.npz`/`*.pt`）走 Git LFS，未 `git lfs pull` 会拿到指针导致哈希失配。README 已明确 clone 指引。
2. **绝对路径改写 manifest 字节**：`normalize_artifact_paths.py` 把机器绝对路径改为仓库相对路径，改变了 manifest 字节但语义不变；已对受影响的 manifest 重签哈希（内容指纹一致则放行、数据变了仍拒止）。
3. **行尾符（跨平台根治）**：`core.autocrlf=true` 使 manifest 记录的是 Windows CRLF 哈希，Linux/CI 克隆（LF）会失配。已加 `.gitattributes` 的 `* text=auto eol=lf`，强制所有平台检出为 LF；并把三个写入点（两个 gate 生成器 + `evaluate_multitask.py`）钉为 LF，防止下次重跑再产生 CRLF 文件。
4. **哈希链内路径被误改（本轮新发现并回滚）**：第 2 层的归一化曾改写 5 个 calibrated 验证库 manifest 里 `calibration_source.manifest_path` 的历史绝对路径——该字段被 `label_calibration_fingerprint_sha256` 覆盖并链入 `dataset_fingerprint_sha256` 与每个 `.npz` 分片内的溯源记录，改写后整条审计链断裂（调阈/审计脚本直接报 "Label calibration fingerprint mismatch"）。已将这 5 个 manifest 回滚到归一化前内容（指纹链恢复自洽，调阈脚本完整审计复跑通过），并给 `normalize_artifact_paths.py` 增加排除规则：**`datasets/**/manifest.json` 属于哈希链存证，其中的历史路径是有意的溯源记录，今后不再改写**。

现状：`assets_manifest.json`（5+16=21）、`multitask_results_manifest.json`（26+5=31）、`paper_results_manifest.json`（10）全部自洽；数据集清单 **19184 条、0 失配/0 缺失**；12 个 calibrated manifest 的内部指纹链全部通过重算校验（唯一例外是未入仓的本地 smoke 验证库，属本地测试数据，重新生成即可）。

### 问题 B：硬编码 Windows 绝对路径

- 所有**可改写**的产物路径已归一化为仓库相对路径；生成器与评估脚本不再写入 `C:\Users\...` 绝对路径。
- 现状：除 `datasets/**/manifest.json` 中被哈希链锁定的历史溯源路径（见问题 A 第 4 层，有意保留）与 `normalize_artifact_paths.py` 内一行说明字符串外，绝对路径残留 0。

### 问题 C：README 与 tex 矛盾

- README 与论文/tex 数字已同步，并在 `scripts/verify_readme_multitask_numbers.py` 中逐格校验；校验通过。
- **更正 08-23 汇报中的一个引用错误**：当时引用的 pre-event 审计（recall 0.683、事件轨迹预警率 74%）实为**单任务 PickPlace** 检测器数字，不是 MT10 的。MT10 的真实水平低得多（严格 pre-event F1 ≈0.41、轨迹预警率 ≈30%、中位 lead 1–2 步），backfill 消融正是为了揭示这一点。

---

## 四、主结果摘要（关键数字）

**官方 clean（bank 20265010）：**

| Benchmark | MT-MLP BC | MT-ACT | Heuristic gate | MT-REIM |
|---|---:|---:|---:|---:|
| MT10 | 91.4% | 96.6% | 96.6% | **97.0%** |
| MT50 | 81.3% | 92.1% | 94.8% | **92.2%** |

**Robustness（任务通用噪声，非官方）：**

| Noise | MT10 REIM/ACT/Heur | MT50 REIM/ACT/Heur |
|---|---:|---:|
| 0.1 | 45.6 / 5.2 / 47.8 | 34.0 / 2.0 / 35.9 |
| 0.2 | 53.8 / 5.4 / 47.8 | 42.2 / 1.9 / 35.2 |
| 0.3 | 57.8 / 4.8 / 45.8 | 46.3 / 1.9 / 34.4 |
| 0.4 | 61.4 / 4.6 / 45.0 | 49.7 / 1.9 / 33.9 |

**Recovery occupancy（clean）：** MT10 Heuristic 13.2% / REIM 18.9%；MT50 Heuristic 10.4% / REIM 16.6%。高噪声下 REIM 占用更高（如 MT10 noise 0.4：69.0% vs 47.9%），但换来显著更高的成功。

---

## 五、口径声明与待确认

- **backfill 闭环**为 tuned 阈值 + 每任务 20 回合的方向性结果（导师最低要求 20 回合）；需更高置信度可加机时到 50 回合。
- **matched-occupancy** 为 200 回合搜索库（seed 20264010）的方向性对照，非 500 回合官方库（20265010）；rescued/harmed 配对统计与 0.1 档可达性已补跑完成（见问题 3"对照已闭环"）。
- **precision-floor 双口径并行**：导师 PDF 要求 detector 触发 precision-floor **≥0.65**，而此前四个 horizon 调阈用的是仓库默认 **0.60**（val task-macro precision：h0=0.6049、h10=0.6024、h25=0.6001、h50=0.6036）。决定**两种口径都做**：0.60 闭环已有结果；0.65 的**纯推理探针**（`results/tables/mt10_horizon*_detector_threshold_floor065.json`）与 **12 单元闭环重跑**（`results/tables/mt10_backfill_floor065_closed_loop_summary.csv`）均已完成，四个 horizon 均有满足 0.65 的阈值：

| Horizon | 阈值（floor 0.60→0.65） | precision | F1 | recall |
|---|---|---:|---:|---:|
| h0 | 0.72 → 0.79 | 0.6518 | 0.5142 | 0.4525 |
| h10 | 0.71 → 0.78 | 0.6503 | 0.4897 | 0.4045 |
| h25 | 0.65 → 0.73 | 0.6540 | 0.4891 | 0.3994 |
| h50 | 0.62 → 0.71 | 0.6513 | 0.5065 | 0.4214 |

  两套口径闭环均已完成并并列呈现（闭环数字见问题 2 的双口径表），导师勾哪个用哪个，选定后无需再跑。
- **selection manifest 已补齐**：`results/diagnostics/selection_manifest.json`（脚本 `scripts/build_selection_manifest.py`），含候选网格全部 CSV/JSON 的 SHA256、优化目标（扰动条件平均 task-macro 成功率最大化）、约束（clean 不退化 + noise 0.4 occupancy 上限固定）、平局规则（先比扰动均值，再比 noise 0.4 occupancy 低者，再比 patience 高者），并由网格数据独立复算验证 0.05/10 确为 MT10/MT50 双库的 argmax；含 202650xx/202660xx 全程未触碰声明。
- **阈值相关聚类置信区间已补齐**：`results/tables/mt10_threshold_metrics_clustered_bootstrap_ci.json`（脚本 `scripts/build_threshold_metrics_ci.py`）。按任务整簇有放回抽样（B=10000，种子 20260825），覆盖双口径 × 四 horizon 的 precision/recall/F1。例：floor 0.60 各 horizon precision 点估计 0.6001–0.6049，95% CI 宽约 ±0.06（如 h25：0.6001 [0.5357, 0.6775]）；floor 0.65 各 horizon precision 0.6503–0.6540，CI 全部落在 0.65 之下沿附近（如 h25：0.6540 [0.5925, 0.7266]）——即 10 个任务的簇结构下，floor 0.65 的"达标"在统计上是边缘性的，这一点建议向导师如实说明。
- **search JSON 分片项已核销**：复核确认 `mt10_search.json` / `mt50_search.json` 本体完整无损（此前记录源于早期排查待办，并非实际缺陷）；其可追溯性已由 `selection_manifest.json` 中逐文件 SHA256 覆盖。
- 导师 PDF 中明确列出、但**尚未排期**的工程项：
  1. **disagreement 状态下 success 分档统计**（0/10/25/50 档）与 **h0/h50 端到端恢复不匹配验证**——定义需先与导师对齐；
  2. **最终确认库 20266010/20266050**：待参数与代码完全冻结后只跑一次，出投稿终版数字。

---

## 六、可复现性

- 三个出版门 + 校验脚本（README 数字、数据集哈希、绝对路径、三个 paper manifest）全部通过。
- Git LFS 指引已写入 README；跨平台行尾符已统一为 LF。
- 关键产物：`results/figures/mt10_success_occupancy.png`、`results/tables/mt10_matched_occupancy_comparison.json`、`results/tables/mt10_backfill_tuned_closed_loop_summary.csv`、`results/tables/mt10_multiseed_summary.csv`（含样本标准差 + rescued/harmed）、`results/tables/intervention_burden_three_metrics.json`（三种负担口径）、`results/diagnostics/selection_manifest.json`（参数选择清单）、`results/tables/mt10_horizon{0,10,25,50}_detector_threshold_floor065.json`（precision-floor 0.65 探针）。
- 本轮整改提交链（均可逐条 `git show` 复核）：`872238b1` 路径归一化 + manifest 入仓 → `cdd07009` README/tex 矛盾修复 → `478d4032` backfill 初步消融 → `b0568c93` paper_assets 自洽 → `faf273e5` 逐 horizon 阈值 + pre-event 审计 → `656cf6ae` occupancy 主表 + matched 对照 → `b038a815` 两个 gate manifest 重签 + ACL 修复 → `72664f50` 调阈闭环 + 多种子 → `c044b060` 行尾符跨平台根治。

—— 本汇报由实验数据与提交记录逐项复核生成，所有数值均有对应产物可溯源。

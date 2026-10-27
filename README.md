# SOP-UQ 学生项目

先读 [题面](docs/problem_statement.pdf)与[快速上手](docs/quickstart.pdf)。服务协议 **3.0**，主数据格式 **2.1**。

## 安装与入口

需要 conda（Miniconda、Miniforge 或 Anaconda）。在 `sop_uq_candidate/` 目录的 bash 或 zsh 中（Windows 请用 WSL）：

```bash
source conda_env.sh   # 首次按 environment.yml 创建 conda 环境 sii_uq（几分钟），以后只激活
export SOP_EXAMINER_URL=http://<考官地址>:8801   SOP_EXAMINER_TOKEN=<本队 token>
export SOP_MODEL_URL=http://<模型地址>:18080      SOP_MODEL_TOKEN=<模型 token>
jupyter notebook scripts/test_example.ipynb
```

不用 conda 时需要 Python 3.11 及以上（Debian/Ubuntu 另装 `python3-venv`）：先 `python3 -m venv .venv && source .venv/bin/activate`，再 `python -m pip install -e "./src[notebook]"`。

| 入口 | 内容 |
| --- | --- |
| `scripts/test_example.ipynb` | 测试 notebook：连接、手动调用工具与 oracle、baseline 作答、采集与拟合、测试运行、提交 |
| `scripts/baseline_example.ipynb` | 离线比较六种风险方法与转人工阈值；自行部署 Qwen |
| `scripts/run_test.py` | 检查并提交 `results.csv`（`--local` 在本地为有标签结果评分，`--submissions` 查看已用次数） |
| `sop-candidate` | 命令行：`collect`、`fit`、`run`、`score`、`submit` |
| `src/test_client/` | 考官与模型客户端、`results.csv` 格式、评分规则 |
| `src/baselines/` | agent、风险估计、查询/转人工策略、运行器；`resources/` 为各域 SOP 与工具 schema |
| `src/qwen_server/` | 可选的模型服务；`profiles/` 为精确模型配置，`runtime.*.json` 为部署模板 |

## 数据文件

每个 `.jsonl` 按 UTF-8 编码，每行一个 JSON 对象。三类业务在每个划分中数量相等。

| 文件（相对于 `data/`） | 行数 | 用途 |
| --- | ---: | --- |
| `train/train.jsonl` | 360 | 学习风险模型、历史错误率 |
| `train/cal_fit.jsonl` | 120 | 概率校准 |
| `train/cal_select.jsonl` | 60 | 选择查询/转人工阈值 |
| `train/dev.jsonl` | 120 | 练习、误差分析和调参 |
| `test/cases.jsonl` | 360 | 测试输入，无标签；其中 72 题允许交互 |
| `test/initial_M9.jsonl` | 360 | 考官冻结的初始轨迹：答案、自报风险、四次采样、工具历史与对话（所有队相同） |
| `manifest.json` | — | 数据批次、行数与 SHA-256 |

题目文件每题含 `schema_version`、`dataset_release_id`、`case_id`、`domain`、`inputs`；有标签划分另含 `training_split` 和 `label`。`test/initial_M9.jsonl` 每行一题：`answer`（冻结的初始答案，失败时为空）、`status`、`p_error`（自报风险）、`samples`（四次采样）、`history`（工具回执与模型输出）、`messages`（不含 system 消息的对话）和 `cost` 等。`dev` 中标准与偏移条件各半，测试同样；单题不标注条件。

| `domain` | 答案字段 |
| --- | --- |
| `dangerous_goods` | `hazard_class` |
| `aircraft_inspection` | `aircraft_ready`、`mechanical_inspection_result`、`electrical_inspection_result`、`component_incident_response`、`component_mismatch_response`、`cross_check_response`、`cross_check_reporting_response` |
| `customer_service` | `final_resolution_status` |

严格取值与业务边界见 `src/baselines/resources/business_rules.md`，工具参数见各域 `toolspecs.json`。

## 工具与 oracle

`examiner.episode(case_id)` 打开一题，`examiner.tool(episode_id, name, arguments, mode)` 调用业务工具，`examiner.oracle(episode_id, field, phase)` 读取真值。每题可用的工具及参数见 `episode["view"]["tools"]`：各域 `toolspecs.json` 中的工具，外加 SOP 要求的业务动作（见各域 `amendment.txt`）。参数取自题目输入和此前的回执，流程见各域 `sop.txt`。

| 业务 | 工具（有噪声的读数） | oracle 字段（phase） |
| --- | --- | --- |
| 危险品 | `calculate_sds_label_score`、`calculate_handling_score`、`calculate_transportation_score`、`calculate_disposal_score` 各给出一项评分，四项评分都有噪声 | `sds`、`handling`、`transportation`、`disposal`（`assessment`） |
| 飞机部件检查 | `VerifyAircraftClearance`、`VerifyMechanicalComponents`、`VerifyElectricalSystems`、`CrossCheckSpecifications` 的结果有噪声；`ReportComponentIncident`、`ReportComponentMismatch`、`ReportCrossCheck` 据此给出报告结论；`updateMaintenanceRecord` 记录维修记录 | `aircraft_ready`、`mechanical_inspection_result`、`electrical_inspection_result`、`cross_check_response`（`assessment`） |
| 客服排障 | 按 SOP 顺序：`validateAccount` → `getAuthenticationDetails` → `createSessionAndOpenTicket` → `checkAccountStatus`；账户停用时先走付款分支 `transferAccountsPayable` → `checkPaymentStatus` → `checkAccountSuspensionStatus`；再 `checkServiceAreaOutage` → `performTechnicalDiagnostics`（诊断）→ `executeTroubleshooting`（排障）；最后 `updateTicket` 记录结论或 `createEscalation` 升级。诊断和排障给出的 `latency`、`jitter`、`bandwidth` 有噪声 | 这三项：`pre_repair`（诊断后）或 `post_repair`（排障后） |

`mode` 决定读数来源：`default` 读主数据源，可能是缓存的旧记录；`refresh`、`remeasure` 重新读主数据源，同一来源的重复读数可能相关；`independent` 读第二个独立数据源。普通调用 0.01，有噪声工具的 `independent` 读数 0.03，oracle 每个字段 0.10，同一字段再读免费。每题预算 2.00、最多 24 次工具调用和 16 次 oracle；步骤顺序不对时返回 `phase_denied`。

有标签题可以不限次数地重新打开，每次都是新的练习，预算和调用次数重新计算。测试题中只有 72 道交互题可以继续调用工具和 oracle：每道测试题第一次成功的 oracle 回复带有考官记录的时间 `first_oracle_at`（这一次 `first_oracle` 为真），此后该题的 `p_initial_error` 不得再修改，由考官人工核查。

## 结果与提交

`results.csv` 每道测试题一行：`case_id, initial_answer_json, p_initial_error, mode, final_answer_json, p_initial_error_after_queries, p_final_error`。非交互题只填前三列；无答案或无概率时留空；初始答案必须是冻结答案。

每队最多 5 次有效提交，每次立即返回完整评分；格式错误或与已评分文件完全相同的文件不占次数，成绩取最好的一次。测试题的工具和 oracle 费用记在本队台账中，换方法或重跑不会清零，所以比较方法请在有标签题上进行。评分反馈只用于评估自己的方法，不得用来推断测试标签。

最快的完整流程：`sop-candidate collect --policy fixed` 一次采集有标签题的初始与查询后轨迹，`sop-candidate fit` 拟合两类风险，`sop-candidate run` 生成测试结果，`python scripts/run_test.py` 提交（见快速上手）。

# SOP-UQ 考生项目（服务协议 2.2；主数据 2.1）

本项目可独立安装，提供公开数据、Qwen 部署、考官服务 SDK、baseline 和结果包工具。
先读 [题目说明](docs/problem_statement.md)、[业务判定规则](public_specs/business_rules.md) 及各域 `public_specs/*/sop.txt`。
公开训练池标签可用于训练、校准、验证和调参；`data/test/cases.jsonl` 没有测试标签。所有工具与 oracle 都在考官服务运行。`dev` 的 120 例验证题允许反复核验，正式测试只开放原定 72 例 oracle。

## 安装与运行

Python 3.11+。在本目录运行：

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
# 仅部署 Qwen 的机器需要模型依赖；远程考生客户端不需要权重或 transformers。
# 推荐另建 Python 3.12.7 模型环境；匹配现有 CUDA 12.8 环境时先安装锁定版本：
pip install -r requirements-model.lock.txt
pip install -e '.[model]'
sop-qwen download --profile model_profiles/M9.json --destination /path/to/Qwen3.5-9B
# 将 examples/runtime.M9.json 中的 model_path 改为实际路径，另存 runtime.json。
export SOP_MODEL_TOKEN='replace-with-a-separate-model-token-at-least-24-characters'
sop-qwen verify --profile model_profiles/M9.json --runtime runtime.json
sop-qwen serve --profile model_profiles/M9.json --runtime runtime.json
```

考官可安装同一份 wheel 或项目运行上述服务，作为公用 Qwen 端点。HTTP 客户端仅需 URL 和独立模型 token；不检查远端机器的本地文件。
`/health` 提供可用性检查，`/v1/profile` 与 `/v1/generate` 需要 Bearer token。profile hash 用于配置一致性检查，不构成远程模型身份认证或完整推理次数证明。
只有服务明确表示尚未执行推理的容量拒绝，客户端才最多尝试三次。超时、未知服务错误或执行后的失败不会自动重试，避免重复生成或悄悄换一份答案。
M9 的公开锁对应 Qwen3.5-9B BF16；M27 锁对应指定 Qwen3.8-27B GGUF 和 llama.cpp b11541。以冻结评测配置允许的模型槽位为准；默认正式评测仅 M9。
模型依赖固定为 Torch 2.8.0、Transformers 5.19.0 等版本，完整版本见 `requirements-model.lock.txt`（Python 3.12.7 / CUDA 12.8）。基础客户端支持 Python 3.11+；锁文件中的 NumPy 2.5.3 需要 Python 3.12+。服务启动时校验权重、tokenizer/config 的文件大小与 SHA-256；CPU 单元测试不加载模型。

M27 使用 `examples/runtime.M27.json`。先安装锁定 llama.cpp；`sop-qwen download --profile model_profiles/M27.json --destination /path/to/M27` 同时下载 GGUF 与锁定 tokenizer/config。runtime 的 `model_path` 必须填写 `/path/to/M27/Qwen3.8-27B-UD-Q4_K_M.gguf`，`tokenizer_path` 填写 `/path/to/M27/tokenizer`。`launch_server=true` 由当前前台进程管理自己的 llama.cpp 子进程；设为 false 并指定 `upstream_url` 可连接本机已部署实例，仍核验其路径、权重、tokenizer/config 和 build。M27 revision 仅指 tokenizer/config 来源，量化发布者未证明其精确 BF16 源提交。

在另一个终端连接考官：

```bash
export SOP_EXAMINER_TOKEN='your-participant-token'
export SOP_MODEL_TOKEN='your-self-hosted-or-shared-model-token'
sop-candidate spec --examiner-url http://127.0.0.1:8766
# 1. 采集公开训练池的初始轨迹；其中的训练、校准和选择数据严格分开使用。
sop-candidate train --examiner-url http://127.0.0.1:8766 --model-url http://127.0.0.1:18080 \
  --profile model_profiles/M9.json --method-id initial --uq self_report --output runs/initial
sop-candidate fit --data data/train --snapshots runs/initial/M9-initial.jsonl \
  --selection-snapshots runs/initial/M9-initial.jsonl --output runs/initial-calibrator.json

# 2. 采集同一查询策略的公开后验轨迹，然后分别拟合、选择阈值。
sop-candidate train --examiner-url http://127.0.0.1:8766 --model-url http://127.0.0.1:18080 \
  --profile model_profiles/M9.json --method-id posterior-active --uq calibrated_disagreement \
  --calibrator runs/initial-calibrator.json --policy active --fixed-count 1 \
  --collect-posterior --output runs/posterior
sop-candidate fit --data data/train --snapshots runs/initial/M9-initial.jsonl \
  --selection-snapshots runs/initial/M9-initial.jsonl \
  --posterior-snapshots runs/posterior/M9-posterior.jsonl \
  --posterior-selection-snapshots runs/posterior/M9-posterior.jsonl \
  --output runs/final-calibrator.json

# 3. 正式服务须由考官预先加载固定初始轨迹。
sop-candidate run --examiner-url http://127.0.0.1:8766 --model-url http://127.0.0.1:18080 \
  --profile model_profiles/M9.json --method-id calibrated-active --uq calibrated_disagreement \
  --calibrator runs/final-calibrator.json --policy active --fixed-count 1 --output runs/test
sop-candidate submit --examiner-url http://127.0.0.1:8766 runs/test/submission.zip
```

同一条 run 命令和同一输出目录可恢复中断。这个静态 baseline 会核对本地配置以防误续跑；自定义在线方法或另一配置可以通过公开客户端操作同一考试会话，配置检查不构成服务端调参禁令。保留 `state.json`：它只含本方允许观察的记录、模型轨迹与回执，不含服务端私有状态。

正式运行使用考官固定的初始答案、证据和共享采样，不能重做或改写初始答案。第一次查询本题 oracle 前，runner 封存本题所有初始评分槽位的风险；不要求全试卷同时封存。允许跨题分析和在线调整。换方法、后续模型、输出目录或提交版本，均继续使用同一考生、试卷的证据与费用账本。

`run` 生成答案草稿和 `submission.zip`，不会自动提交或交卷。正式测试最多接收五个不同的有效完整版本，前四次可以继续修改和查询，第五次自动交卷。重复传输同一逻辑答案不占次数。`pack --run-id ... --output ...` 可根据当前服务端草稿重新打包。

```bash
# 上传当前版本；加 --close 可在成功接收后提前交卷。
sop-candidate submit --examiner-url http://127.0.0.1:8766 runs/test/submission.zip
# 已经上传有效版本后，也可单独交卷。
sop-candidate close --examiner-url http://127.0.0.1:8766 --run-id <run_id>

# dev 验证练习不占正式测试额度，可反复重开和核验。
sop-candidate practice --examiner-url http://127.0.0.1:8766 \
  --model-url http://127.0.0.1:18080 --profile model_profiles/M9.json \
  --method-id dev-practice --uq self_report --policy all_facts --threshold 0.5 \
  --output runs/dev
```

自定义方法可用 `sop-candidate rpc --examiner-url ... <operation> --body-file request.json` 调用公开的 `initial`、`final`、`checkpoint_initial` 和 `restart_dev` 等操作；请求文件不包含凭据。`--continuation-model-url` 可选择另一个后续求解服务，不改变官方初始评分槽位。已查询本题真值的事前风险永远固定，未核验题可以在后续版本中更新。

`withheld` 表示整场考试尚未公布分数，不是提交失败。公布后用 `sop-candidate report --examiner-url ... <evaluation_id>` 读取报告；此后不能修改或重新报名。验证练习允许即时评估，训练运行的保护配额不适用于独立 `dev` 练习入口。测试账单计入交卷时全部实际调用，即使某次查询发生在最后一次上传之后。

## Baseline 的确切含义

所附 baseline 使用同一个远程工具 agent，初始最多 20 次主调用、整个案例最多 32 次；有合法初始提案时额外使用恰好 4 次无工具共享采样。这些是参考与 baseline 的配置，考生可以自行选择后续求解方法。正常格式的辅助报告可保留在本地轨迹，评分答案只含主目标字段。

| UQ 方法 | 实现与训练来源 |
| --- | --- |
| `self_report` | 模型自报错误概率；缺失时按失败提交 |
| `disagreement` | 四次采样相对初始答案的不一致数，含非法样本，使用 `(d+1)/(4+2)` 平滑 |
| `historical_failure` | 公开 `train` 标签计算的分域平滑错误率 |
| `calibrated_self_report` / `calibrated_disagreement` | 公开 `cal_fit` 标签上的正则 logistic 概率校准 |
| `supervised` | 公开 `train` 标签上的 logistic 回归；特征仅为自报、采样不一致、缺失、已观察工具冲突与调用统计 |

静态校准器绑定数据版本、模型 profile、提示词、消息整理规则、调用上限、采样方式、风险估计及公共协议实现，还绑定实际预算和费用。影响轨迹或预测的配置变化需要重采或重拟合；旧产物不能未经校验混用。该检查不禁止自定义在线方法更新参数。工具回执在输入模型前统一压紧格式，并删除传输编号，保留业务值、时间、来源、观察编号和费用。原始回执仍保存在审计记录中。

查询后的两个风险有各自的校准模型，使用公开 `cal_fit` 查询轨迹拟合，不能用初始校准器冒充。实际追加查询前检查查询策略与后验校准器是否匹配；没有查询时沿用相应的事前估计，不要求不存在的后验样本。未经校准的 `self_report` 前后使用自报；可用 `--uq self_report --policy fixed --threshold 0.5` 运行启发式对照。

默认阈值来自公开 `cal_select` 按普通错误、严重错误和转人工的实际代价选择；`--threshold` 是显式覆盖，不能宣称最优。`diagnostics.json` 会报告风险值个数、查询比例和全部转人工等退化情况。无法产生合法答案的阶段按失败提交，不补造答案或概率。

| 控制器 | 查询及终局 |
| --- | --- |
| `no_oracle` | 无 oracle，AUTO |
| `all_defer` | 无 oracle，DEFER；仍提交影子答案 |
| `threshold` | 无 oracle，风险达到阈值则 DEFER |
| `fixed` | 按公开固定字段顺序查询至 `fixed-count`，续跑后按阈值决策 |
| `all_facts` | 查询当前合法且预算可承受的所有可核验字段，再续跑；不请求完整状态 |
| `active` | 初始风险达到阈值才查询；按已观察冲突、缺失、数值分散度排序选择至 `fixed-count`，续跑后按阈值决策 |

`active` 是公开启发式，不宣称估计最优查询的信息价值。工具和 oracle 费用由考官计量。自部署调用次数由本 runner 记录，是规范与申报，考官不能据此证明没有额外本地调用。

## 公开报告检查

```bash
sop-candidate check-public-report technical_report.md
```

该命令仅在本地检查文本中的凭据、敏感字段和明显测试核验对应表，不查询服务或私有标签，不返回疑似凭据的原文。发现问题或不能扫描的格式时退出码为 2；通过检查仍需人工复核。给考官的私有复现材料与公开报告分开，不能把测试真值或他人的材料放入公开报告。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用明确标识的 fake backend，只验证协议、HTTP 鉴权、远程客户端、校准与重试行为；不是 Qwen 推理效果实验。这些测试不代表模型运行成绩。

## 数据与复现边界

`data/` 是主数据；`challenge_data/` 是 72 例独立边界检查。两者使用不同的数据版本，不合并排名。题目是教学模拟，不提供真实危险品、适航或维修决策依据；各域声明随 `public_specs/` 及安装资源完整提供。

发布包由明确的 `PUBLIC_RELEASE_FILES.json` 列表生成。新增源码、说明或配置须先审查后加入列表；误放的凭证文件会使打包失败。发布包内部 manifest 记录每个文件的哈希。

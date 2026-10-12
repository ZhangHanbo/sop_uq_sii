# SOP-UQ 考生项目（服务协议 2.2，主数据格式 2.1）

本项目包含公开数据、Qwen 模型服务、考官服务客户端、baseline 方法和提交打包工具，可以单独安装。

先阅读[题面](docs/problem_statement.md)和[详细规则](docs/rules.md)，再看[业务规则](public_specs/business_rules.md)和各业务域的 `public_specs/*/sop.txt`。训练数据的标签可以用于训练、校准、验证和调参；`data/test/cases.jsonl` 不含标签。所有工具和 oracle 都在考官服务上运行。`dev` 的 120 道题可以不限次数地调用 oracle；正式测试只有 72 道交互题可以调用 oracle。

## 安装与部署模型

需要 Python 3.11 及以上。在本目录运行：

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
# 只有部署 Qwen 的机器需要安装模型依赖；只连接远程服务的考生客户端不需要权重或 transformers。
# 建议另建一个 Python 3.12.7 环境部署模型；如果使用 CUDA 12.8，先安装锁定的版本：
pip install -r requirements-model.lock.txt
pip install -e '.[model]'
sop-qwen download --profile model_profiles/M9.json --destination /path/to/Qwen3.5-9B
# 把 examples/runtime.M9.json 中的 model_path 改成实际路径，另存为 runtime.json。
export SOP_MODEL_TOKEN='replace-with-a-separate-model-token-at-least-24-characters'
sop-qwen verify --profile model_profiles/M9.json --runtime runtime.json
sop-qwen serve --profile model_profiles/M9.json --runtime runtime.json
```

考官也可以用同一套代码部署公共 Qwen 服务。客户端只需要服务地址和模型服务的 token，不检查远端机器上的文件。

- `/health` 用于检查服务是否可用；`/v1/profile` 和 `/v1/generate` 需要 Bearer token。
- profile hash 只用于检查配置是否一致，不能证明远端运行的确实是这个模型，也不能证明调用次数。
- 只有当服务明确表示请求还没开始推理就因容量不足被拒绝时，客户端才会重试，最多尝试三次。超时、未知错误或推理之后的失败都不重试，以免重复生成或换成另一份答案。
- 模型依赖版本固定为 Torch 2.8.0、Transformers 5.19.0 等，完整列表见 `requirements-model.lock.txt`（Python 3.12.7，CUDA 12.8）。客户端支持 Python 3.11 及以上；锁定的 NumPy 2.5.3 需要 Python 3.12 及以上。
- 服务启动时校验权重、tokenizer 和 config 文件的大小与 SHA-256。CPU 单元测试不加载模型。

M9 对应 Qwen3.5-9B BF16；M27 对应指定的 Qwen3.8-27B GGUF 和 llama.cpp b11541。可用的模型以考官的评测配置为准，正式考试默认只用 M9。

使用 M27 时：

1. 安装锁定版本的 llama.cpp。
2. 运行 `sop-qwen download --profile model_profiles/M27.json --destination /path/to/M27`，同时下载 GGUF 和锁定的 tokenizer/config。
3. 以 `examples/runtime.M27.json` 为模板，`model_path` 填 `/path/to/M27/Qwen3.8-27B-UD-Q4_K_M.gguf`，`tokenizer_path` 填 `/path/to/M27/tokenizer`。
4. `launch_server=true` 时由当前进程启动并管理 llama.cpp；设为 false 并指定 `upstream_url` 时连接本机已运行的 llama.cpp，启动时仍会核对模型路径、权重、tokenizer/config 和版本号。

M27 的 revision 只指 tokenizer/config 的来源；量化版本的发布者没有说明它对应的 BF16 权重的确切版本。

## 训练、正式测试与提交

在另一个终端连接考官服务：

```bash
export SOP_EXAMINER_TOKEN='your-participant-token'
export SOP_MODEL_TOKEN='your-self-hosted-or-shared-model-token'
sop-candidate spec --examiner-url http://127.0.0.1:8766

# 1. 在训练集上采集初始轨迹，然后拟合风险模型。train、cal_fit、cal_select 分开使用。
sop-candidate train --examiner-url http://127.0.0.1:8766 --model-url http://127.0.0.1:18080 \
  --profile model_profiles/M9.json --method-id initial --uq self_report --output runs/initial
sop-candidate fit --data data/train --snapshots runs/initial/M9-initial.jsonl \
  --selection-snapshots runs/initial/M9-initial.jsonl --output runs/initial-calibrator.json

# 2. 用同一个查询策略在训练集上采集查询后的轨迹，再拟合查询后的风险模型并选阈值。
sop-candidate train --examiner-url http://127.0.0.1:8766 --model-url http://127.0.0.1:18080 \
  --profile model_profiles/M9.json --method-id posterior-active --uq calibrated_disagreement \
  --calibrator runs/initial-calibrator.json --policy active --fixed-count 1 \
  --collect-posterior --output runs/posterior
sop-candidate fit --data data/train --snapshots runs/initial/M9-initial.jsonl \
  --selection-snapshots runs/initial/M9-initial.jsonl \
  --posterior-snapshots runs/posterior/M9-posterior.jsonl \
  --posterior-selection-snapshots runs/posterior/M9-posterior.jsonl \
  --output runs/final-calibrator.json

# 3. 正式测试。考官需要事先装好固定的初始轨迹。
sop-candidate run --examiner-url http://127.0.0.1:8766 --model-url http://127.0.0.1:18080 \
  --profile model_profiles/M9.json --method-id calibrated-active --uq calibrated_disagreement \
  --calibrator runs/final-calibrator.json --policy active --fixed-count 1 --output runs/test
sop-candidate submit --examiner-url http://127.0.0.1:8766 runs/test/submission.zip
```

`run` 中断后，用同一条命令和同一个输出目录重新运行即可继续。请保留输出目录中的 `state.json`，其中只有你能看到的记录、模型轨迹和回执，不含服务端的私有数据。runner 会检查本地配置是否与之前一致，防止把两次不同的运行混在一起；这只是 baseline 的本地检查，自己的方法可以随时调整，并通过同一个考试会话继续操作。

正式测试使用考官固定的初始答案、调用记录和共享采样，不能重新生成或修改初始答案。runner 在某道题第一次调用 oracle 之前，会先锁定这道题所有初始模型的风险；不需要一次锁定全部题目。可以利用其他题目的信息，也可以在考试中更新方法。更换方法、后续模型、输出目录或提交版本，都继续使用同一个考生、同一张试卷的调用记录和费用。

`run` 只生成答案草稿和 `submission.zip`，不会自动提交或交卷。正式测试最多接收 5 个不同的版本：前 4 个版本提交后还可以继续修改和查询，第 5 个版本被接收后自动交卷。重复上传同一份答案不计次数。`pack --run-id ... --output ...` 可以按服务端当前的草稿重新打包。

```bash
# 提交当前版本；加 --close 表示提交成功后立即交卷。
sop-candidate submit --examiner-url http://127.0.0.1:8766 runs/test/submission.zip
# 已提交过至少一个版本后，也可以单独交卷。
sop-candidate close --examiner-url http://127.0.0.1:8766 --run-id <run_id>
```

正式测试期间，提交后返回 `withheld`，表示已接收但还没有公布成绩，不是提交失败。成绩公布后，用 `sop-candidate report --examiner-url ... <evaluation_id>` 读取报告；公布后不能再修改或重新报名。测试费用按交卷时的全部调用计算，包括最后一次提交之后的调用。

## dev 练习

```bash
# dev 练习不占用正式测试的次数和额度，可以反复进行。
sop-candidate practice --examiner-url http://127.0.0.1:8766 \
  --model-url http://127.0.0.1:18080 --profile model_profiles/M9.json \
  --method-id dev-practice --uq self_report --policy all_facts --threshold 0.5 \
  --output runs/dev
sop-candidate submit --examiner-url http://127.0.0.1:8766 runs/dev/submission.zip
```

练习时 oracle 不限次数、不受预算限制，也不需要先锁定风险，提交后立即返回成绩。每道题的普通工具仍限 24 次。要重新开始，可以换一个 `--method-id`，或者调用 `restart_dev`。训练运行的次数上限（见 `spec` 返回的 `max_train_runs` 和 `max_train_evaluations`）不适用于 dev 练习。

## 自己实现方法

可以用 `sop-candidate rpc --examiner-url ... <operation> --body-file request.json` 直接调用服务接口，例如 `initial`、`final`、`checkpoint_initial` 和 `restart_dev`。请求文件中不要写凭据。`--continuation-model-url` 可以指定另一个模型服务来继续作答，不影响考官固定的初始答案。调用过 oracle 的题，查询前的风险不能再改；没调用过的题可以在后续版本中修改。

## Baseline 说明

所附 baseline 使用同一个调用工具的 agent：初始阶段最多调用模型 20 次，整道题最多 32 次；给出合法初始答案后，再额外采样 4 次（不调用工具）。这些是参考 agent 和 baseline 的设置，你可以用别的方法继续作答。模型输出中多余的报告字段会保留在本地轨迹里，评分只看主评分字段。

| 风险估计方法 | 说明 |
| --- | --- |
| `self_report` | 模型自报的出错概率；缺失时按失败提交 |
| `disagreement` | 记 4 次采样中与初始答案不一致（含不合法）的次数为 d，取 `(d+1)/(4+2)` |
| `historical_failure` | 按业务域计算的 `train` 错误率（加平滑） |
| `calibrated_self_report` / `calibrated_disagreement` | 在 `cal_fit` 上用带正则的 logistic 回归做概率校准 |
| `supervised` | 在 `train` 上训练的 logistic 回归；特征只有自报概率、采样不一致率、是否缺失、已观测到的工具读数冲突和调用次数 |

`fit` 生成的校准文件记录了生成训练轨迹时的数据版本、模型配置、提示词、消息整理方式、调用上限、采样方式、风险估计代码、公共协议代码、预算和费用。影响轨迹或预测的任何一项变化后，需要重新采集轨迹或重新拟合；程序会拒绝不匹配的旧校准文件。这个检查不限制你自己的在线方法更新参数。工具回执在交给模型之前会统一格式，并删掉传输用的编号，保留业务值、时间、数据源、记录编号和费用；原始回执保存在本地记录中。

查询后的两个风险各有一个校准模型（保存在同一个校准文件中），用训练集上 `cal_fit` 部分的查询轨迹拟合，不能用查询前的校准模型代替。只有在实际要查询时，程序才检查查询策略和校准文件是否匹配；某题没有查询时，直接沿用查询前的风险。不经校准的 `self_report` 在查询前后都用模型自报的概率；可以用 `--uq self_report --policy fixed --threshold 0.5` 运行这个启发式对照。

默认阈值在 `cal_select` 上按普通错误、严重错误和转人工的实际代价选取；`--threshold` 用于手动指定，不能声称是最优值。`diagnostics.json` 会报告不同风险值的个数、查询比例、是否全部转人工等情况。无法给出合法答案时按失败提交，不会编造答案或概率。

| 策略 | 查询与决策 |
| --- | --- |
| `no_oracle` | 不查询，全部自动处理 |
| `all_defer` | 不查询，全部转人工；仍然提交答案 |
| `threshold` | 不查询，风险不低于阈值时转人工 |
| `fixed` | 按固定字段顺序查询 `fixed-count` 个字段，继续作答后按阈值决策 |
| `all_facts` | 查询当前所有可查、且预算允许的字段，再继续作答 |
| `active` | 初始风险不低于阈值时才查询；按已观测读数的冲突、缺失和数值离散程度排序，查询前 `fixed-count` 个字段，继续作答后按阈值决策 |

`active` 是启发式方法，不保证查询的信息价值最优。工具和 oracle 的费用由考官服务统计；自己部署模型时的调用次数由 runner 记录，只作为申报，考官无法据此证明没有额外的本地调用。

## 公开报告检查

```bash
sop-candidate check-public-report technical_report.md
```

这个命令只在本地检查文本中是否有凭据、敏感字段，以及明显的"测试题编号对应 oracle 返回值"的表格；不连接服务，也不使用任何隐藏标签，输出中不会重复疑似凭据的原文。发现问题或遇到无法检查的文件格式时，退出码为 2。检查通过后仍需要人工复核。给考官的复现材料与公开报告分开，公开报告中不能出现测试题的真实值或其他考生的材料。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用明确标注的假模型，只检查协议、HTTP 鉴权、客户端、校准和重试逻辑，不代表 Qwen 的推理效果。

## 数据与发布

`data/` 是主数据；`challenge_data/` 是 72 个独立的边界案例。两者的数据版本不同，成绩不合并。本题是教学模拟，结果不能用于真实的危险品、适航或维修决策；各业务域的声明随 `public_specs/` 和安装资源一起提供。

发布包按 `PUBLIC_RELEASE_FILES.json` 中列出的文件生成。新增源码、说明或配置时，需要先审查再加入列表；如果目录中混入了未列出的文件（例如凭据文件），打包会失败。发布包中的 manifest 记录了每个文件的哈希。

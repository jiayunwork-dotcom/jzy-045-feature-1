# Enzyme Kinetics HTTP Service

长期在线的酶促反应速率计算端点（Flask / Python 3.12）。只提供酶–底物层的
速率与竞争性抑制换算，以 JSON over HTTP 对外服务，不含网页界面、登录或订单等外围功能。

## 动力学模型

无抑制（Michaelis–Menten）：

```
v = Vmax · [S] / ([S] + Km)
```

竞争性抑制：

```
v       = Vmax · [S] / ([S] + Km · (1 + [I]/Ki))
Km_app  = Km · (1 + [I]/Ki)
Vmax_app = Vmax          # 最大速率不变
```

Lineweaver–Burk 不变量（端点返回 `lineweaver_burk.slope/y_intercept` 供交叉校验）：

```
1/v = (Km_app/Vmax) · (1/[S]) + 1/Vmax
```

竞争性抑制下纵截距 `1/Vmax` 恒等不变，斜率 `Km_app/Vmax` 随 `[I]` 单调增大。
抑制类型只接受 `"competitive"`；非竞争性/混合型一律 400 拒绝，不会被悄悄重新解释。
`Ki` 极大时 `[I]/Ki → 0`，结果连续退回普通米氏速率。

预置示范酶 **hexokinase**：`Vmax=100 μmol/min/mg`、`Km=0.1 mM`，
可手算复核：`[S]=0.1` 时 `v=50`；`[I]=Ki=0.2` 时 `Km_app=0.2`，在 `[S]=0.2` 处仍为 `v=50`。

## 目录结构

```
app/
  __init__.py       # Flask 应用工厂（错误处理器、蓝图装配）
  kinetics.py       # 正向：米氏速率核心（纯函数、无状态）
  fitting.py        # 反向：由 ([S], v) 散点拟合 Vmax/Km（纯函数、多起点 LM）
  inhibition.py     # 竞争性抑制换算（Km_app / Vmax_app / LB 系数）
  validation.py     # 入参校验（非法/矛盾输入、未知字段拒绝）
  enzyme_store.py   # 酶参数档持久化（JSON + fcntl 跨进程锁 + 原子替换）
  routes.py         # HTTP 路由
  errors.py         # 统一 JSON 错误响应
wsgi.py             # gunicorn 入口（wsgi:app）
tests/              # 192 条自动化测试（含拟合收敛/不收敛/合成真值回归护栏）
Dockerfile          # python:3.12-slim，非 root 用户，/data 卷，HEALTHCHECK
docker-compose.yml
```

正反两个方向是**独立模块**：`kinetics.py` 只做"常数 → 速率"，
`fitting.py` 只做"散点 → 常数"，各自无状态、可单独演进与测试，
正向代码完全不依赖拟合模块（反之亦然）。拟合模块只用标准库 `math`，
不引入任何重量级数值库：多起点 **Levenberg–Marquardt**（对数参数空间，
天然保证 Vmax/Km 为正）、收敛判据（步长/残差下降/梯度最优性三套检验）
与初始化策略（Lineweaver–Burk、Eadie–Hofstee 双倒数线性化 + Km 一维
剖面扫描）均在模块内实现并被测试覆盖。

并发隔离：计算核心是无状态纯函数；参数档写入经线程锁 + `fcntl` 文件锁
（gunicorn 多 worker 也安全），临时文件 `fsync` 后 `os.replace` 原子落盘，
存储文件路径由环境变量 `ENZYME_STORE_PATH` 决定（容器内为 `/data/enzymes.json`）。

## 容器构建与启动

```bash
docker build -t enzyme-kinetics-api:1.0.0 .
docker run -d -p 8000:8000 -v enzyme-data:/data enzyme-kinetics-api:1.0.0
# 或
docker compose up -d --build
```

健康检查：`GET http://localhost:8000/health` → `{"status":"ok"}`

本地直接运行（需 Python 3.12）：

```bash
pip install -r requirements.txt
gunicorn --bind=0.0.0.0:8000 --workers=2 --threads=4 wsgi:app
# 数据文件默认落在 ./data/enzymes.json，首次启动自动创建并种子化
```

## 接口

### 1. 无抑制速率 `POST /v1/rate`

内联常数或点名已登记酶（二选一）：

```bash
curl -XPOST localhost:8000/v1/rate -H 'Content-Type: application/json' \
  -d '{"vmax":100,"km":0.1,"substrate":0.1}'
# {"rate":50.0,"saturation_fraction":0.5,"vmax":100.0,"km":0.1,
#  "substrate":0.1,"lineweaver_burk":{"slope":0.001,"y_intercept":0.01}}

curl -XPOST localhost:8000/v1/rate -H 'Content-Type: application/json' \
  -d '{"enzyme":"hexokinase","substrate":0.1}'
```

### 2. 竞争性抑制速率与表观常数 `POST /v1/rate/inhibited`

```bash
curl -XPOST localhost:8000/v1/rate/inhibited -H 'Content-Type: application/json' \
  -d '{"enzyme":"hexokinase","substrate":0.2,"inhibitor":0.2,"ki":0.2,
       "inhibition_type":"competitive"}'
# {"rate":50.0,"alpha":2.0,"apparent":{"km":0.2,"vmax":100.0}, ...}
```

### 3. 反推动力学常数（非线性最小二乘拟合）`POST /v1/fit`

提交若干 `(substrate, observed_rate)` 实测对，端点在无抑制米氏方程下
最小化残差平方和，反推最可能的 `Vmax` 与 `Km`，并返回拟合优度、参数
标准误与优化过程信息：

```bash
curl -XPOST localhost:8000/v1/fit -H 'Content-Type: application/json' \
  -d '{"measurements":[
        {"substrate":0.02,"observed_rate":16.7},
        {"substrate":0.05,"observed_rate":33.3},
        {"substrate":0.1, "observed_rate":50.0},
        {"substrate":0.2, "observed_rate":66.7},
        {"substrate":0.5, "observed_rate":83.3},
        {"substrate":1.0, "observed_rate":90.9},
        {"substrate":2.0, "observed_rate":95.2}]}'
# {"converged":true,"vmax":99.97,"km":0.09992,"n_points":7,
#  "goodness_of_fit":{"r_squared":0.99999,"adjusted_r_squared":...,
#                     "residual_sum_squares":...,"rmse":...,
#                     "residual_standard_error":...},
#  "standard_errors":{"vmax":...,"km":...,"vmax_relative":...,"km_relative":...},
#  "optimization":{"iterations":7,"starts_tried":4,"initial_vmax":...,"initial_km":...}}
```

点名已登记的酶时，额外返回拟合值与参数档登记值的相对偏离（fraction /
percent），方便判断新数据是否还符合该酶原登记的动力学特征；不点名就
**只**返回拟合结果本身：

```bash
curl -XPOST localhost:8000/v1/fit -H 'Content-Type: application/json' \
  -d '{"enzyme":"hexokinase","measurements":[ ... ]}'
# ...,"profile_comparison":{"enzyme":"hexokinase",
#     "reference_vmax":100.0,"reference_km":0.1,
#     "vmax_deviation_fraction":-0.0003,"km_deviation_fraction":-0.0008,
#     "vmax_deviation_percent":-0.03,"km_deviation_percent":-0.08}}
```

拟合不可信时不会硬吐数字：少于 3 个点、底物全部相同、速率零方差、
迭代预算内不收敛、参数相对标准误过大（底物挤在同一数量级的典型症状）、
或存在与曲线形态明显矛盾的离群点，均以 `422` 明确拒绝（错误码见下）。

### 4. 酶参数档管理（落地保存、重启后保留）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/enzymes` | 列出全部参数档名称 |
| GET | `/v1/enzymes/<name>` | 取单个参数档 |
| PUT | `/v1/enzymes/<name>` | 登记/更新（201 新建 / 200 更新） |
| DELETE | `/v1/enzymes/<name>` | 删除（204） |

```bash
curl -XPUT localhost:8000/v1/enzymes/catalase -H 'Content-Type: application/json' \
  -d '{"vmax":55,"km":0.4,"description":"catalase profile"}'
```

## 非法输入处理（均为 `400`，带 `error` + `reason`）

* `vmax ≤ 0`、`km ≤ 0`、`substrate < 0` → 拒绝（`substrate = 0` 合法，速率精确为 0）
* `ki ≤ 0`、`inhibitor < 0` → 与竞争性语义矛盾，拒绝而非猜测
* `inhibition_type` 非 `"competitive"`（含 non-competitive / mixed）→ 拒绝
* 缺失字段、非数字、NaN/Infinity、布尔值、未知字段、非 JSON 对象 → 拒绝
* `/v1/fit` 测量记录的 `substrate`/`observed_rate` 为负数、非数字、非有限值、
  布尔，或记录不是对象/含未知字段 → 拒绝（reason 带 `measurements[i]` 索引）
* 点名不存在的酶 → `404 enzyme_not_found`

## 拟合拒绝处理（`/v1/fit`，`422`，带机器可读 `error` + `reason`）

| error | 触发条件 |
| --- | --- |
| `insufficient_data` | 测量点少于 3 个，无法约束 Vmax、Km 两个自由参数 |
| `fit_unidentifiable` | 所有底物浓度相同（单点重复）/ 速率全为零 / 速率零方差 / 协方差奇异 |
| `fit_not_converged` | 多起点 LM 在迭代上限（默认 100）内均未满足任何收敛判据，不返回中间结果 |
| `fit_unreliable` | 参数相对标准误过大（底物范围太窄）、存在外学生化残差超阈值的离群点、R² 过低，或最优解跑向 Km→0（数据形态与米氏曲线矛盾） |

> 400 = 请求形态/数值非法；422 = 形态合法但数据撑不出可信解。两种情况
> 都不会返回可能被误用的参数数字。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

关键回归护栏：`[S]=Km ⇒ v=Vmax/2`；竞争性抑制下 `Vmax_app=Vmax` 与 LB 纵截距
恒定；半饱和点随 `[I]` 右移；高底物下速率收敛到同一 `Vmax`；Vmax 加倍速率处处
加倍；`Ki→∞` 连续退回米氏速率；真实 HTTP 并发互不串值、参数档重启后仍可取用。
**拟合方向**：由"已知真值 Vmax/Km + 受控高斯噪声"合成数据，25 个随机种子下
反推参数必须落在随噪声收紧的容差内（`tests/test_fitting.py` 的
`test_recovery_tightens_as_noise_shrinks`）；离群点、窄底物范围、重复底物、
点数不足、LM 迭代预算耗尽（`max_iterations=0/1` 确定性不收敛路径）均有
拒绝护栏。

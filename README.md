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
  fitting.py        # 反向：由 ([S], v) 散点非线性最小二乘拟合 Vmax/Km
                    #       （手写多起点 Levenberg–Marquardt，显式收敛判据）
  inhibition.py     # 竞争性抑制换算（Km_app / Vmax_app / LB 系数）
  validation.py     # 入参校验（非法/矛盾输入、未知字段拒绝）
  enzyme_store.py   # 酶参数档持久化（JSON + fcntl 跨进程锁 + 原子替换）
  routes.py         # HTTP 路由
  errors.py         # 统一 JSON 错误响应
wsgi.py             # gunicorn 入口（wsgi:app）
tests/              # 179 条自动化测试（不变量/非法输入/并发/重启持久化/拟合回归护栏）
Dockerfile          # python:3.12-slim，非 root 用户，/data 卷，HEALTHCHECK
docker-compose.yml
```

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

### 3. 由实测散点反推动力学常数 `POST /v1/rate/fit`

反向端点：提交若干 `(substrate, observed_rate)` 测量对，在无抑制米氏方程
假设下做**非线性最小二乘**（米氏方程对 Vmax 线性、对 Km 非线性，不能套线性
回归），求残差平方和最小的 `Vmax`/`Km`。求解器是从零实现的多起点
Levenberg–Marquardt（对数参数空间阻尼 Gauss–Newton，依赖清单之外**不引入**
任何数值库），收敛判据（步长 / 目标函数相对下降 / 平稳梯度三者满足其一）
与迭代上限均显式给出，没收敛绝不吐出中间结果。

```bash
curl -XPOST localhost:8000/v1/rate/fit -H 'Content-Type: application/json' \
  -d '{"measurements":[
        {"substrate":0.01,"observed_rate":9.0},
        {"substrate":0.1,"observed_rate":50.3},
        {"substrate":1.0,"observed_rate":90.7}],
       "max_iterations":100}'
# {"vmax":...,"km":...,"converged":true,"iterations":4,"n_points":10,
#  "reliable":true,"well_identified":true,"warnings":[],
#  "goodness_of_fit":{"r_squared":0.9997,"sse":...,"rmse":...,
#                     "residual_standard_error":...,"max_abs_residual":...,
#                     "identifiability_collinearity":0.82},
#  "initial_guess":{"vmax":...,"km":...}}
```

拟合质量：`goodness_of_fit` 给出决定系数 `r_squared`、残差平方和 `sse`、
`rmse`、回归残差标准误、最大绝对残差，以及最优处 Vmax/Km 两列 Jacobian 的
相关性 `identifiability_collinearity`（越接近 1 越不可辨识）。`reliable`
综合可辨识性、R²、是否采到饱和段、有无 3σ 离群点给出总体可信度；`warnings`
逐条说明 `narrow_substrate_range` / `no_saturation_observed` /
`weak_parameter_identifiability` / `outlier_residual` / `poor_fit`。

点名已登记酶时附加 `comparison`（不点名绝不对任何对象做对照）：

```bash
curl -XPOST localhost:8000/v1/rate/fit -H 'Content-Type: application/json' \
  -d '{"enzyme":"hexokinase","measurements":[...]}'
# 响应额外含: "enzyme":"hexokinase",
#   "comparison":{"registered":{"vmax":100.0,"km":0.1},
#                 "fitted":{"vmax":...,"km":...},
#                 "absolute_deviation":{"vmax":...,"km":...},
#                 "relative_deviation":{"vmax":...,"km":...}}
```

不可信情形如实拒绝（均带 `error` + `reason`，不硬吐数字）：

* `422 insufficient_data`：测量点少于 3 个，不足以约束两个自由参数；
* `422 fit_unidentifiable`：所有点底物浓度相同、速率全为零或全相同、或
  数据全部挤在低 [S] 线性段导致 Vmax/Km 的 Jacobian 列近似共线；
* `422 fit_did_not_converge`：所有初始起点在迭代上限内均未收敛；
* `400 validation_error`：缺字段、非数字、NaN/Infinity、布尔、负数、
  未知字段、非 JSON 对象、`max_iterations` 越界等；
* `404 enzyme_not_found`：点名的酶未登记。

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
* 点名不存在的酶 → `404 enzyme_not_found`

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

关键回归护栏：`[S]=Km ⇒ v=Vmax/2`；竞争性抑制下 `Vmax_app=Vmax` 与 LB 纵截距
恒定；半饱和点随 `[I]` 右移；高底物下速率收敛到同一 `Vmax`；Vmax 加倍速率处处
加倍；`Ki→∞` 连续退回米氏速率；真实 HTTP 并发互不串值、参数档重启后仍可取用。

拟合方向（`tests/test_fitting.py`）的回归护栏：用"已知真实 Vmax/Km + 不同量级
高斯噪声"的合成数据反推，噪声越小回收越准（多组种子/多噪声量级参数化）；
精确数据 R²=1 且残差为零；`max_iterations=1` 必走非收敛路径并抛
`FitConvergenceError`；少点、同底物重复、全零/恒定速率、深线性段共线全部拒绝；
脏数据（负/NaN/Infinity/非数字/布尔）在拟合前挡回；跨数量级单位缩放不变性。

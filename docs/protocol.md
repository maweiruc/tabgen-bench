# Benchmark and Evaluation Protocol — 1.0.0

`standard_v1` uses Adult 1590 v2, Bank Marketing 1461 v1 and Credit-G 31 v1.
Each generator uses the same stratified 60/20/20 real split, seed 20261025.
Replicate seeds are [11, 23, 37, 53, 71], and synthetic row count equals training row count.
Bootstrap, Independent Marginal, Gaussian Copula and CPU CTGAN (100 epochs) make 60 cells.
The source/content/schema locks and suite hashes are normative; they are checked before use.
The calibration plugin is separate from the 16 main metrics, for 17 versioned results.

## 执行与审计

EvaluationEngine 仅导入 Core，接受显式 MetricRegistry；接口绑定在 interfaces/api.py。
Core EvaluationPolicy 描述固定前置项和通用 ResultGate。默认总是先执行三项 Validity，
schema_pass 非成功或 value!=1 时，其他读取 synthetic 的插件返回 not_applicable，
reason=blocked_by_validity。依赖源数据的 TRTR/calibration 独立执行。未知 metric ID、
重复请求是配置错误；已注册插件异常仅令自身 failed，其余结果保留。

插件只得到 required_inputs + 明确 optional_inputs、声明的 capabilities；不会获得全部表。
缺少 required input/capability 返回 NA/null，可选参考选择在插件内部显式处理。每项结果
记录 run/dataset/metric/domain/version、实际 reference split、参数 hash、schema/rule hash、
seed、可用输入身份、起止时间和诊断 details。failed/NA 没有分数。Engine 对插件结果的
ID/domain/version/direction 做一致性检查，不接受错标结果。缓存与运行 manifest 保留计算身份。

## Validity

结构错误：missing/extra/duplicate/order、empty_table、unexpected_row_count、unsupported_storage。
无法进行逐格检查时 schema_pass=0、两率 NA。可处理但类型非法的标量（包括 Inf）进入
逐格 type_mismatch，不被默默转换。NA/None/NaN 是缺失；字符串 nan/unknown 是普通类别。

numeric/integer 接受有限实数但排除 bool；integer 允许 35.0 等无损数值存储，不接受字符串
35 或非整数。boolean 接受 bool/numpy.bool_；categorical 接受有限数值、布尔、字符串，
不把 dtype 当语义。不支持的容器/日期对象属于结构问题。nullable=false 的缺失仅记录
null_not_allowed；非缺失同时检查整数性、明示边界、明示 allowed_values，可一格多原因。

单元格只计一次 invalid，行也只计一次；分母含目标列。逐列 reason 同时记录 count/rate、
rule_version/constraint_source。重复事件行合法。合法域仅使用冻结规则，不读取 train 最值
或类别全集。Context 和所有算法保持原始表；不裁剪、不重新排列、不删行或修复。

## Fidelity

真实比较参考优先明确 real_test，否则 real_reference；不因 test 为空而隐式替换参考。
KS 对 numeric/integer 非缺失样本用 scipy 二样本 statistic，按适用列等权均值，不用 p-value。
TV 是 0.5 * sum(abs(p-q))，含显式 NA 类别；字符串/布尔/数值 token 区分，数值 1/1.0
按无损数值表示等价。Missingness 是各列 NA 比例绝对差的等权均值。没有适用列时 NA。

Pairwise 使用 train 每列 0..1 的 11 个等距分位点，去重后最多 10 个 bins；内边界右侧归箱，
边界外分别 UNDERFLOW/OVERFLOW，缺失独立 NA；全缺失训练列使用 UNOBSERVED_NUMERIC。
类别 train 频数前 20，平局按规范 token 排序，其余 OTHER，缺失独立。按 schema 顺序选前
45 对列，计算二维频率 TV 再均值；记录实际 pairs/bins/fit input。完整模式只拟合 train；
reference-only 模式可拟合明确标注 ordinary reference 的表，记录该来源。只有 held-out test
没有 train/ordinary reference 时 Pairwise 返回 NA，不偷用 test 拟合分箱。

C2ST 从 test/reference 与 synthetic 各无放回选 min(n_real,n_syn) 行；n<4 返回 NA。固定 seed
对合并的等容量样本以 test_size=0.3 分层划分判别 fit/eval。所有列（含联合生成的目标）
参加 joint C2ST。ColumnTransformer 的 imputer/scaler/encoder 仅 fit 判别训练折；数值用
median + missing indicator（保留全缺失列）、StandardScaler；类别显式 typed tokens +
OneHotEncoder(handle_unknown=ignore)。RandomForestClassifier(n_estimators=200,
random_state=seed,n_jobs=1)，仅 held-out 概率的 AUC，不翻转小于 0.5 的分数。

独立校准插件用 validation/test 的相同等量/70–30/RF 协议，要求相同 source_id、不同
split_id。缺独立参考时 NA；不伪造 holdout，不参与生成拟合。身份来自调用者，此身份声明不
证明患者/个体独立性或检验任意外部数据的行位置互斥。记录抽样和判别 fit/eval 位置。

## Utility

要求显式二分类 TargetSpec；标签按正/负 token 映射为 1/0，目标不进入特征。缺失/未知
标签、单类、train<4/test<2 返回 NA；不会删标签行或换模型。TRTR fit real_train，TSTR fit
synthetic，二者 predict 同一 real_test。预处理相同但分别仅 fit 各自训练表，与 C2ST 使用
相同数值/类别变换。LogisticRegression(max_iter=1000,solver=lbfgs,random_state=seed)，
收敛错误令该项 failed。未知测试类别保持 ignore，记录数量，不重拟合编码器。

Signed gap = TSTR - TRTR，保留符号。结果附两个原值/模型/特征/样本数。各插件单独计算，
插件不缓存拟合；相同 seed/参数保证两项和 gap 使用同构过程。

## Privacy-Proximity diagnostics

DCR：n_ref=min(n_train,n_test,1024)，synthetic 最多 512 行，无放回抽样，独立局部 RNG，
同次评价的四个 DCR 插件用相同 seed 和位置。数值距离 abs(x-y)/range_train，range 只来自
完整 train；range=0 时相等为 0、不等为 1；类别不等=1。双方缺失=0、单方缺失=1。
按所有列（含目标）等权平均，不裁剪超出训练范围的距离。

最近邻只累计最多 64×128 的二维块，不建立 n_syn×n_train×p tensor。
train_closer_rate 使用严格小于，tie_rate 使用浮点结果精确相等，不引入 epsilon。
记录每侧容量、抽样位置/seed、train-derived ranges、tie rate、距离中位数和块大小。

Exact Copy 对完整 train 的规范 typed 行 token 做 SHA256 membership，对完整 synthetic
计比例，不套用 512/1024 子采样。正确识别 NA、重复行、无损整数/浮点存储；字符串 nan
不同于缺失。两种生成器均为诊断对照：Bootstrap 为均匀行抽样，Independent Marginal 为
每列独立均匀抽样，均保留列名/顺序/dtype/缺失，用局部 numpy RNG，不读 holdout 或规则。

所有 proximity 结果都附非隐私保证声明；没有 privacy_score、真实攻击或统计推断。

# Gate2 终局删除 pilot：运行前协议

状态：2026-10-25 写定，须在首次 geometry 编码/结果查看之前由执行包记录本文件与源码 SHA256。
本协议是预先固定的开发 pilot，未注册最终确认集；后续修改必须另存版本并披露，不能覆盖失败条件。

## 目标与边界

检验一个具体问题：在预测器已能组合两份互补证据时，基于实际视觉特征距离或 attention 的保留规则，是否会选择语义重复的两帧而丢失另一项必要证据。
本轮只做生成器、几何预检、reader 验证与终局单次删除比较；**不训练 retention student**。
不保证 attention/diversity 失败，不把新视觉外观、K=2 或反事实删除本身当作创新。

本轮主动去掉未来 continuation：最后一次历史项进入发生在 query q，随即冻结历史槽；三个删除动作都直接用同一个冻结 reader 预测 q+3/q+6。
这不是多步最优控制，也没有“删除后又恢复历史档案”的操作。

## 数据与容量

每个基础 family 包含两个外观版本，每个版本是完整的四成员 quartet：bits 为 00、01、10、11。
第一位 a 与第二位 b 独立且均衡，分别控制未来 x/y 方向；必须逐 quartet 验证单独翻转任一位会改变对应未来分支。
相同 family 的两种 style 共享时间、动作、顺序、场景因素，只有重复证据的呈现样式不同。
每个 quartet 内近期观察、动作与非 cue 像素相同，只有相应 cue 内容和后续目标随 bits 改变。

| Split | 随机种子 | Quartet 数 | 跨 style 配对 family 数 | Episode 数 |
|---|---:|---:|---:|---:|
| TRAIN | 20261028 | 240 | 120 | 960 |
| Validation | 20261029 | 48 | 24 | 192 |
| Geometry | 20261030 | 24 | 12 | 96 |
| Development | 20261031 | 48 | 24 | 192 |

Style 名固定为 `benign_duplicate` 和 `distant_style_duplicate`。
“distant”只是预定的外观干预名称，不预先断言 I-JEPA 特征距离真的更远。
每个 split 完整平衡 2 styles × 2 duplicate axes × 6 cue permutations，共24个设计单元；quartet的四个bits全保留。
跨 style 的 family 是统计重采样单元，不能把两个style或四个siblings算成独立场景。

q 由 seed 决定，为12到18的整数。三个单帧 cue 位于 q−6、q−5、q−4；代码角色固定为 A、A′、B。
A与A′表示本场景被重复的那个语义位，B表示另一位；因此duplicate-axis=b时，A/A′指y方向证据，B指x方向证据。
A′与A必须表达同一个语义位，外观可不同。六种角色顺序完整平衡。
q以前的动作相同且为零，q开始的相同动作序列驱动四种未来结果；bits/角色不能藏在动作、episode ID或时间戳里。
本轮没有新增“显眼但无关”的第三类 decoy；重复证据仍有真实用途，只是与另一帧冗余。

所有比较用 R=4、K=2。共享 FIFO 前缀在 q−1 保留 q−6/q−5；q时 q−4 恰离开recent，形成真实的K+1 parent `[q−6,q−5,q−4]`。
recent固定为 `[q−3,q−2,q−1,q]`。先前不足容量时的处理遵循现有 StreamingMemory，不能提前清理或特权塞入cue。
在 q 从三个候选中删除一项，冻结两项历史；q以后的图像仅作评价目标，不再进入策略。

## 按顺序执行的五项检查

### 1. 真正的 baseline 几何预检，先于 reader 训练

使用固定官方 I-JEPA 编码配置及本轮 TRAIN-only normalization，直接调用现有 `memory.py` 的 visual-diversity 实现与真实归一化特征。
记录三种删除产生的保留对、该实现实际使用的距离/目标值、tie-break与语义类别；不得以像素距离、原始RGB视觉观感或另一个近似距离替代。
该步骤早于本轮投影basis拟合和reader参数拟合；normalization只能用TRAIN，不得用geometry/dev统计量。
同parent的FIFO及`terminal_uniform`选择也在reader训练前保存，作为不依赖reader的参照。
后者以固定seed17和稳定family key产生三候选上的均匀删除动作，配对style/siblings共用同一随机选择；保存确切RNG映射。
它只控制这个终局动作，不称为从episode起点运行的Algorithm-R reservoir，因为此前各方法共享FIFO前缀。
Attention权重依赖训练后的reader，不能伪称训练前已预测其选择；其选择在reader冻结之后、任何student训练之前保存。

**Geometry gate**：24个预定GEOMETRY quartets中，至少一个quartet有成员被实际diversity策略选为语义重复的保留对。
同时报告这一选择是严格几何偏好还是依赖tie-break；前者与后者不得混称。
若没有任何一个，停止，不训练reader、不追加搜索seed/样式，并报告本次几何干预未形成所需竞争。
所有style与全部quartet均保存，包括成功对照；不能筛掉几何不符合预期的样本，也不能用DEV决定是否修改外观。

### 2. 独立两位与四种未来必须真实成立

每个quartet核对四个bits各一次、两个单bit反事实共享其余因素、q前仅相应cue内容变化、q后两个factor各自改变未来目标。
检查输入当前/最近窗口不泄露bits，已执行动作一致，所有候选确在过去，目标索引q+3/q+6存在。
除调试用annotation外，不给reader或baseline语义角色、bits、split、family ID、预定query、未来目标或未来cue信息。
预测器仍可接收任务接口允许的计划动作；retention评分不接收未执行动作。

### 3. Reader先证明能组合和区分内容

采用与上一完成reader相同的 `DynamicsPredictor`：hidden=128、heads=4、layers=2、dropout=0、rank=128。
`heads`与`layers`是沿用配置中的溯源字段；当前实现并不读取这两个字段，不应据此将它描述为四头、两层Transformer。
实际实现为内容打分、带年龄的加权均值/速度、动作交互及MLP输出；投影与输出basis秩上限128。
新任务有四种未来，故**重拟合TRAIN-only normalization与input/output basis**，不能复用旧Gate仅两分支的输出子空间。
编码器固定；训练reader一个seed17，1024次更新，每批8quartets/32episodes，validation每128步并含step0。
Normalization使用TRAIN每个episode从0到q的所有帧，保留重复episode-frame的频数，逐维标准差下限0.05。
Input basis使用上述TRAIN前缀；output basis使用TRAIN所有合法起点t=3..q的h3/h6残差及每quartet六对未来残差差值（差值乘2）。
PCA随机seed17、3次迭代；input whitening的奇异值标准差下限0.05。

每次更新以seed17 RNG无放回抽8个TRAIN quartets，跨更新可重复；同quartet四个siblings共享起点、历史索引和dropout决定。
起点类别按每4次更新轮换：`query`固定t=q；`preuse`在整数3..q−6中均匀采样；`middle`在整数3..q中均匀采样。
每类起点内按固定顺序轮换四种历史模式，候选全集始终是0..t−4：

- `randomK`：均匀无放回取最多2项；
- `latestK`：最近最多2项；
- `latest3`：在最近最多3项中均匀无放回取最多2项；
- `random_size`：先均匀取容量0..min(2,可用项数)，再均匀无放回取相应项。

选完历史后以概率0.15将该quartet的历史清空；此dropout只用于TRAIN。
训练loss固定为 `1×feature MSE + 2×六对预测差值误差的均值 + 0.5×有序对ranking loss均值`。
对每个quartet无序对的两个方向，ranking为 `softplus(-4×(other_error−own_error)/target_separation)`；仅保留目标特征MSE separation>1e-6的quartet/horizon，不满足时不产生该ranking项。
优化器为AdamW，learning rate=0.001、weight decay=0.0001；梯度范数截断1；MultiStepLR在第341与682次更新后乘0.5。

Validation不dropout；对3起点×4历史模式的12格分别计算feature MSE，再等权平均，选择最早的最小值。
每次验证的每格RNG重置为`17000 + 100×origin_index + mode_index`，所以不同checkpoint使用同一验证采样。
选择之后写入checkpoint hash，才打开DEV做reader判定与删除比较。不得依据DEV更换checkpoint/阈值/架构。

主内容控制固定两个历史槽age为 `[6,4]`，近期窗口、时间戳、过去/计划动作、预测horizons全部相同，只替换两个槽的内容：
`AB`、`A′B`、`AA′`、`Ablank`、`Bblank`、`blank2`、`wrongA`、`wrongB`（A/B是上述角色，不固定对应物理x/y轴）。
blank为同场景中性背景；wrongA/B来自同quartet只翻对应bit的成员。它们是显式的内容干预诊断，不能冒充原始在线轨迹。
原生parent比较另外保留真实三个时间戳，不能将内容干预的结果混入在线保留指标。

语义读出是预测feature对同quartet四种真实未来feature的距离：主距离为h3/h6等权MSE，另存各horizon结果。
它是**匹配quartet的特征原型识别**，不是解码像素中的物理坐标准确率，也不是给部署policy开放未来原型。
绝对容差 `1e-6`、相对容差0定义最近原型集合；主four-way准确率要求真实/预期反事实类为唯一最近原型，平局不算通过。
另存普通固定argmin准确率、集合包含率与每位正确率，避免掩盖ties。

**Reader gate**同时要求在DEV整体及每个style内：

- `AB`与`A′B`的joint-four-way准确率各≥80%；
- `wrongA`和`wrongB`分别识别为对应单bit翻转成员的准确率各≥80%，证明两位内容均被利用；
- `AB`和`A′B`各自的MSE均低于`Ablank`且低于`Bblank`，四个差各严格大于 `1e-6`；不能用两个互补arm的平均值掩盖其中一个失败。

以上每个condition包含4项准确率检查和4项MSE检查，整体与两个style共24项，必须全部通过。
“各互补arm分别优于各单证据arm”是首次编码和结果查看之前的门槛澄清。
这是执行前确定的pilot筛查条件，不保证达到。每个duplicate axis、cue permutation、horizon全部报告，但不因小单元偶然高分选择样本。
若gate失败，结论是reader尚不能支持保留策略的解释，停止方法开发；不得称baseline丢了“reader可用”的互补证据。

原架构对recent并非完整序列读取：除current外的recent内容没有一般性利用保证。
早期起点检查只单独报告**两个目标均晚于最后cue、且不晚于q**的静止阶段，不把它称为一般early-reader质量验证。
相同静止目标可能产生局部删除标签平局，但不推出随机保留；平局行为取决于明确的tie-break与其他已学规律。

### 4. 终局同parent枚举，与语义正确集合分开

对DEV每个原生K+1 parent枚举三个删除动作，预测器、recent、动作和目标一致。
目标固定为 `L = (MSE_h3 + MSE_h6)/2`，每次只有一个共同删除；不能分别挑两个horizon的最优动作再拼接。
最优删除集合为 `D*={d:L_d≤min(L)+1e-6}`，相对容差0；固定argmin仅作实现行为记录。
另存语义正确删除集合：保留对必须包含两个不同factor的证据。不得假设reader的 `D*` 必然等于此集合。

逐quartet检查：四种bits条件下精确/容差最优删除集合是否一致、是否有公共最优动作、与语义正确集合的交集、全部动作的margin。
若siblings因reader误差偏好不同，则如实报告；不同bits的历史本来不同，这也不是“相同过去对应矛盾oracle”的证明。
反过来，任务真值对所有siblings都要求保留两个factor，也不能保证reader特有的全部oracle差距可由过去学到。
若多个动作同为最优，报告整集合，不能用tie-break差异制造排序翻转。

比较现有attention和visual-diversity；未来attention teacher可作为离线显著性对照，但必须标明其未来访问权限，不称为可部署baseline。
报告每个方法的语义重复对比例、joint MSE、regret、语义正确保留率和原型识别率；完整保留成功对照。
不生成或拟合新的retention scorer；候选数为3的离线枚举不计作部署内存。

### 5. 结果分流与下一阶段

三类结果在看结果前固定：

1. **Reader gate失败**：只能归因本任务/reader接口尚未建立，先报告失败原因；不能据此声称attention或diversity遗漏价值。
2. **Reader gate通过、强baseline也解决**：报告未发现所需保留缺口；这属于有效负结果，不靠追加筛seed或构造更偏的外观挽救结论。
3. **Reader gate通过、出现严格同parent损失差距**：只说明这个开发任务中存在候选缺口。必须进一步检查其跨family/style稳定性、是否与语义目标一致、能否从过去识别；再预先制定学生/对照并使用**另一个独立新seed集合**验证，才讨论方法收益。

Regret>1e-6定义非tie的正差距；同时给出绝对margin与完整分布，不能将数值非零等同有研究意义的效应。
仅有样例不构成正结论；至少报告48DEV quartets/24families的全部结果、各style、两duplicate axes、六种顺序。
区间如报告，按24个paired-style family进行2,000次bootstrap，seed20261025；探索性多比较不宣称确认性显著性。
本轮没有最终确认集，也没有自动启动student或decoy扩展的规则。

## 证据与可复核要求

保存设计与源hash、quartet/family/split映射、seed、动作、角色时序、真实RGB及feature绑定、TRAIN拟合ID、geometry全结果、reader学习曲线/选择凭据、全部控制预测与目标、每parent三动作损失与argmin集合、完整gate判定。
任何GPU执行须独立记录实际CUDA设备/UUID和调度状态；生成文件、提交作业和完成结果分开报告。
只有gates与数据契约全部通过，才允许将失败解释为有限历史项选择问题。
MGPO/VitaMem/OSL-MR已覆盖相关的延迟信用或反事实监督思想；这个pilot不会据此宣称新方法。

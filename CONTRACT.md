# 封存批：覆盖质量核定

## 现有材料和计算

此包的分析单位是源片批的质量缺陷，不是逐像元噪声。面积单位为平方公里，成果块互不重叠。所有概率和费用都是本次登记的业务输入，不需要外推或另估参数。

- data/batches.csv：id 是源片批号，bad_n/den 是该批存在缺陷的先验概率。所有批号在观测质检之前相互独立；同一批号的缺陷状态在本包中只有一份。
- data/pools.jsonl：每行是一次合并质检归档修订。batches 是受检批号，lo、hi 表示其中缺陷批的数量落在闭区间内。检查无测量误差，所列批号无重复。id、rev、deleted 是质检身份、正整数修订号和撤销标志。按 id 取最高 rev，撤销胜者不约束缺陷；不能退回旧版。任意同 id、同 rev 异内容都是冲突，即使已有更高版本；完全同内容重放可忽略。归档顺序不是修订顺序。
- data/mosaic.json：tiles 列出互不重叠的成果块及 area。rule 中批号字符串表示该原批质量良好；all、any 是合取、择一，atleast 的第二项为最少良好子项数，其余是子项。这是留存血缘的质量判据，不是待估计的概率公式。policy 给出面积及概率门槛。
- data/rework.json：jobs 的 repairs 是该任务重制后保证良好的批号，cost 为一次费用。任务可重叠，批号修复效果取并集，费用仍逐任务相加。inspections 可准确查明指定原始批号的缺陷状态，cost 是每次查证的确定费用。

source.load() 返回归并后的 priors、pools、tiles、jobs、inspections、policy。source.evidence_ok(model,faults) 检查一个指定原始缺陷集合是否符合全部有效质检；source.weight(model,faults) 返回该集合的先验质量；source.evaluate(model,faults,selected) 返回指定原始缺陷集合、指定重制任务 ID 集合下的 usable_tiles 和 area。source.names(rule) 提取血缘批号。以上函数没有推断未知缺陷或制定策略。

python3 source.py 展示实际材料规模、有效质检、费用、门槛及两个指定情形的成果，两个情形不是后验结论。
python3 checks.py 核对原有读取和单情形业务。checks.fixture() 提供三批号的小例：原质检要求 a、b 恰一批有缺陷，c 未检查；现状面积 0、3 的条件概率分别为 1/4、3/4，原证据概率为 1/2。小例查证 a 费用 1，重制 a 或 b 各费用 3；门槛是面积至少 5 的概率至少 19/20。可用它独立穷举检查推断及决策，不能用它的穷举方法枚举本批完整缺陷集合。

## 本轮决策的含义

从全部原质检事件 E 条件化开始。策略只能查证尚未知的登记批号，或选择一个重制任务子集并结束；空子集也允许。查证完成后可以继续查证或结束。每项最多查一次，重制只在终点一次执行。费用为沿实际路径支付的查证费与终点任务费用之和；期望按 P(原缺陷 | E) 计算，不是按查证答复均匀平均。

查证答复 0 表示原批良好，1 表示原批有缺陷。重制改变成果规则中相应批号的质量，但原缺陷、先验质量、E 及已取得答复都不变。每个正概率终点单独满足 policy；不能仅让整棵树的平均达标概率过线。不能依据未查答案选动作。动作价值相等时优先结束；同类动作比较 ID 序列的 Python 字符串字典序，结束动作的任务 ID 先排序，查证动作序列仅含查证 ID。

知识状态用三字符编码，位置依 inspections 的文件顺序，? 表示未查，0/1 表示查证答复。所谓全部可达状态是从 ??? 经任何允许查证顺序可达且 P(答复 | E)>0 的全部部分答复，不局限于最终选中策略访问的状态。零概率状态不列入策略，也不能对它做条件化。任一状态的价值只含从该状态开始的后续费用，不再包含已付查证费。

## 固定分析入口和产物

仅处理这个封存包；不要求通用概率引擎、交互页面或外部服务。保留 source.py、checks.py、data/ 的内容及现有断言。

python3 analysis.py run 从 data/ 重算并写入 result.json 和 REPORT.md。
python3 analysis.py verify 重算本批并核对已有产物，运行原回归；用直接枚举缺陷的小例判定器核对后验分布、条件分支和最优费用，该判定器不得调用分析器的状态合并或策略求解函数。还须以结果中的全部状态及动作核对终点门槛、分支权重、动作选择与费用递推。核对不符须非零退出，不得先覆盖产物掩盖差异。
python3 analysis.py demo 展示从真实材料重算的现状覆盖风险、最优期望费用和策略中各非零答复分支的后续动作。
.pointwise-runtime.json 的 verify、demo 分别改为上述 verify、demo argv，setup 保持为空。

result.json 使用下列外部结构，概率及费用的有理数统一为最简 n/d 字符串，分母为正；整数也写 n/1，概率分布只列正质量：
{
  "evidence_probability": "n/d",
  "current": {
    "tile_usable": {"T00": "n/d"},
    "area_pmf": [[0, "n/d"]],
    "tail_probability": "n/d"
  },
  "states": {
    "???": {
      "probability": "n/d",
      "plans": [{"jobs": [], "cost": 0, "tail_probability": "n/d"}],
      "value": "n/d",
      "action": {"inspect": "I0"}
    }
  },
  "expected_cost": "n/d"
}
这是结构示意，不是本批答案。evidence_probability 是 P(E)；current 是原质检后的现状；tile_usable 包含全部成果块；area_pmf 按面积递增。状态的 probability 是 P(该部分答复 | E)，不是不同部分答复间归一化的权重。每个状态的 plans 包含六项任务的全部子集，jobs 内部排序，plans 按 jobs 序列字典序排列；tail_probability 是在该状态原证据下、执行该组合后达到面积门槛的概率。结束动作写 {"stop": [任务ID序列]}，与 inspect 二选一。expected_cost 等于根状态 value。

REPORT.md 须使人能由原始行定位修订选择、质检依赖与共享血缘，解释现状分布为何不能由块边际概率独立拼成；说明所合并的分析状态、转移质量、覆盖量记账、释放状态的条件及完整性依据，并用所有备选动作的条件期望说明策略最优性。只需论证本批的合并和决策，不要求任意图的通用复杂度定理。

若原材料同版冲突、非法或 P(E)=0，run 非零退出并给出原因，不写部分核定，也不覆盖已有报告与结果。

"""Joint Backward Layout & Sequence search —— 顺序 × 布局 的联合反向搜索。

2026-09-28: 欧氏距离基线已实现于 search.py / bridge.py / run_bridge.py。
运行 python -m jbls.run_bridge。当前范围见 README.md / ASSUMPTIONS.md。
以下保留原方案笔记；旧 cost、Hall、精确搜索的迁移尚未完成。

论文一固定装配顺序 pi、只对后缀布局 x_{k:n} 反向赋值；论文二只搜顺序、不管布局。
本包把 pi 也变成搜索变量：反向树上每个节点同时决定"第 k 步装哪个零件"和
"它初始摆在哪"。

    节点 N_k = (已定后缀顺序 pi_{k:n}, 对应后缀布局 x_{k:n}, 累计代价)

两篇在方向上天然一致：论文一是后缀反向赋值；论文二 `planning/sequence.py` 的
存储本身就是 removal 顺序再 reverse 成装配顺序（assembly-by-disassembly）。

## 步可行性

单臂方案下稳定性是硬约束，与论文一原有各项合取：

    phi_k = 稳定性 ∧ 抓取可行 ∧ IK ∧ 碰撞 ∧ 运动

- 稳定性：`asmcore.mechanics.equilibrium.check_equilibrium(...).status == "feasible"`。
  任一中间状态 S=0 即判该顺序不可行（单臂没有第二只手可托），直接剪枝。
  被剪原因可用 `find_support_requirements` 诊断，但它不进入搜索。
- 其余各项：`adapters` 的三个协议。

## 待搬运

`sealp/examples/layout/bsfs/` 已实现固定 pi 下的完整反向搜索，逐一提升为正式模块：

    domain.py   候选域、离散 staging 网格、占据位图
    cost.py     J_nom（末端折线长度）与 admissible 下界（取放直线弦）
    pruning.py  Hall 定理匹配 + 域传播
    oracle.py   L0/L1/L2/L3 分层可行性（此处接 adapters，并加入稳定性项）
    search.py   精确 A*/分支定界 + anytime beam

## 未决的理论点

论文一的后缀保持性（extension invariance，见 sealp 的 `bsfs/ASSUMPTIONS.md`）
成立的前提是 **pi 固定**：加入更早的 x_{k-1} 不会使 j >= k 的步骤失效。一旦 pi 也
成为搜索变量，这一条需要重新叙述与证明——它既是本工作的理论贡献，也是最容易
被审稿人追问的地方。实现前先把新的假设写进本包的 ASSUMPTIONS.md。
"""

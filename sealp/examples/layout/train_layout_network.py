"""统一训练脚本 (CLI)。

用法示例:
    python -m sealp.examples.layout.train_layout_network \
        --dataset sealp/examples/layout/_output/layout_dataset_v2.jsonl \
        --model mlp --epochs 200 --batch-size 32 --lr 1e-3 \
        --split-mode stratified --early-stop-patience 30 \
        --save-dir checkpoints/layout_models_debug

支持模型: mlp / deepsets / set_transformer / gcn / gat / transformer /
          pointnet / cvae / diffusion / sagpn / all

训练目标:
    L = L_cls + alpha*L_score_feasible + beta*L_xy_elite + gamma*L_station_elite
    - L_cls          : 所有样本;
    - L_score        : 只 feasible;
    - L_xy / L_station: 只 elite feasible (score >= 分位阈值)。

验证指标: ROC-AUC / PR-AUC / recall@K / precision@K / top-K avg true score /
          score MAE·RMSE·Spearman / enrichment。
best checkpoint 用 composite metric。
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning.losses import LossWeights
from layout_learning.relseqgen_losses import RelSeqGenLossWeights
from layout_learning.models import MODEL_NAMES, size_kwargs
from layout_learning.train import train_model, resolve_run_dir, resolve_shared_split_path


_METRIC_CHOICES = ["composite", "pr_auc", "roc_auc", "auc", "recall_at_k",
                   "precision_at_k", "topk_hit", "topk_avg_score_norm",
                   "score_spearman", "feas_acc"]


def _parse_score_threshold(spec: str):
    """返回 (elite_quantile or None, abs_threshold)。

    - "quantile:0.70" -> (0.70, 0.0)  按训练集 feasible 分数分位数
    - "0.4"           -> (None, 0.4)  绝对阈值
    """
    spec = str(spec).strip()
    if spec.lower().startswith("quantile:"):
        return float(spec.split(":", 1)[1]), 0.0
    return None, float(spec)


def _parse_pos_weight(spec: str):
    """解析 --pos-weight: 'auto' 或正浮点数。"""
    text = str(spec).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pos-weight 必须是 'auto' 或正浮点数，例如 2.5"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("--pos-weight 必须大于 0")
    return value


def _parse_args():
    p = argparse.ArgumentParser(description="Layout learning trainer")
    p.add_argument("--dataset", required=True, help="layout_dataset.jsonl 路径")
    p.add_argument(
        "--dataset-manifest", default=None,
        help="不可变数据 manifest；datasets/repro 路径会自动要求同名 manifest")
    p.add_argument("--model", default="mlp",
                   help=f"模型名 (可选: {MODEL_NAMES + ['all']})")
    p.add_argument("--model-size", default=None, choices=[None, "small", "base"],
                   help="生成式模型尺寸 (sagpn/cvae/diffusion); scorer 忽略")
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="覆盖模型 hidden 维度 (seqrel/deepsets 等)")
    p.add_argument("--dropout", type=float, default=None,
                   help="覆盖模型 dropout")
    p.add_argument("--disable-relation", action="store_true",
                   help="SeqRel 结构消融: 完全跳过 relation/message-passing 层")
    p.add_argument("--disable-sequence", action="store_true",
                   help="SeqRel 结构消融: 屏蔽 order/adjacent/parent 信息, 保留 spatial")
    p.add_argument(
        "--relation-mode", default=None,
        choices=["staging_dynedge", "staging_topology_only"],
        help="DynaSeqRel 关系模式; 默认 staging_dynedge")
    p.add_argument(
        "--dynedge-edge-encoding", default=None,
        choices=["raw_v1", "edge_mlp_v2"],
        help="动态边编码; 新实验建议 edge_mlp_v2，raw_v1 兼容首轮 checkpoint")
    p.add_argument(
        "--dynedge-head-mode", default=None,
        choices=[
            "shared_v1",
            "task_specific_score_v3",
            "frozen_residual_score_adapter_v4",
        ],
        help="DynEdge head 模式; 新实验建议 task_specific_score_v3")
    p.add_argument("--strict-shared-init", action="store_true",
                   help="v3: 从同 seed 的 v2 reference 精确复制所有公共参数")
    p.add_argument("--shared-init-reference", default="v2", choices=["v2"])
    p.add_argument("--model-init-seed", type=int, default=None)
    p.add_argument("--score-trunk-init-seed", type=int, default=None)
    p.add_argument("--dataloader-seed", type=int, default=None)
    p.add_argument("--base-checkpoint", default=None,
                   help="adapter-v4 所需的 v2 best-PR checkpoint")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="adapter-v4 必需；冻结完整 v2 base，仅训练 adapter")
    p.add_argument("--adapter-hidden-dim", type=int, default=64)
    p.add_argument("--adapter-dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (默认自动)")
    p.add_argument("--save-dir", default=None,
                   help="实验输出目录; 默认 {run-root}/{model}/{split}/seed{N}/")
    p.add_argument("--run-root", default=os.path.join("checkpoints", "layout_models_repro"),
                   help="规范实验根目录 (与 model/split/seed 组合)")
    p.add_argument("--split-indices", default=None,
                   help="共享 train/val 划分 JSON; 默认 {run-root}/_splits/{split}/seed{N}/")
    p.add_argument("--code-version", default="seqrel-v2-repro",
                   help="写入 checkpoint/config 的代码版本标签")
    p.add_argument("--topk", type=int, default=10, help="recall@K / precision@K 的 K")
    # 特征版本: v1=旧固定归一化(兼容旧ckpt/MLP baseline); v2=自适应归一化(面向迁移)
    p.add_argument("--feature-version", default="v1", choices=["v1", "v2"])
    # split & early stop
    p.add_argument("--split-mode", default="stratified",
                   choices=["stratified", "random", "seed_holdout", "region_holdout",
                            "geometry_holdout"])
    p.add_argument("--region-holdout-mode", default="auto",
                   choices=["auto", "rc", "xy_grid"],
                   help="region_holdout 分桶方式: rc(3x3网格)/xy_grid(连续站位网格)/auto")
    p.add_argument("--region-xy-grid", type=int, default=3,
                   help="xy_grid 分桶的网格划分数")
    p.add_argument("--early-stop-patience", type=int, default=30,
                   help="0 表示关闭 early stopping")
    p.add_argument("--early-stop-metric", default="composite", choices=_METRIC_CHOICES)
    p.add_argument(
        "--save-metric-checkpoints", action="store_true",
        help="额外保存 best_pr_auc.pt / best_spearman.pt / best_legacy_composite.pt")
    p.add_argument("--min-delta", type=float, default=1e-4)
    # loss 权重
    p.add_argument("--alpha", type=float, default=1.0, help="score 回归权重 (等价 --score-weight)")
    p.add_argument("--score-weight", type=float, default=None,
                   help="score 回归权重别名; 提供时覆盖 --alpha (向后兼容)")
    p.add_argument("--beta", type=float, default=1.0, help="xy proposal 权重")
    p.add_argument("--gamma", type=float, default=0.5, help="station/region 权重")
    p.add_argument("--kl", type=float, default=0.01, help="CVAE KL 权重")
    p.add_argument(
        "--pos-weight", type=_parse_pos_weight, default="auto",
        help="可行样本正类加权: 'auto'=按当前训练 split 的 neg/pos 自动计算，或指定正浮点数")
    p.add_argument("--score-threshold", default="quantile:0.70",
                   help="elite 阈值: 'quantile:0.70' 或绝对值如 '0.4'")
    # ---- seqrel 专用 loss / 调试 (其它模型默认关闭, 不受影响) ----
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="pair-ranking loss 权重 (seqrel 建议 0.5); 0=关闭")
    p.add_argument("--fail-weight", type=float, default=0.0,
                   help="fail 辅助分类权重 (seqrel 建议 0.2); 0=关闭")
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--rank-min-score-gap", type=float, default=0.05)
    p.add_argument("--rank-pairs-per-batch", type=int, default=256)
    p.add_argument("--use-focal", action="store_true",
                   help="L_cls 使用 focal loss (默认 BCEWithLogits)")
    p.add_argument("--no-focal", action="store_true",
                   help="显式关闭 focal, 使用 BCEWithLogits (覆盖 --use-focal)")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--debug-grad", action="store_true",
                   help="每 epoch 前几个 batch 打印 logits/梯度诊断并做自动检查")
    p.add_argument("--debug-batches", type=int, default=3)
    p.add_argument("--limit-samples", type=int, default=0,
                   help=">0 时分层截断数据集 (overfit smoke test 用)")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="随机置换标签的 sanity check (指标应崩到随机水平)")
    p.add_argument("--dry-run", action="store_true",
                   help="只解析并检查参数/路径/输出目录，不进入 epoch 训练循环")
    # ---- RelSeqGen generator ----
    p.add_argument("--generator-pose-mode", default="predict",
                   choices=["predict", "fixed"],
                   help="RelSeqGen 姿态预测模式; fixed 仅调试")
    p.add_argument("--generator-elite-quantile", type=float, default=0.70,
                   help="RelSeqGen proposal 监督的 elite feasible 分位数")
    p.add_argument("--generator-score-threshold", type=float, default=0.0,
                   help="RelSeqGen elite 绝对分数阈值 (0=使用分位数)")
    p.add_argument("--geometry-holdout-domains", default="",
                   help="逗号分隔 geometry domain 列表, 用于 geometry_holdout split")
    p.add_argument("--teacher-forcing-start", type=float, default=1.0)
    p.add_argument("--teacher-forcing-end", type=float, default=0.3)
    p.add_argument("--scheduled-sampling-start-epoch", type=int, default=30)
    return p.parse_args()


def main():
    args = _parse_args()
    if args.disable_relation and args.disable_sequence:
        raise ValueError("--disable-relation 与 --disable-sequence 不能同时启用")
    if (args.disable_relation or args.disable_sequence) and args.model != "seqrel":
        raise ValueError("--disable-relation/--disable-sequence 仅适用于 --model seqrel")
    if args.relation_mode is not None and args.model != "dynaseqrel_dynedge":
        raise ValueError("--relation-mode 仅适用于 --model dynaseqrel_dynedge")
    if (args.dynedge_edge_encoding is not None
            and args.model != "dynaseqrel_dynedge"):
        raise ValueError(
            "--dynedge-edge-encoding 仅适用于 --model dynaseqrel_dynedge")
    if (args.dynedge_head_mode is not None
            and args.model != "dynaseqrel_dynedge"):
        raise ValueError(
            "--dynedge-head-mode 仅适用于 --model dynaseqrel_dynedge")
    adapter_mode = (
        args.model == "dynaseqrel_dynedge"
        and args.dynedge_head_mode == "frozen_residual_score_adapter_v4")
    if args.strict_shared_init:
        if args.model != "dynaseqrel_dynedge":
            raise ValueError("--strict-shared-init 仅适用于 dynaseqrel_dynedge")
        if args.dynedge_head_mode != "task_specific_score_v3":
            raise ValueError(
                "--strict-shared-init 要求 "
                "--dynedge-head-mode task_specific_score_v3")
        if adapter_mode:
            raise ValueError("strict shared init 与 adapter-v4 不能同时启用")
    if adapter_mode:
        if not args.base_checkpoint:
            raise ValueError("adapter-v4 必须提供 --base-checkpoint")
        if not args.freeze_backbone:
            raise ValueError("adapter-v4 必须显式提供 --freeze-backbone")
        if args.early_stop_metric != "score_spearman":
            raise ValueError(
                "adapter-v4 的 --early-stop-metric 必须为 score_spearman")
        effective_score_weight = (
            args.score_weight if args.score_weight is not None else args.alpha)
        if effective_score_weight != 1.0 or args.rank_weight != 0.5:
            raise ValueError(
                "adapter-v4 固定 --score-weight 1.0 --rank-weight 0.5")
        if args.fail_weight != 0.0 or args.use_focal:
            raise ValueError(
                "adapter-v4 不训练分类/fail loss；不得传 --use-focal，"
                "--fail-weight 必须为 0")
    elif args.freeze_backbone or args.base_checkpoint:
        raise ValueError(
            "--freeze-backbone/--base-checkpoint 仅适用于 adapter-v4")
    elite_quantile, abs_thr = _parse_score_threshold(args.score_threshold)
    # auto 模式的真正权重会在 train_model 完成 train/val split 后计算。
    # 这里使用 1.0 作为临时占位，避免把字符串传入 loss dataclass。
    initial_pos_weight = 1.0 if args.pos_weight == "auto" else float(args.pos_weight)
    score_weight = args.score_weight if args.score_weight is not None else args.alpha
    use_focal = False if args.no_focal else args.use_focal
    weights = LossWeights(alpha=score_weight, beta=args.beta, gamma=args.gamma,
                          kl=args.kl, pos_weight=initial_pos_weight,
                          score_threshold=abs_thr,
                          rank_weight=args.rank_weight,
                          fail_weight=args.fail_weight,
                          rank_margin=args.rank_margin,
                          rank_min_score_gap=args.rank_min_score_gap,
                          rank_pairs_per_batch=args.rank_pairs_per_batch,
                          use_focal=use_focal,
                          focal_gamma=args.focal_gamma,
                          focal_alpha=args.focal_alpha)
    models = MODEL_NAMES if args.model == "all" else [args.model]
    shared_split = args.split_indices
    if shared_split is None:
        shared_split = resolve_shared_split_path(args.run_root, args.split_mode, args.seed)
    holdout_domains = [
        d.strip() for d in args.geometry_holdout_domains.split(",") if d.strip()]
    rq = RelSeqGenLossWeights(
        elite_quantile=float(args.generator_elite_quantile),
        score_threshold=float(args.generator_score_threshold),
    )
    for name in models:
        mkwargs = size_kwargs(name, args.model_size)   # 解析后的显式构造参数
        if args.hidden_dim is not None:
            mkwargs["hidden"] = args.hidden_dim
        if args.dropout is not None:
            mkwargs["dropout"] = args.dropout
        if args.disable_relation:
            mkwargs["disable_relation"] = True
        if args.disable_sequence:
            mkwargs["disable_sequence"] = True
        if name == "dynaseqrel_dynedge":
            mkwargs["relation_mode"] = args.relation_mode or "staging_dynedge"
            mkwargs["dynamic_k_spatial"] = 2
            mkwargs["edge_encoding"] = (
                args.dynedge_edge_encoding or "edge_mlp_v2")
            mkwargs["head_mode"] = (
                args.dynedge_head_mode or "task_specific_score_v3")
            if adapter_mode:
                mkwargs["adapter_hidden_dim"] = args.adapter_hidden_dim
                mkwargs["adapter_dropout"] = args.adapter_dropout
        if name == "relseqgen":
            mkwargs["pose_mode"] = args.generator_pose_mode
        save_dir = args.save_dir or resolve_run_dir(
            args.run_root, name, args.split_mode, args.seed)
        required_outputs = (
            f"{name}_best.pt", "metrics.json", "training_history.csv",
            "config.json", "split_indices.json", "train.log",
        )
        if (args.disable_relation or args.disable_sequence
                or name == "dynaseqrel_dynedge") and all(
                os.path.isfile(os.path.join(save_dir, p)) for p in required_outputs):
            raise FileExistsError(
                "结构消融目录已包含完整产物，拒绝覆盖: "
                f"{os.path.abspath(save_dir)}")
        print("=" * 70)
        print(f"[train] training model = {name}  size={args.model_size or 'default'} "
              f"kwargs={mkwargs}")
        print(f"[train] save_dir = {os.path.abspath(save_dir)}")
        print(f"[train] shared_split = {os.path.abspath(shared_split)}")
        print("=" * 70)
        if args.dry_run:
            if not os.path.isfile(args.dataset):
                raise FileNotFoundError(f"dataset 不存在: {args.dataset}")
            if not os.path.isfile(shared_split):
                raise FileNotFoundError(f"shared split 不存在: {shared_split}")
            print("[dry-run] model_kwargs =", mkwargs)
            print("[dry-run] loss =", {
                "score_weight": weights.alpha,
                "rank_weight": weights.rank_weight,
                "fail_weight": weights.fail_weight,
                "use_focal": weights.use_focal,
                "focal_alpha": weights.focal_alpha,
                "focal_gamma": weights.focal_gamma,
            })
            print("[dry-run] validation passed; training not started")
            continue
        train_model(
            dataset_path=args.dataset,
            model_name=name,
            save_dir=save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
            loss_weights=weights,
            pos_weight=args.pos_weight,
            model_kwargs=mkwargs,
            topk=args.topk,
            split_mode=args.split_mode,
            region_holdout_mode=args.region_holdout_mode,
            region_xy_grid=args.region_xy_grid,
            feature_version=args.feature_version,
            elite_quantile=elite_quantile,
            early_stop_patience=args.early_stop_patience,
            early_stop_metric=args.early_stop_metric,
            min_delta=args.min_delta,
            save_metric_checkpoints=args.save_metric_checkpoints,
            debug_grad=args.debug_grad,
            debug_batches=args.debug_batches,
            limit_samples=args.limit_samples,
            shuffle_labels=args.shuffle_labels,
            split_indices_path=args.split_indices,
            shared_split_path=shared_split,
            run_root=args.run_root,
            code_version=args.code_version,
            strict_shared_init=args.strict_shared_init,
            shared_init_reference=args.shared_init_reference,
            model_init_seed=args.model_init_seed,
            score_trunk_init_seed=args.score_trunk_init_seed,
            dataloader_seed=args.dataloader_seed,
            base_checkpoint_path=args.base_checkpoint,
            dataset_manifest_path=args.dataset_manifest,
            geometry_holdout_domains=holdout_domains or None,
            generator_pose_mode=args.generator_pose_mode,
            relseqgen_loss_weights=rq if name == "relseqgen" else None,
            generator_elite_quantile=args.generator_elite_quantile,
            verbose=True,
        )


if __name__ == "__main__":
    main()

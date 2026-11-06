"""把一次 ``assembly_demo`` 运行结果做成可离线打开的对比页面（viewer.html）。

    & $py -X utf8 -m curobo_bridge.make_viewer curobo_bridge/_output/assembly_bridge_20260918_200433

页面：左右两个 3D 视图按"同一规则"执行时间同步回放 cuRobo 与 WRS RRT 的完整装配；
下方是两条共用时间轴的甘特图和逐段明细表。数据全部内嵌（three.js 从 jsDelivr 加载）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "viewer_template.html"


def build(run_dir: Path) -> Path:
    viz = json.loads((run_dir / "viz.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data = dict(
        viz=viz,
        summary=run["summary"],
        setup=run.get("setup_s", {}),
        screening=run.get("screening_by_part", []),
        screening_s=run.get("grasp_screening_s"),
        layout_attempts=len(run.get("layout_attempts", [])),
        example=run.get("example", "tower"),
        order=run["order"],
        site=run["site"],
        created=run["created"],
        segments={n: [dict(part=s["part"], leg=s["leg"], success=s["success"],
                           source=s["source"], plan_s=s["plan_s"], verify_s=s["verify_s"],
                           same_s=s.get("same_s"), native_s=s.get("exec_s"),
                           tiers=s["tiers"]) for s in segs]
                  for n, segs in run["segments"].items()},
    )
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = run_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))

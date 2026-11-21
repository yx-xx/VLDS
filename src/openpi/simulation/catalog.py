"""Scene and policy choices. Catalog discovery never imports a simulator or a model."""

# ruff: noqa: RUF001 -- Chinese interface text uses Chinese punctuation.

import ast
import importlib.util
import pathlib

from openpi.simulation import paths

LIBERO_SUITES = {
    "libero_spatial": "空间关系",
    "libero_object": "目标物体",
    "libero_goal": "目标要求",
    "libero_10": "多步骤任务",
    "libero_90": "综合任务",
}
POLICIES = {
    "libero": {
        "pi05": ("π₀.₅ · LIBERO", "pi05_libero"),
        "pi0": ("π₀ · LIBERO", "pi0_libero"),
        "pi0_fast": ("π₀-FAST · LIBERO", "pi0_fast_libero"),
    },
    "aloha": {"pi0": ("π₀ · ALOHA 双臂", "pi0_aloha_sim")},
}


def _literal_assignment(path: pathlib.Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing {name} in {path}")


def operation_type(name: str) -> str:
    if " and " in name:
        return "组合操作"
    if "stack" in name:
        return "堆叠"
    if any(word in name for word in ("open", "close")):
        return "开合操作"
    if "turn " in name:
        return "设备开关"
    if "push" in name:
        return "推移"
    return "抓取放置"


def build_catalog() -> dict:
    suites = []
    index = paths.LIBERO_ROOT / "benchmark/libero_suite_task_map.py"
    if index.exists():
        task_map = _literal_assignment(index, "libero_task_map")
        order = _literal_assignment(paths.LIBERO_ROOT / "benchmark/__init__.py", "task_orders")[0]
        for suite, label in LIBERO_SUITES.items():
            names = task_map[suite]
            if suite != "libero_90":
                names = [names[i] for i in order]
            tasks = []
            for task_id, name in enumerate(names):
                instruction = name.replace("_", " ")
                tasks.append({"id": task_id, "name": instruction, "kind": operation_type(instruction)})
            suites.append({"id": suite, "name": label, "tasks": tasks})
    aloha_available = importlib.util.find_spec("gym_aloha") is not None
    return {
        "libero": {
            "name": "LIBERO · Franka 单臂",
            "available": bool(suites) and (paths.ROOT / "examples/libero/.venv/bin/python").exists(),
            "note": "选择任务场景与官方初始状态，由模型现场生成动作。",
            "suites": suites,
        },
        "aloha": {
            "name": "ALOHA Sim · 双臂",
            "available": aloha_available,
            "note": "默认权重针对方块交接；插接推理需要兼容的插接任务权重。",
            "suites": [
                {
                    "id": "transfer",
                    "name": "双臂交接",
                    "tasks": [{"id": 0, "name": "Transfer cube", "kind": "物体交接"}],
                },
                {
                    "id": "insertion",
                    "name": "双臂插接",
                    "tasks": [{"id": 0, "name": "Bimanual insertion", "kind": "精细插接"}],
                },
            ],
        },
    }


def validate_selection(selection: dict, catalog: dict) -> dict:
    dataset = selection.get("dataset", "libero")
    if dataset not in catalog or not catalog[dataset]["available"]:
        raise ValueError("所选环境尚未安装。LIBERO 请运行 examples/libero/setup_franka.sh。")
    suite = next((s for s in catalog[dataset]["suites"] if s["id"] == selection.get("suite")), None)
    if suite is None:
        raise ValueError("请选择有效的任务类型。")
    task_id = int(selection.get("task_id", 0))
    task = next((t for t in suite["tasks"] if t["id"] == task_id), None)
    if task is None:
        raise ValueError("请选择有效的任务。")
    seed = int(selection.get("seed", 7))
    initial_state = int(selection.get("initial_state", 0))
    if not 0 <= seed < 2**32 or initial_state < 0:
        raise ValueError("种子须在 0 至 4294967295 之间，初始状态编号须非负。")
    model = selection.get("model", next(iter(POLICIES[dataset])))
    if model not in POLICIES[dataset]:
        raise ValueError("所选模型与当前环境不匹配。")
    checkpoint = str(selection.get("checkpoint", "")).strip()
    if checkpoint and not checkpoint.startswith("gs://") and not pathlib.Path(checkpoint).expanduser().is_dir():
        raise ValueError("检查点目录不存在。")
    if checkpoint and not checkpoint.startswith("gs://"):
        checkpoint = str(pathlib.Path(checkpoint).expanduser().resolve())
    return {
        "dataset": dataset,
        "suite": suite["id"],
        "task_id": task_id,
        "task_name": task["name"],
        "seed": seed,
        "initial_state": initial_state,
        "model": model,
        "checkpoint": checkpoint,
        "software_rendering": bool(selection.get("software_rendering", True)),
    }


def policy_spec(selection: dict) -> tuple[str, str]:
    if selection["dataset"] == "aloha" and selection["suite"] == "insertion" and not selection["checkpoint"]:
        raise ValueError("插接任务需要相应的 ALOHA 检查点；可以先打开场景，或填入兼容权重后加载模型。")
    config = POLICIES[selection["dataset"]][selection["model"]][1]
    return config, selection["checkpoint"] or f"gs://openpi-assets/checkpoints/{config}"

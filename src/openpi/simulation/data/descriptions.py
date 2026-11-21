"""Readable labels for known demonstrations; original instructions remain available."""

# ruff: noqa: RUF001
import contextlib
import json

PROFILES = {
    "yifengzhu-hf/LIBERO-datasets": (
        "LIBERO · 原始仿真演示",
        "空间、物体、目标和长任务演示，保留原始 HDF5 数据；具备模型与状态的回合可三维回放。",
        "Franka Panda 单臂",
    ),
    "physical-intelligence/libero": (
        "LIBERO · 桌面物体操作",
        "杯碗摆放、食品收纳、抽屉开合，以及微波炉和炉灶操作。",
        "Franka Panda 单臂",
    ),
    "lerobot/aloha_sim_transfer_cube_human": (
        "ALOHA · 双臂方块交接",
        "右臂抓取方块，再将方块交给左臂；观看双臂如何配合。",
        "ALOHA 双臂",
    ),
    "lerobot/pusht": (
        "PushT · 平面推物",
        "用推杆移动 T 形物块，使其与桌面上的 T 形目标重合。",
        "平面推杆",
    ),
    "lerobot/xarm_lift_medium": (
        "xArm · 方块抬升",
        "机械臂接近、抓住并抬起方块，观察夹爪与物体的接触。",
        "xArm 单臂",
    ),
}

_TASK_PAIRS = [
    (
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "白杯放左盘，黄白杯放右盘",
    ),
    (
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "白杯放盘中，巧克力布丁放盘右侧",
    ),
    ("put the yellow and white mug in the microwave and close it", "黄白杯放入微波炉并关门"),
    ("turn on the stove and put the moka pot on it", "打开炉灶并放上摩卡壶"),
    ("put both the alphabet soup and the cream cheese box in the basket", "字母汤与奶油奶酪放入篮中"),
    ("put both the alphabet soup and the tomato sauce in the basket", "字母汤与番茄酱放入篮中"),
    ("put both moka pots on the stove", "两个摩卡壶放到炉灶上"),
    ("put both the cream cheese box and the butter in the basket", "奶油奶酪与黄油放入篮中"),
    ("put the black bowl in the bottom drawer of the cabinet and close it", "黑碗放入柜子底层抽屉并关上"),
    ("pick up the book and place it in the back compartment of the caddy", "书放入收纳架后侧隔间"),
    ("put the bowl on the plate", "碗放到盘子上"),
    ("put the wine bottle on the rack", "酒瓶放到架子上"),
    ("open the top drawer and put the bowl inside", "打开顶层抽屉并放入碗"),
    ("put the cream cheese in the bowl", "奶油奶酪放入碗中"),
    ("put the wine bottle on top of the cabinet", "酒瓶放到柜顶"),
    ("push the plate to the front of the stove", "盘子推到炉灶前方"),
    ("turn on the stove", "打开炉灶"),
    ("put the bowl on the stove", "碗放到炉灶上"),
    ("put the bowl on top of the cabinet", "碗放到柜顶"),
    ("open the middle drawer of the cabinet", "打开柜子中层抽屉"),
    ("pick up the black bowl next to the cookie box and place it on the plate", "饼干盒旁的黑碗放到盘子上"),
    (
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "木柜顶层抽屉内的黑碗放到盘子上",
    ),
    ("pick up the black bowl on the ramekin and place it on the plate", "烤盅上的黑碗放到盘子上"),
    ("pick up the black bowl on the stove and place it on the plate", "炉灶上的黑碗放到盘子上"),
    (
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "盘子与烤盅之间的黑碗放到盘子上",
    ),
    ("pick up the black bowl on the cookie box and place it on the plate", "饼干盒上的黑碗放到盘子上"),
    ("pick up the black bowl next to the plate and place it on the plate", "盘子旁的黑碗放到盘子上"),
    ("pick up the black bowl next to the ramekin and place it on the plate", "烤盅旁的黑碗放到盘子上"),
    ("pick up the black bowl from table center and place it on the plate", "桌面中央的黑碗放到盘子上"),
    ("pick up the black bowl on the wooden cabinet and place it on the plate", "木柜上的黑碗放到盘子上"),
    ("pick up the cube with the right arm and transfer it to the left arm", "右臂抓取方块并交给左臂"),
    ("push the t-shaped block onto the t-shaped target", "将 T 形物块推到 T 形目标上"),
    ("pick up the cube and lift it", "抓住方块并抬起"),
]
_TASK_PAIRS.extend(
    (f"pick up the {name} and place it in the basket", f"{label}放入篮中")
    for name, label in (
        ("orange juice", "橙汁"),
        ("ketchup", "番茄沙司"),
        ("cream cheese", "奶油奶酪"),
        ("bbq sauce", "烧烤酱"),
        ("alphabet soup", "字母汤"),
        ("milk", "牛奶"),
        ("salad dressing", "沙拉酱"),
        ("butter", "黄油"),
        ("tomato sauce", "番茄酱"),
        ("chocolate pudding", "巧克力布丁"),
    )
)
TASK_LABELS = dict(_TASK_PAIRS)


def task_label(instruction):
    return TASK_LABELS.get(instruction.strip().rstrip(".").lower(), instruction)


def dataset_source(root):
    repo = root.name.replace("--", "/")
    if root.is_dir():
        for name in ("dataset_source.json", "preview_source.json"):
            marker = root / name
            if marker.is_file():
                with contextlib.suppress(OSError, ValueError):
                    return json.loads(marker.read_text()).get("repo_id", repo)
        canonical = f"{root.parent.name}/{root.name}"
        if canonical in PROFILES:
            return canonical
    return repo


def dataset_description(index):
    known = PROFILES.get(dataset_source(index.root))
    if known:
        return known
    kinds = "、".join(sorted({task.kind for task in index.tasks})) or "未标注操作"
    return index.root.stem, f"已索引的操作类型：{kinds}。", index.robot or "机器人未标注"

"""Dataset versions compatible with the installed LeRobot and LIBERO environments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str
    format: str = "lerobot"


SOURCES = {
    "libero": DatasetSource("physical-intelligence/libero", "v2.0"),
    "aloha": DatasetSource("lerobot/aloha_sim_transfer_cube_human", "v2.1"),
    "pusht": DatasetSource("lerobot/pusht", "v2.1"),
    "xarm": DatasetSource("lerobot/xarm_lift_medium", "v2.1"),
    # This is the repository used by LIBERO's own download utility.
    "libero_raw": DatasetSource("yifengzhu-hf/LIBERO-datasets", "main", "libero_hdf5"),
}

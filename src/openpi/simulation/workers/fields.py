"""Run recorded-action controls or geometric field sequences in full LIBERO dynamics.

This worker uses the standalone LIBERO interpreter. It never imports torch or a
learned policy. Geometric switching has no validated operation observer, so even
successful runs are diagnostic and cannot release a VLDS training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.fields import primitive
from openpi.simulation.runtime.field_control import OSCFieldAdapter


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _relocate_xml(xml, source):
    # Use the replay adapter's existing asset relocation rather than another path map.
    from openpi.simulation.workers.replay import relocate_assets

    return relocate_assets(xml, source)


def actual_pose(env):
    sim = env.sim
    # mj_step can leave derived site transforms at the preceding physics state.
    # Refresh before matching the state used by OSC.set_goal / controller.update.
    sim.forward()
    return (
        sim.data.get_site_xpos("gripper0_grip_site").copy(),
        sim.data.get_site_xmat("gripper0_grip_site").copy(),
        sim.data.get_site_xvelp("gripper0_grip_site").copy(),
        sim.data.get_site_xvelr("gripper0_grip_site").copy(),
    )


def capture_query(group, observation, env, step, *, before_activation=True):
    p, r, v, omega = actual_pose(env)
    group.attrs["control_step"] = step
    group.attrs["simulation_time"] = float(env.sim.data.time)
    group.attrs["capture_before_activation"] = before_activation and group.name.rsplit("/", 1)[-1] != "final"
    group.attrs["memory_status"] = "observer_not_available; no privileged facts supplied"
    group["position"], group["rotation"] = p, r
    group["linear_velocity"], group["angular_velocity"] = v, omega
    group["joint_position"] = observation["robot0_joint_pos"]
    group["gripper_position"] = observation["robot0_gripper_qpos"]
    for camera in ("agentview_image", "robot0_eye_in_hand_image"):
        group.create_dataset(camera, data=observation[camera], compression="gzip", compression_opts=1)


def privileged_audit(env, names):
    """Evaluation-only geometry/contact evidence; never read by the action controller."""
    poses, bilateral = [], []
    for name in names:
        obj = env.env.objects_dict[name]
        poses.append(
            np.r_[
                env.sim.data.get_body_xpos(obj.root_body),
                Rotation.from_matrix(env.sim.data.get_body_xmat(obj.root_body)).as_quat(),
            ]
        )
        bilateral.append(bool(env.env._check_grasp(env.robots[0].gripper, obj)))  # noqa: SLF001 -- evaluation-only installed helper.
    return np.array(poses), np.array(bilateral, dtype=bool)


def goal_predicates(env):
    return [
        {"predicate": list(goal), "value": bool(env.env._eval_predicate(goal))}  # noqa: SLF001 -- independent task evaluator.
        for goal in env.env.parsed_problem["goal_state"]
    ]


def _environment(request, raw):
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    bddl = _text(raw["data"].attrs["bddl_file_name"])
    relative = bddl.split("bddl_files/", 1)[-1]
    path = Path(get_libero_path("bddl_files")) / relative
    env = ControlEnv(
        bddl_file_name=str(path),
        robots=["Panda"],
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
        hard_reset=False,
    )
    env.seed(request["seed"])
    env.reset()
    return env


def run(request, output):
    import mujoco
    import robosuite
    from robosuite.controllers import base_controller
    from robosuite.controllers import osc

    output.mkdir(parents=True, exist_ok=True)
    raw_file = Path(request["source_file"])
    stat = raw_file.stat()
    if stat.st_size != request["source_size"] or stat.st_mtime_ns != request["source_mtime_ns"]:
        raise ValueError("Raw demonstration changed after field construction")
    controller_sources = [
        Path(inspect.getfile(osc)),
        Path(inspect.getfile(base_controller)),
        Path(__file__),
        Path(inspect.getfile(OSCFieldAdapter)),
        Path(primitive.__file__),
        Path(primitive.seds.__file__),
    ]
    report = {
        "status": "running",
        "mode": request["mode"],
        "source_episode_index": request["source_episode_index"],
        "scope": "full dynamics; geometric field switching only; no semantic observer, learned policy, or real-time claim",
        "training_ready": False,
        "gripper_feedback_required": request.get("gripper_feedback", False),
        "seed": request["seed"],
        "max_steps": request["max_steps"],
        "environment": {
            "python": sys.version.split()[0],
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
        },
        "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in controller_sources},
        "sequence_results": [],
    }
    for source_path in controller_sources:
        destination = output / "source_snapshot" / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
    atomic_json(output / "result.json", report)
    started, env = time.monotonic(), None
    try:
        with h5py.File(raw_file) as raw:
            source = raw[request["source_group"]]
            xml = _relocate_xml(_text(source.attrs["model_file"]), raw_file)
            initial_state, source_actions = source["states"][0], source["actions"][:]
            env = _environment(request, raw)
            trials = request["sequences"] if request["mode"] == "fields" else [None]
            for trial_index, sequence in enumerate(trials):
                # Restore only before an entire sequence, never at a field boundary.
                env.reset_from_xml_string(xml)
                observation = env.set_init_state(initial_state)
                controller = env.robots[0].controller
                controller.update(force=True)
                controller.reset_goal()
                config = primitive.ControlConfig(**request["control"])
                if not np.isclose(env.env.control_timestep, config.dt):
                    raise ValueError("Actual controller frequency differs from field configuration")
                p, r, _, _ = actual_pose(env)
                c = env.sim.data.get_body_xmat("robot0_right_hand").T @ r
                if not np.allclose(c, request["body_to_control_rotation"], atol=1e-6):
                    raise ValueError("Source hand-to-control rotation differs from the full environment")
                if not np.allclose(p, request["source_initial_position"], atol=1e-5):
                    raise ValueError("Restored full-environment control point differs from the source initial state")
                report["controller"] = {
                    "control_dt": float(env.env.control_timestep),
                    "physics_dt": float(env.sim.model.opt.timestep),
                    "kp": np.asarray(controller.kp).tolist(),
                    "damping_ratio": (np.asarray(controller.kd) / (2 * np.sqrt(controller.kp))).tolist(),
                    "input_min": np.asarray(controller.input_min).tolist(),
                    "input_max": np.asarray(controller.input_max).tolist(),
                    "output_min": np.asarray(controller.output_min).tolist(),
                    "output_max": np.asarray(controller.output_max).tolist(),
                }
                adapter = OSCFieldAdapter(controller, config) if sequence is not None else None
                trial_result = {
                    "trial": trial_index,
                    "completed_fields": 0,
                    "status": "running",
                    "task_success": False,
                    "control_steps": 0,
                    "field_results": [],
                    "intermediate_resets": 0,
                }
                target = output / f"trial_{trial_index}.hdf5"
                temporary = target.with_name(target.name + ".tmp")
                with h5py.File(temporary, "w") as trajectory:
                    trajectory.attrs["scope"] = report["scope"]
                    trajectory.attrs["training_ready"] = False
                    log, queries = [], trajectory.create_group("queries")
                    object_names = sorted(env.env.objects_dict)
                    object_poses, bilateral_contacts, predicate_values = [], [], []
                    predicates = [list(goal) for goal in env.env.parsed_problem["goal_state"]]
                    audit_queries = trajectory.create_group("audit_observations")
                    audit_queries.attrs["not_policy_input"] = True
                    audit_queries.attrs["sampling"] = "fixed stride after action; separate from activation queries"

                    def record_audit(
                        step_observation,
                        control_step,
                        names=object_names,
                        pose_log=object_poses,
                        contact_log=bilateral_contacts,
                        predicate_log=predicate_values,
                        image_log=audit_queries,
                    ):
                        poses, bilateral = privileged_audit(env, names)
                        pose_log.append(poses)
                        contact_log.append(bilateral)
                        predicate_log.append([item["value"] for item in goal_predicates(env)])
                        stride = request.get("audit_stride", 0)
                        if stride and control_step % stride == 0:
                            capture_query(
                                image_log.create_group(str(control_step)),
                                step_observation,
                                env,
                                control_step,
                                before_activation=False,
                            )

                    if sequence is None:
                        for step, action in enumerate(source_actions[: request["max_steps"]]):
                            observation, _, _, _ = env.step(action)
                            p, r, v, omega = actual_pose(env)
                            log.append(
                                np.r_[
                                    step + 1,
                                    -1,
                                    env.sim.data.time,
                                    p,
                                    Rotation.from_matrix(r).as_quat(),
                                    v,
                                    omega,
                                    action,
                                ]
                            )
                            record_audit(observation, int(log[-1][0]))
                            trial_result["control_steps"] = step + 1
                        trial_result["status"] = (
                            "complete" if len(source_actions) <= request["max_steps"] else "task_budget_exhausted"
                        )
                    else:
                        step, previous_command = 0, None
                        for field_index, record in enumerate(sequence):
                            current = primitive.Primitive.from_record(record)
                            capture_query(queries.create_group(str(field_index)), observation, env, step)
                            arrival = primitive.Arrival()
                            activated_at = step
                            field_result = {
                                "field_index": field_index,
                                "activation_step": step,
                                "command": current.command,
                                "command_changed": current.command != previous_command,
                                "kind": current.kind,
                            }
                            feedback = primitive.CommandFeedback(observation["robot0_gripper_qpos"])
                            require_feedback = (
                                request.get("gripper_feedback", False) and current.command != previous_command
                            )
                            feedback_settled = not require_feedback
                            previous_command = current.command
                            complete = False
                            while step < request["max_steps"]:
                                p, r, _, _ = actual_pose(env)
                                action, expected = adapter.action(current, p, r)
                                observation, _, _, _ = env.step(action)
                                position_error, rotation_error = adapter.verify_goal(expected)
                                p, r, v, omega = actual_pose(env)
                                step += 1
                                log.append(
                                    np.r_[
                                        step,
                                        field_index,
                                        env.sim.data.time,
                                        p,
                                        Rotation.from_matrix(r).as_quat(),
                                        v,
                                        omega,
                                        action,
                                    ]
                                )
                                if require_feedback:
                                    feedback_settled = feedback.update(observation["robot0_gripper_qpos"], config.dt)
                                complete = arrival.update(current, p, r, v, omega, config) and feedback_settled
                                record_audit(observation, int(log[-1][0]))
                                if complete:
                                    break
                            field_result.update(
                                completed=complete,
                                control_steps=step - activated_at,
                                exit_position_error_m=float(np.linalg.norm(p - current.field.goal)),
                                exit_rotation_error_rad=float(Rotation.from_matrix(current.rotation @ r.T).magnitude()),
                                last_controller_position_error_m=position_error,
                                last_controller_rotation_element_error=rotation_error,
                                semantic_handoff_verified=False,
                                command_feedback_settled=feedback_settled,
                                command_feedback_required=require_feedback,
                            )
                            trial_result["field_results"].append(field_result)
                            trial_result["completed_fields"] += int(complete)
                            if not complete:
                                break
                        trial_result["control_steps"] = step
                        trial_result["status"] = (
                            "geometric_sequence_complete"
                            if trial_result["completed_fields"] == len(sequence)
                            else "task_budget_exhausted"
                        )
                    trial_result["task_success"] = bool(env.check_success())
                    trial_result["final_goal_predicates"] = goal_predicates(env)
                    capture_query(queries.create_group("final"), observation, env, trial_result["control_steps"])
                    trajectory.create_dataset("control_log", data=np.array(log), compression="gzip", compression_opts=1)
                    trajectory.attrs["control_log_columns"] = json.dumps(
                        [
                            "step",
                            "field_index",
                            "simulation_time",
                            "px",
                            "py",
                            "pz",
                            "qx",
                            "qy",
                            "qz",
                            "qw",
                            "vx",
                            "vy",
                            "vz",
                            "wx",
                            "wy",
                            "wz",
                            "a0",
                            "a1",
                            "a2",
                            "a3",
                            "a4",
                            "a5",
                            "gripper",
                        ]
                    )
                    audit = trajectory.create_group("privileged_audit")
                    audit.attrs["schema"] = "vlds.libero.privileged_audit.v2"
                    audit.attrs["not_policy_input"] = True
                    audit.attrs["bilateral_contact_is_not_held"] = True
                    audit.attrs["object_names"] = json.dumps(object_names)
                    audit.attrs["pose_columns"] = "px,py,pz,qx,qy,qz,qw"
                    audit.attrs["goal_predicates"] = json.dumps(predicates)
                    audit.create_dataset("goal_predicate_values", data=np.array(predicate_values), compression="gzip")
                    audit.create_dataset("object_pose", data=np.array(object_poses), compression="gzip")
                    audit.create_dataset(
                        "bilateral_finger_contact", data=np.array(bilateral_contacts), compression="gzip"
                    )
                    trajectory.attrs["result"] = json.dumps(trial_result)
                os.replace(temporary, target)
                trial_result["trajectory_file"] = target.name
                report["sequence_results"].append(trial_result)
                atomic_json(output / "result.json", report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "result.json", report)
        if env is not None:
            env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.request.read_text()), args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

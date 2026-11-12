"""Render IMU attachment positions and sensor frames on a synthesized SMPL pose."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

# The synthesis snapshot uses the existing MobilePoser SMPL implementation.
try:
    import mobileposer
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "waist_streamposer_posacc" / "training"))

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths

# Always use this synthesis snapshot's attachment implementation and configuration.
_spec = importlib.util.spec_from_file_location("synthesis_surface_imu", Path(__file__).with_name("surface_imu.py"))
_surface = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_surface)
attachment_points = _surface.attachment_points
load_attachments = _surface.load_attachments
DEFAULT_CONFIG = Path(__file__).with_name("wristankle_surface_imu.json")


LABELS = ("lw", "rw", "ls", "rs", "waist")
LABEL_TEXT = {
    "lw": "Left wrist",
    "rw": "Right wrist",
    "ls": "Left ankle",
    "rs": "Right ankle",
    "waist": "Waist",
}
COLORS = {
    "lw": "#d95f02",
    "rw": "#1b9e77",
    "ls": "#7570b3",
    "rs": "#e7298a",
    "waist": "#e6ab02",
}
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def render_interactive(mesh, faces, positions, config, output):
    """Offline, orbitable placement inspection using unmodified SMPL world coordinates."""
    import plotly.graph_objects as go

    names = ("左腕 lw", "右腕 rw", "左踝 ls", "右踝 rs", "前腰 waist")
    fig = go.Figure(go.Mesh3d(
        x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#becbd3", opacity=1, name="SMPL 表面", hoverinfo="skip",
        flatshading=False, lighting=dict(ambient=.6, diffuse=.8), showlegend=True,
    ))
    for label, name, point in zip(LABELS, names, positions):
        a = config["attachments"][label]
        fig.add_trace(go.Scatter3d(
            x=[float(point[0])], y=[float(point[1])], z=[float(point[2])],
            mode="markers+text", text=[name], textposition="top center", name=name,
            marker=dict(size=6, color=COLORS[label], line=dict(color="white", width=1)),
            hovertemplate=(f"{name}<br>X=%{{x:.5f}} m<br>Y=%{{y:.5f}} m<br>Z=%{{z:.5f}} m"
                           f"<br>face={a['face']}<br>骨骼 joint={a['joint']}<extra></extra>"),
        ))
    # Optional surface normals describe the skin direction, not calibrated IMU axes.
    normals = np.asarray([config["attachments"][label]["mount_rotation"] for label in LABELS])[:, :, 2]
    xs, ys, zs = [], [], []
    for p, normal in zip(positions, normals):
        end = p + .10 * normal
        xs.extend([float(p[0]), float(end[0]), None])
        ys.extend([float(p[1]), float(end[1]), None])
        zs.extend([float(p[2]), float(end[2]), None])
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#333333", width=5),
                              name="表面朝外方向（非 IMU 轴）", visible="legendonly", hoverinfo="skip"))

    def camera(eye):
        return dict(eye=dict(zip("xyz", eye)), up=dict(x=0, y=1, z=0), projection=dict(type="orthographic"))

    views = [("正面", (0, .1, 2.1)), ("背面", (0, .1, -2.1)),
             ("人体左侧", (2.1, .1, 0)), ("人体右侧", (-2.1, .1, 0))]
    buttons = [dict(label=name, method="relayout", args=[{"scene.camera": camera(eye)}]) for name, eye in views]
    buttons.append(dict(label="俯视", method="relayout", args=[{"scene.camera": dict(
        eye=dict(x=0, y=2.1, z=.01), up=dict(x=0, y=0, z=-1), projection=dict(type="orthographic"))}]))
    ranges = [dict(label="全身", method="relayout", args=[{
        f"scene.{axis}axis.autorange": True for axis in "xyz"}])]
    for name, idx in (("左腕近看", 0), ("右腕近看", 1), ("左踝近看", 2), ("右踝近看", 3), ("前腰近看", 4)):
        p = positions[idx]
        ranges.append(dict(label=name, method="relayout", args=[{
            f"scene.{axis}axis.range": [float(p[j] - .20), float(p[j] + .20)] for j, axis in enumerate("xyz")
        }]))
    fig.update_layout(
        title="SMPL T-pose：最新合成配置的五个真实表面挂点",
        scene=dict(aspectmode="data", dragmode="orbit", camera=camera((1.1, .5, 2)),
                   xaxis=dict(title="SMPL X / m"), yaxis=dict(title="SMPL Y 向上 / m"),
                   zaxis=dict(title="SMPL Z 向前 / m")),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.10, buttons=buttons),
                     dict(type="dropdown", x=0, y=1.02, buttons=ranges)],
        margin=dict(l=0, r=0, t=160, b=80), paper_bgcolor="#f5f7fa",
        annotations=[dict(x=0, y=-.06, xref="paper", yref="paper", showarrow=False, align="left",
                          text="左键拖动旋转 · 滚轮缩放 · 右键拖动平移 · 双击恢复<br>"
                               "点位按冻结三角面和重心权重计算，未人为外移；点击图例可隐藏模型、查看被遮挡点。<br>"
                               "检查位置请结合正面/背面/俯视及近看菜单；最新配置：wristankle_surface_imu.json")],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, full_html=True,
                   config=dict(scrollZoom=True, responsive=True, displaylogo=False))
    print(f"Interactive viewer: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, help="optional synthesized dataset .pt; omit to render a T-pose")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequence", type=int, default=0, help="sequence index in the dataset")
    parser.add_argument("--frame", type=int, default=0, help="frame index in the selected sequence")
    parser.add_argument("--output", type=Path, default=None, help="output image; defaults beside the dataset")
    parser.add_argument("--axis-length", type=float, default=0.14, help="sensor-axis length in meters")
    parser.add_argument("--frame-type", choices=("calibrated", "mounted"), default="calibrated",
                        help="calibrated: actual synthesis ori; mounted: bone rotation @ mount_rotation")
    args = parser.parse_args()

    if args.axis_length <= 0:
        raise ValueError("axis-length must be positive")

    config = load_attachments(args.config)
    model = ParametricModel(paths.smpl_file)
    if args.data:
        data_path = args.data
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        if data.get("synthesis", {}).get("mode") != "surface":
            raise ValueError(f"Expected surface synthesis data, got {data.get('synthesis', {}).get('mode')!r}")
        if tuple(data["layout_labels"]) != LABELS:
            raise ValueError(f"Expected sensor order {LABELS}, got {data['layout_labels']}")
        if not 0 <= args.sequence < len(data["pose"]):
            raise IndexError(f"sequence must be in [0, {len(data['pose']) - 1}]")
        pose_seq = data["pose"][args.sequence]
        if not 0 <= args.frame < len(pose_seq):
            raise IndexError(f"frame must be in [0, {len(pose_seq) - 1}]")
        pose = pose_seq[args.frame:args.frame + 1]
        shape = data["shape"][args.sequence]
        tran = data["tran"][args.sequence][args.frame:args.frame + 1]
        rotations, _, vertices = model.forward_kinematics(pose, shape, tran, calc_mesh=True)
        sensor_frames = data["ori"][args.sequence][args.frame, :len(LABELS)].numpy()
        title = f"{data_path.name} | sequence {args.sequence}, frame {args.frame}"
        default_output = data_path.parent / f"{data_path.stem}_seq{args.sequence}_frame{args.frame}_imu3d.png"
    else:
        # This SMPL model's zero pose already is the T-pose.
        pose = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 24, 1, 1)
        shape = torch.zeros(10)
        tran = torch.zeros(1, 3)
        rotations, _, vertices = model.forward_kinematics(pose, shape, tran, calc_mesh=True)
        sensor_bones = [config["attachments"][label]["joint"] for label in LABELS]
        sensor_frames = rotations[0, sensor_bones].numpy()
        title = "Five-IMU placement and local frames | SMPL T-pose"
        default_output = Path(__file__).parent / "imu_3d_tpose.png"

    sensor_bones = [config["attachments"][label]["joint"] for label in LABELS]
    bone_frames = rotations[0, sensor_bones].numpy()
    mounts = np.asarray([config["attachments"][label]["mount_rotation"] for label in LABELS])
    # Same expression as surface_kinematics: ori = (R_bone @ M) @ M.T.
    expected_calibrated = (bone_frames @ mounts) @ mounts.transpose(0, 2, 1)
    np.testing.assert_allclose(sensor_frames, expected_calibrated, atol=1e-5)
    if args.frame_type == "mounted":
        sensor_frames = bone_frames @ mounts
        if not args.output:
            default_output = default_output.with_stem(default_output.stem + "_mounted")
    if not args.data:
        np.testing.assert_allclose(bone_frames, np.broadcast_to(np.eye(3), bone_frames.shape), atol=1e-6)
        neutral = np.asarray([config["attachments"][label]["neutral_position_m"] for label in LABELS])
        np.testing.assert_allclose(attachment_points(vertices, config, list(LABELS))[0].numpy(), neutral, atol=1e-5)
    title += "\n" + ("Calibrated synthesis ori = bone frame" if args.frame_type == "calibrated"
                            else "Mounted device frame = bone frame @ mount_rotation")

    labels = list(LABELS)
    positions = attachment_points(vertices, config, labels)[0].numpy()
    mesh = vertices[0].numpy()
    faces = np.asarray(model.face)
    if not args.data:
        render_interactive(mesh, faces, positions, config, Path(__file__).with_name("imu_placement_3d.html"))

    # Right-handed display transform: (X, Y, Z) -> (X, -Z, Y).
    def plot_xyz(points):
        return np.asarray(points)[..., [0, 2, 1]] * np.array([1., -1., 1.])

    plot_positions = plot_xyz(positions)
    plot_mesh = plot_xyz(mesh)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 10), facecolor="#f5f7fa")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f5f7fa")
    triangles = plot_mesh[faces]
    surface = Poly3DCollection(triangles, facecolors="#c9d3d8", edgecolors="none", alpha=0.65)
    ax.add_collection3d(surface)

    for i, label in enumerate(labels):
        point = plot_positions[i]
        color = COLORS[label]
        ax.scatter(*point, s=72, color=color, edgecolors="white", linewidths=1.1, depthshade=False)
        ax.text(*(point + np.array([0.025, 0.025, 0.025])), LABEL_TEXT[label], color=color, fontsize=9)
        # ori stores local-to-world sensor rotations; columns are the displayed local axes.
        for axis_idx, axis_color in enumerate(AXIS_COLORS):
            direction = plot_xyz(sensor_frames[i, :, axis_idx]) * args.axis_length
            ax.quiver(*point, *direction, color=axis_color, linewidth=1.8, arrow_length_ratio=0.18)

    center = plot_mesh.mean(axis=0)
    extent = np.max(np.ptp(plot_mesh, axis=0)) * 0.58
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("-Z (m)")
    ax.set_zlabel("Y (m)")
    ax.view_init(elev=14, azim=66)
    ax.set_title(title, pad=18)

    sensor_handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[k], label=LABEL_TEXT[k]) for k in labels]
    axis_handles = [plt.Line2D([], [], color=AXIS_COLORS[i], label=f"local {name}") for i, name in enumerate("XYZ")]
    ax.legend(handles=sensor_handles + axis_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False)
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if args.data:
        print(f"Dataset: {data_path}; layout: {data['layout']}; frame: sequence {args.sequence}, frame {args.frame}")
    else:
        print(f"Pose: neutral SMPL shape, zero-pose T-pose; sensors: {', '.join(labels)}")
    print(f"Frame type: {args.frame_type}; config: {args.config}")
    print("IMU X axes (SMPL world XYZ):", sensor_frames[:, :, 0].tolist())
    output.with_suffix(".json").write_text(json.dumps({
        "config": str(args.config.resolve()), "frame_type": args.frame_type,
        "labels": labels, "positions_smpl_world": positions.tolist(),
        "rotation_columns_smpl_world": sensor_frames.tolist(),
    }, indent=2), encoding="utf-8")
    print(f"Image: {output}")


if __name__ == "__main__":
    main()

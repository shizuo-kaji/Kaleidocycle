"""Static figures and an interactive notebook view of the certified example."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .collisions import CrossingEvolution

EDGE_COLOURS = ("#187bcd", "#d45d22")


def plot_crossing_diagnostics(evolution: CrossingEvolution) -> Figure:
    """Plot edge separation and topology, with a gap at the undefined contact."""
    fig = Figure(figsize=(10, 3.8), layout="constrained")
    distance, topology = fig.subplots(1, 2)
    time = evolution.times * 1000
    distance.plot(
        time,
        evolution.min_distance,
        color="#187bcd",
        label="Minimum over all non-adjacent edges",
    )
    distance.set(
        xlabel="Time from contact (× 10⁻³)",
        ylabel="Distance (edge length = 1)",
        title="A finite-time contact",
    )
    topology.plot(time, evolution.writhe, color="#187bcd", label="Gauss Wr")
    topology.plot(
        time, evolution.linking, color="#d45d22", linestyle="--", label="Lk = Tw + Wr"
    )
    topology.set(
        xlabel="Time from contact (× 10⁻³)",
        ylabel="Wr / Lk",
        title="The jump is not a smooth drift",
    )
    topology.legend(frameon=False)
    for ax in (distance, topology):
        ax.axvline(0, color="0.55", lw=1, linestyle=":")
        ax.grid(alpha=0.15)
    fig.suptitle("K12 · μ = 1.2 rad · anti-oriented · first mKdV flow")
    return fig


def plot_crossing_shapes(evolution: CrossingEvolution) -> Figure:
    """Three static 3D snapshots, retained even without a widget-capable viewer."""
    fig = Figure(figsize=(11, 4.2), layout="constrained")
    middle = int(np.argmin(abs(evolution.times)))
    all_points = evolution.vertices.reshape(-1, 3)
    centre = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    radius = np.ptp(all_points, axis=0).max() * 0.55
    for position, index in enumerate([0, middle, len(evolution.times) - 1], 1):
        ax = fig.add_subplot(1, 3, position, projection="3d")
        points = evolution.vertices[index]
        ax.plot(*points.T, color="0.55", lw=1.2)
        for edge, colour in zip((0, 6), EDGE_COLOURS, strict=True):
            ax.plot(
                *points[edge : edge + 2].T, color=colour, lw=3, label=f"Edge {edge}"
            )
        ax.set(
            xlim=(centre[0] - radius, centre[0] + radius),
            ylim=(centre[1] - radius, centre[1] + radius),
            zlim=(centre[2] - radius, centre[2] + radius),
            title=f"t = {evolution.times[index]:+.4f}",
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-55)
        ax.set_axis_off()
        if position == 3:
            ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Whole curve · identical scale · edge length = 1")
    return fig


def plot_crossing_detail(evolution: CrossingEvolution) -> Figure:
    """Orthographic close-ups in the contact normal plane; no exaggerated motion."""
    middle = int(np.argmin(abs(evolution.times)))
    points = evolution.vertices[middle]
    t0, t6 = points[1] - points[0], points[7] - points[6]
    normal = np.cross(t0, t6)
    normal /= np.linalg.norm(normal)
    horizontal = t0 - t6
    horizontal /= np.linalg.norm(horizontal)
    origin = points[0] + 0.8214635512039877 * t0
    fig = Figure(figsize=(11, 3), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, index in zip(axes, (0, middle, len(evolution.times) - 1), strict=True):
        for edge, colour in zip((0, 6), EDGE_COLOURS, strict=True):
            segment = evolution.vertices[index, edge : edge + 2] - origin
            ax.plot(
                segment @ horizontal,
                segment @ normal,
                color=colour,
                lw=2,
                label=f"Edge {edge}",
                linestyle="-" if edge == 0 else "--",
            )
        ax.set(
            xlim=(-0.04, 0.04),
            ylim=(-0.008, 0.008),
            xlabel="In-plane projection (edge units)",
            title=f"t = {evolution.times[index]:+.4f}",
        )
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Contact normal coordinate")
    axes[-1].legend(frameon=False)
    fig.suptitle("Orthographic close-up: the strands exchange sides")
    return fig


def create_crossing_viewer(evolution: CrossingEvolution) -> Any:
    """Create an ipympl 3D view with a time slider (use %matplotlib widget)."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    with plt.ioff():
        fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    points = evolution.vertices[0]
    (line,) = ax.plot(*points.T, color="0.5", lw=1.5)
    highlights = [
        ax.plot(*points[e : e + 2].T, color=c, lw=4, label=f"Edge {e}")[0]
        for e, c in zip((0, 6), EDGE_COLOURS, strict=True)
    ]
    centre = points.mean(axis=0)
    radius = np.ptp(evolution.vertices.reshape(-1, 3), axis=0).max() * 0.55
    ax.set(
        xlim=(centre[0] - radius, centre[0] + radius),
        ylim=(centre[1] - radius, centre[1] + radius),
        zlim=(centre[2] - radius, centre[2] + radius),
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(evolution.times) - 1,
        description="Time sample",
        continuous_update=True,
        layout=widgets.Layout(width="95%"),
    )
    readout = widgets.HTML()
    contact = widgets.Button(description="Go to contact")
    view = widgets.ToggleButtons(options=["Whole curve", "Crossing close-up"])

    def change_view(change: Any = None) -> None:
        focus = np.array([0.8214635512039877, 0, 0])
        point = focus if view.value == "Crossing close-up" else centre
        extent = 0.05 if view.value == "Crossing close-up" else radius
        ax.set(
            xlim=(point[0] - extent, point[0] + extent),
            ylim=(point[1] - extent, point[1] + extent),
            zlim=(point[2] - extent, point[2] + extent),
        )
        fig.canvas.draw_idle()

    view.observe(change_view, names="value")

    def update(change: Any = None) -> None:
        index = slider.value
        points = evolution.vertices[index]
        line.set_data_3d(*points.T)
        for edge, artist in zip((0, 6), highlights, strict=True):
            artist.set_data_3d(*points[edge : edge + 2].T)
        wr = evolution.writhe[index]
        topology = (
            "Wr and Lk undefined at contact"
            if np.isnan(wr)
            else f"Wr = {wr:.8f}, Lk = {evolution.linking[index]:.1f}"
        )
        readout.value = (
            f"<b>t = {evolution.times[index]:+.6f}</b> · minimum distance = "
            f"{evolution.min_distance[index]:.6g}<br>{topology}"
        )
        fig.canvas.draw_idle()

    slider.observe(update, names="value")
    contact.on_click(
        lambda _: setattr(slider, "value", int(np.argmin(abs(evolution.times))))
    )
    update()
    return widgets.VBox([slider, widgets.HBox([contact, view]), readout, fig.canvas])

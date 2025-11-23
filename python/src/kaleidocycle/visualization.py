"""Visualization utilities for kaleidocycles including 3D plotting and animation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .geometry import Kaleidocycle


def compute_tetrahedron_vertices(
    curve: np.ndarray,
    hinges: np.ndarray,
    width: float = 0.15,
) -> tuple[np.ndarray, list[list[int]]]:
    """Compute vertices and faces for the tetrahedral band structure.

    Parameters
    ----------
    curve:
        (n, 3) array of curve points.
    hinges:
        (n, 3) array of hinge vectors (should have same length as curve).
    width:
        Half-width of the tetrahedra along hinge directions.

    Returns
    -------
    vertices:
        (2*n, 3) array of all vertices.
    faces:
        List of faces, where each face is a list of vertex indices.
    """
    pts = np.asarray(curve)
    h = np.asarray(hinges)

    n = pts.shape[0]
    if h.shape[0] < n:
        # Pad hinges if needed (for closed structures)
        h = np.vstack([h, h[0:1]])

    # Create vertices: each curve point gets two vertices along the hinge
    vertices = np.zeros((2 * n, 3), dtype=float)
    for i in range(n):
        hinge_norm = h[i] / (np.linalg.norm(h[i]) + 1e-10)
        vertices[2 * i] = pts[i] + width * hinge_norm
        vertices[2 * i + 1] = pts[i] - width * hinge_norm

    # Create faces for each tetrahedral segment
    faces = []
    for i in range(n - 1):
        # Vertices for this segment
        v0 = 2 * i  # top at curve[i]
        v1 = 2 * i + 1  # bottom at curve[i]
        v2 = 2 * (i + 1)  # top at curve[i+1]
        v3 = 2 * (i + 1) + 1  # bottom at curve[i+1]

        # Four triangular faces of the tetrahedron
        faces.append([v0, v2, v1])  # Top-right triangle
        faces.append([v1, v2, v3])  # Bottom-right triangle
        faces.append([v0, v1, v3])  # Left triangle
        faces.append([v0, v3, v2])  # Back triangle

    return vertices, faces


def plot_curve(
    curve: np.ndarray,
    ax: Axes | None = None,
    *,
    color: str = "steelblue",
    linewidth: float = 2.0,
    marker: str = "o",
    markersize: float = 6.0,
    title: str | None = None,
    show_axes: bool = True,
) -> Axes:
    """Plot a 3D curve.

    Parameters
    ----------
    curve:
        (n, 3) array of 3D points forming the curve.
    ax:
        Matplotlib 3D axes. If None, creates a new figure.
    color:
        Line and marker color.
    linewidth:
        Width of the connecting line.
    marker:
        Marker style for points.
    markersize:
        Size of markers.
    title:
        Optional title for the plot.
    show_axes:
        Whether to display axis labels and grid.

    Returns
    -------
    ax:
        The matplotlib 3D axes object.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(curve)
    if pts.ndim != 2 or pts.shape[1] != 3:
        msg = f"expected (n, 3) curve array, got shape {pts.shape}"
        raise ValueError(msg)

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        "-",
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    if show_axes:
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()

    # Set equal aspect ratio
    _set_equal_aspect_3d(ax, pts)

    return ax


def plot_hinges(
    curve: np.ndarray,
    hinges: np.ndarray,
    ax: Axes | None = None,
    *,
    scale: float = 0.3,
    color: str = "crimson",
    alpha: float = 0.7,
    title: str | None = None,
) -> Axes:
    """Plot hinge vectors as arrows emanating from curve points.

    Parameters
    ----------
    curve:
        (n, 3) array of 3D points.
    hinges:
        (n, 3) array of unit hinge vectors.
    ax:
        Matplotlib 3D axes. If None, creates a new figure.
    scale:
        Length scaling factor for the arrows.
    color:
        Arrow color.
    alpha:
        Arrow transparency.
    title:
        Optional title for the plot.

    Returns
    -------
    ax:
        The matplotlib 3D axes object.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(curve)
    h = np.asarray(hinges)

    if pts.ndim != 2 or pts.shape[1] != 3:
        msg = f"expected (n, 3) curve array, got shape {pts.shape}"
        raise ValueError(msg)
    if h.ndim != 2 or h.shape[1] != 3:
        msg = f"expected (n, 3) hinges array, got shape {h.shape}"
        raise ValueError(msg)

    # Handle different array lengths (hinges may be n+1 for closed cycle)
    n_arrows = min(pts.shape[0], h.shape[0])

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    # Plot curve first
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        "-o",
        color="steelblue",
        linewidth=1.5,
        markersize=4,
        alpha=0.6,
    )

    # Plot hinge arrows
    for i in range(n_arrows):
        ax.quiver(
            pts[i, 0],
            pts[i, 1],
            pts[i, 2],
            h[i, 0] * scale,
            h[i, 1] * scale,
            h[i, 2] * scale,
            color=color,
            alpha=alpha,
            arrow_length_ratio=0.3,
            linewidth=2,
        )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    all_pts = np.vstack([pts[:n_arrows], pts[:n_arrows] + h[:n_arrows] * scale])
    _set_equal_aspect_3d(ax, all_pts)

    return ax


def plot_tetrahedron(
    curve: np.ndarray,
    hinges: np.ndarray,
    ax: Axes | None = None,
    *,
    width: float = 0.15,
    facecolor: str = "lightblue",
    edgecolor: str = "darkblue",
    alpha: float = 0.7,
    linewidth: float = 0.5,
    title: str | None = None,
    show_curve: bool = False,
) -> Axes:
    """Plot the tetrahedral structure of a kaleidocycle using triangular faces.

    This creates a solid 3D visualization showing the actual tetrahedral
    units that make up the kaleidocycle structure, similar to the visTet
    function in Maple.

    Parameters
    ----------
    curve:
        (n, 3) array of curve points.
    hinges:
        (n, 3) array of hinge vectors.
    ax:
        Matplotlib 3D axes. If None, creates a new figure.
    width:
        Half-width of the tetrahedra along hinge directions.
    facecolor:
        Color for the tetrahedral faces.
    edgecolor:
        Color for the edges.
    alpha:
        Transparency of the faces (0=transparent, 1=opaque).
    linewidth:
        Width of the edges.
    title:
        Optional title for the plot.
    show_curve:
        Whether to overlay the curve.

    Returns
    -------
    ax:
        The matplotlib 3D axes object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pts = np.asarray(curve)
    h = np.asarray(hinges)

    if pts.ndim != 2 or pts.shape[1] != 3:
        msg = f"expected (n, 3) curve array, got shape {pts.shape}"
        raise ValueError(msg)
    if h.ndim != 2 or h.shape[1] != 3:
        msg = f"expected (n, 3) hinges array, got shape {h.shape}"
        raise ValueError(msg)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Compute tetrahedral vertices and faces
    vertices, faces = compute_tetrahedron_vertices(pts, h, width=width)

    # Create polygon collection
    poly_vertices = [[vertices[idx] for idx in face] for face in faces]
    poly_collection = Poly3DCollection(
        poly_vertices,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly_collection)

    # Optionally show curve
    if show_curve:
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            "-",
            color="crimson",
            linewidth=2,
            alpha=0.8,
        )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    _set_equal_aspect_3d(ax, vertices)

    return ax


def _create_band_animation(
    frames_curves: list[np.ndarray],
    frames_hinges: list[np.ndarray],
    *,
    width: float = 0.15,
    facecolor: str = "lightblue",
    edgecolor: str = "darkblue",
    alpha: float = 0.7,
    linewidth: float = 0.5,
    title: str | None = None,
    scalar_properties: list[str] | None = None,
    animation_obj: 'KaleidocycleAnimation | None' = None,
    show_curve: bool = False,
    untwist: bool = False,
    interval: int = 100,
    figsize: tuple[float, float] = (10, 8),
) -> tuple['Figure', object]:
    """Create animation of kaleidocycle band.

    Helper function for plot_band to create animations.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation as mpl_animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n_frames = len(frames_curves)

    # Precompute all vertices for each frame
    all_vertices = []
    all_quad_faces = []
    for i in range(n_frames):
        pts = frames_curves[i]
        h = frames_hinges[i]
        vertices, _ = compute_tetrahedron_vertices(pts, h, width=width)
        all_vertices.append(vertices)

        # Create quadrilateral faces for this frame
        n_curve_points = len(pts)
        k = n_curve_points - 1
        quad_faces = []
        for j in range(k):
            if untwist:
                quad = [
                    vertices[2 * j],
                    vertices[2 * j + 1],
                    vertices[2 * (j + 1)],
                    vertices[2 * (j + 1) + 1]
                ]
            else:
                quad = [
                    vertices[2 * j],
                    vertices[2 * j + 1],
                    vertices[2 * (j + 1) + 1],
                    vertices[2 * (j + 1)]
                ]
            quad_faces.append(quad)
        all_quad_faces.append(quad_faces)

    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Determine global bounds for consistent axis limits
    all_verts_flat = np.vstack([v for v in all_vertices])
    x_min, x_max = all_verts_flat[:, 0].min(), all_verts_flat[:, 0].max()
    y_min, y_max = all_verts_flat[:, 1].min(), all_verts_flat[:, 1].max()
    z_min, z_max = all_verts_flat[:, 2].min(), all_verts_flat[:, 2].max()

    # Add padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    z_pad = (z_max - z_min) * 0.1

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Try to set equal aspect ratio
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    # Initialize collections
    poly_collection = Poly3DCollection(
        all_quad_faces[0],
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly_collection)

    # Optional curve line
    curve_line = None
    if show_curve:
        pts = frames_curves[0]
        curve_line, = ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            "-",
            color="crimson",
            linewidth=2,
            alpha=0.8,
        )

    # Title text
    title_text = ax.set_title("", fontsize=12, fontweight="bold")

    def init():
        """Initialize animation."""
        return (poly_collection, title_text) if curve_line is None else (poly_collection, curve_line, title_text)

    def update(frame_idx):
        """Update animation frame."""
        # Update polygon collection
        poly_collection.set_verts(all_quad_faces[frame_idx])

        # Update curve line if shown
        if curve_line is not None:
            pts = frames_curves[frame_idx]
            curve_line.set_data(pts[:, 0], pts[:, 1])
            curve_line.set_3d_properties(pts[:, 2])

        # Update title with scalar properties
        title_str = title if title else f"Frame {frame_idx + 1}/{n_frames}"

        if scalar_properties and animation_obj is not None:
            # Add scalar properties to title
            prop_strs = []
            for prop_name in scalar_properties:
                if prop_name in animation_obj.scalar_properties:
                    value = animation_obj.scalar_properties[prop_name][frame_idx]
                    prop_strs.append(f"{prop_name}={value:.4g}")

            if prop_strs:
                title_str += " | " + ", ".join(prop_strs)

        title_text.set_text(title_str)

        return (poly_collection, title_text) if curve_line is None else (poly_collection, curve_line, title_text)

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=interval,
        blit=False,  # blit=False is needed for 3D animations
    )

    return fig, anim


def plot_band(
    curve: np.ndarray | list | None = None,
    hinges: np.ndarray | list | None = None,
    ax: Axes | None = None,
    *,
    animation: 'KaleidocycleAnimation | None' = None,
    width: float = 0.15,
    facecolor: str = "lightblue",
    edgecolor: str = "darkblue",
    alpha: float = 0.7,
    linewidth: float = 0.5,
    title: str | None = None,
    scalar_properties: list[str] | None = None,
    show_curve: bool = False,
    untwist: bool = False,
    interval: int = 100,
    figsize: tuple[float, float] = (10, 8),
) -> Axes | tuple['Figure', object]:
    """Plot the kaleidocycle as a band of non-planar quadrilaterals.

    This creates a 3D visualization showing the kaleidocycle as a collection
    of quadrilateral faces connecting the vertices. Can plot static frames
    or create animations from KaleidocycleAnimation or list of binormals.

    Parameters
    ----------
    curve:
        (n, 3) array of curve points. If None and animation is provided,
        curves are computed from animation frames. Can also be a list of
        curves for animation.
    hinges:
        (n, 3) array of hinge vectors, or list of hinge arrays for animation.
        If animation is provided, this is ignored.
    ax:
        Matplotlib 3D axes. If None, creates a new figure. Ignored for animations.
    animation:
        KaleidocycleAnimation instance for creating animations.
    width:
        Half-width of the band along hinge directions.
    facecolor:
        Color for the quadrilateral faces.
    edgecolor:
        Color for the edges.
    alpha:
        Transparency of the faces (0=transparent, 1=opaque).
    linewidth:
        Width of the edges.
    title:
        Optional title for the plot. For animations with scalar_properties,
        this is used as a base title.
    scalar_properties:
        List of scalar property names to display in animation title
        (e.g., ["penalty", "energy"]). Only used for animations.
    show_curve:
        Whether to overlay the curve.
    untwist:
        If True, use alternative vertex ordering for the quadrilaterals.
    interval:
        Delay between animation frames in milliseconds (default 100).
    figsize:
        Figure size for animations (width, height) in inches.

    Returns
    -------
    ax or (fig, anim):
        For static plots: matplotlib 3D axes object.
        For animations: tuple of (Figure, Animation) object.

    References
    ----------
    Corresponds to visBand function in Maple code (line 332).

    Examples
    --------
    Static plot:
    >>> from kaleidocycle import random_hinges, binormals_to_tangents, tangents_to_curve, plot_band
    >>> import matplotlib.pyplot as plt
    >>> hinges = random_hinges(6, seed=42).as_array()
    >>> tangents = binormals_to_tangents(hinges, normalize=False)
    >>> curve = tangents_to_curve(tangents)
    >>> ax = plot_band(curve, hinges)
    >>> plt.show()

    Animation from KaleidocycleAnimation:
    >>> from kaleidocycle import Kaleidocycle, generate_animation, KaleidocycleAnimation, plot_band
    >>> kc = Kaleidocycle(6, seed=42)
    >>> frames = generate_animation(kc.hinges, num_frames=20)
    >>> anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine_gordon")
    >>> anim.compute_scalar_property("penalty")
    >>> fig, ani = plot_band(animation=anim, scalar_properties=["penalty"])
    >>> plt.show()

    Animation from list of binormals:
    >>> frames = generate_animation(kc.hinges, num_frames=20)
    >>> fig, ani = plot_band(hinges=frames)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Determine if this is an animation request
    is_animation = False
    frames_hinges = None
    frames_curves = None

    if animation is not None:
        # Animation from KaleidocycleAnimation
        is_animation = True
        frames_hinges = animation.frames
        # Compute curves from hinges
        from .geometry import binormals_to_tangents, tangents_to_curve
        frames_curves = []
        for hinges_frame in frames_hinges:
            tangents = binormals_to_tangents(hinges_frame, normalize=False)
            curve_frame = tangents_to_curve(tangents, center=True)
            frames_curves.append(curve_frame)
    elif isinstance(hinges, list):
        # Animation from list of binormals
        is_animation = True
        frames_hinges = hinges
        # Compute curves from hinges
        from .geometry import binormals_to_tangents, tangents_to_curve
        frames_curves = []
        for hinges_frame in frames_hinges:
            tangents = binormals_to_tangents(hinges_frame, normalize=False)
            curve_frame = tangents_to_curve(tangents, center=True)
            frames_curves.append(curve_frame)
    elif isinstance(curve, list):
        # Animation from list of curves
        is_animation = True
        frames_curves = curve
        if hinges is None:
            # Compute hinges from curves
            from .geometry import curve_to_binormals
            frames_hinges = [curve_to_binormals(c) for c in frames_curves]
        else:
            frames_hinges = hinges if isinstance(hinges, list) else [hinges] * len(frames_curves)

    if is_animation:
        # Create animation
        return _create_band_animation(
            frames_curves,
            frames_hinges,
            width=width,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            title=title,
            scalar_properties=scalar_properties,
            animation_obj=animation,
            show_curve=show_curve,
            untwist=untwist,
            interval=interval,
            figsize=figsize,
        )

    # Static plot (original behavior)
    if curve is None or hinges is None:
        raise ValueError("For static plots, both curve and hinges must be provided")

    pts = np.asarray(curve)
    h = np.asarray(hinges)

    if pts.ndim != 2 or pts.shape[1] != 3:
        msg = f"expected (n, 3) curve array, got shape {pts.shape}"
        raise ValueError(msg)
    if h.ndim != 2 or h.shape[1] != 3:
        msg = f"expected (n, 3) hinges array, got shape {h.shape}"
        raise ValueError(msg)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    # Compute tetrahedral vertices
    vertices, _ = compute_tetrahedron_vertices(pts, h, width=width)

    # Number of segments: k = nops(V)/2 - 2
    # vertices has 2*n points where n is number of curve points
    # k = n - 1 (number of segments connecting curve points)
    n_curve_points = len(pts)
    k = n_curve_points - 1

    # Create quadrilateral faces
    # Vertices are indexed: 2*i (top at curve[i]), 2*i+1 (bottom at curve[i])
    quad_faces = []
    for i in range(k):
        if untwist:
            # Untwisted ordering: [V[2*i], V[2*i+1], V[2*(i+1)], V[2*(i+1)+1]]
            quad = [
                vertices[2 * i],          # top at curve[i]
                vertices[2 * i + 1],      # bottom at curve[i]
                vertices[2 * (i + 1)],    # top at curve[i+1]
                vertices[2 * (i + 1) + 1] # bottom at curve[i+1]
            ]
        else:
            # Twisted ordering: [V[2*i], V[2*i+1], V[2*(i+1)+1], V[2*(i+1)]]
            quad = [
                vertices[2 * i],          # top at curve[i]
                vertices[2 * i + 1],      # bottom at curve[i]
                vertices[2 * (i + 1) + 1], # bottom at curve[i+1]
                vertices[2 * (i + 1)]     # top at curve[i+1]
            ]
        quad_faces.append(quad)

    # Create polygon collection
    poly_collection = Poly3DCollection(
        quad_faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly_collection)

    # Optionally show curve
    if show_curve:
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            "-",
            color="crimson",
            linewidth=2,
            alpha=0.8,
        )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    _set_equal_aspect_3d(ax, vertices)

    return ax


def create_rotation_animation(
    curve: np.ndarray,
    *,
    frames: int = 90,
    elevation: float = 25.0,
    interval: int = 50,
    figsize: tuple[float, float] = (6, 5),
) -> tuple[Figure, object]:
    """Create a rotating animation of the curve.

    Parameters
    ----------
    curve:
        (n, 3) array of 3D points.
    frames:
        Number of animation frames.
    elevation:
        Viewing elevation angle in degrees.
    interval:
        Delay between frames in milliseconds.
    figsize:
        Figure size (width, height) in inches.

    Returns
    -------
    fig:
        The matplotlib figure.
    anim:
        The animation object (use `anim.to_jshtml()` to display in notebooks).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    pts = np.asarray(curve)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    (line,) = ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        "-o",
        color="steelblue",
        linewidth=2,
        markersize=6,
    )

    ax.set_xlim(np.min(pts[:, 0]) - 0.5, np.max(pts[:, 0]) + 0.5)
    ax.set_ylim(np.min(pts[:, 1]) - 0.5, np.max(pts[:, 1]) + 0.5)
    ax.set_zlim(np.min(pts[:, 2]) - 0.5, np.max(pts[:, 2]) + 0.5)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.3)

    def init():
        return (line,)

    def update(frame):
        angle = 360.0 * frame / frames
        ax.view_init(elev=elevation, azim=angle)
        return (line,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=interval,
        blit=True,
    )

    return fig, anim


def _set_equal_aspect_3d(ax: Axes, points: np.ndarray) -> None:
    """Set equal aspect ratio for 3D plots.

    Parameters
    ----------
    ax:
        Matplotlib 3D axes.
    points:
        (n, 3) array of points to determine bounds.
    """
    pts = np.asarray(points)

    # Compute ranges
    ranges = [np.ptp(pts[:, i]) for i in range(3)]
    max_range = max(ranges)

    # Compute centers
    centers = [pts[:, i].mean() for i in range(3)]

    # Set limits
    for center, label in zip(
        centers, ["set_xlim", "set_ylim", "set_zlim"], strict=True
    ):
        getattr(ax, label)(center - max_range / 2, center + max_range / 2)

    ax.set_box_aspect([1, 1, 1])


def _apex_2d(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Find third 2D point w such that |w-u|=a and |w-v|=b.

    Uses the cosine rule to find the angle, then rotates the vector (v-u)
    by 90 degrees to find the perpendicular direction.

    Parameters
    ----------
    u, v:
        Two 2D points (shape (2,)).
    a:
        Distance from u to the new point.
    b:
        Distance from v to the new point.

    Returns
    -------
    w:
        The third point as a 2D array.

    References
    ----------
    Corresponds to apex function in Maple code (line 363).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Distance between u and v
    c = np.linalg.norm(v - u)

    if c < 1e-10:
        raise ValueError("Points u and v are too close")

    # Cosine rule: a^2 = b^2 + c^2 - 2bc*cos(angle)
    # Rearranged: cos(angle) = (a^2 + c^2 - b^2) / (2ac)
    cos_angle = (a * a + c * c - b * b) / (2 * a * c)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    sin_angle = np.sqrt(1 - cos_angle * cos_angle)

    # Direction from u to v
    direction = (v - u) / c

    # Perpendicular direction (90 degree rotation: [x, y] -> [-y, x])
    perpendicular = np.array([-direction[1], direction[0]])

    # New point w = u + a*cos_angle*direction + a*sin_angle*perpendicular
    w = u + a * cos_angle * direction + a * sin_angle * perpendicular

    return w


def _tetsheet_2d(
    vertices: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
) -> list[np.ndarray]:
    """Unfold a tetrahedron into 2D given its 3D vertices.

    Computes the 2D positions of 7 vertices that represent the unfolded
    tetrahedron, starting from two given 2D positions.

    Parameters
    ----------
    vertices:
        4x3 array of tetrahedron vertices in 3D (V[0..3]).
    u1, u2:
        Initial 2D positions (shape (2,) each).

    Returns
    -------
    u:
        List of 7 2D points representing the unfolded tetrahedron.

    References
    ----------
    Corresponds to tetsheet function in Maple code (line 364).
    """
    V = np.asarray(vertices, dtype=float)

    # Compute all edge lengths from 3D tetrahedron
    # a = |V[0] - V[1]|
    # b = |V[0] - V[2]|
    # c = |V[0] - V[3]|
    # d = |V[1] - V[2]|
    # e = |V[1] - V[3]|
    a = np.linalg.norm(V[0] - V[1])
    b = np.linalg.norm(V[0] - V[2])
    c = np.linalg.norm(V[0] - V[3])
    d = np.linalg.norm(V[1] - V[2])
    e = np.linalg.norm(V[1] - V[3])

    # Initialize 7 vertices (using 0-indexing)
    u = [None] * 7

    # u[0] and u[1] are given
    u[0] = np.asarray(u1, dtype=float)
    u[1] = np.asarray(u2, dtype=float)

    # u[2] = apex(u[0], u[1], b, d)
    u[2] = _apex_2d(u[0], u[1], b, d)

    # u[3] = apex(u[2], u[1], a, e)
    u[3] = _apex_2d(u[2], u[1], a, e)

    # u[4] = apex(u[3], u[1], c, a)
    u[4] = _apex_2d(u[3], u[1], c, a)

    # u[6] = apex(u[3], u[4], a, b)
    u[6] = _apex_2d(u[3], u[4], a, b)

    # u[5] = apex(u[6], u[4], d, a)
    u[5] = _apex_2d(u[6], u[4], d, a)

    return u


def paper_model(
    hinges: np.ndarray,
    curve: np.ndarray | None = None,
    ax: Axes | None = None,
    *,
    width: float = 0.15,
    linewidth: float = 1.5,
    edgecolor: str = "black",
    facecolors: list[str] | None = None,
    alpha: float = 0.3,
    title: str | None = None,
) -> Axes:
    """Create a 2D paper model (unfolding pattern) of the kaleidocycle.

    This creates a flat template that can be printed, cut out, and folded
    to create a physical kaleidocycle model. The function unfolds each
    tetrahedron in the chain into a 2D pattern.

    Parameters
    ----------
    hinges:
        (n+1, 3) array of hinge (binormal) vectors.
    curve:
        Optional (n+1, 3) array of curve points. If None, computed from hinges.
    ax:
        Matplotlib 2D axes. If None, creates a new figure.
    width:
        Half-width of the tetrahedra along hinge directions.
    linewidth:
        Width of the edge lines.
    edgecolor:
        Color of the edges.
    facecolors:
        Optional list of colors for each tetrahedron. If None, uses default colors.
    alpha:
        Transparency of the faces.
    title:
        Optional title for the plot.

    Returns
    -------
    ax:
        Matplotlib 2D axes with the paper model plot.

    References
    ----------
    Corresponds to paper_model function in Maple code (line 366).

    Example
    -------
    >>> from kaleidocycle import random_hinges, paper_model
    >>> import matplotlib.pyplot as plt
    >>> hinges = random_hinges(6, seed=42).as_array()
    >>> ax = paper_model(hinges)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    h = np.asarray(hinges, dtype=float)
    if h.ndim != 2 or h.shape[1] != 3:
        msg = f"expected (n+1, 3) hinges array, got shape {h.shape}"
        raise ValueError(msg)

    # Compute curve if not provided
    if curve is None:
        from .geometry import binormals_to_tangents, tangents_to_curve
        tangents = binormals_to_tangents(h, normalize=False)
        curve = tangents_to_curve(tangents, center=True)
    else:
        curve = np.asarray(curve, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Compute tetrahedral vertices in 3D
    vertices_3d, _ = compute_tetrahedron_vertices(curve, h, width=width)

    n = len(h) - 1  # Number of tetrahedra

    # Default colors if not provided
    if facecolors is None:
        # Use different colors for variety
        facecolors = ["lightblue", "lightcoral", "lightgreen", "lightyellow"] * ((n + 3) // 4)

    # Initialize starting positions for 2D unfolding
    # Start with u[2] at origin and u[3] on x-axis
    V0 = vertices_3d[0:4]  # First tetrahedron
    edge_length = np.linalg.norm(V0[0] - V0[1])

    u_prev2 = np.array([0.0, 0.0])
    u_prev3 = np.array([edge_length, 0.0])

    # Iterate through each tetrahedron
    # Each tetrahedron connects curve points i and i+1
    # vertices_3d has 2 vertices per curve point (top and bottom along hinge)
    for i in range(n):
        # Get vertices for this tetrahedron
        # Vertices are: [v_i_top, v_i_bottom, v_{i+1}_top, v_{i+1}_bottom]
        V = np.array([
            vertices_3d[2 * i],          # top at curve[i]
            vertices_3d[2 * i + 1],      # bottom at curve[i]
            vertices_3d[2 * (i + 1)],    # top at curve[i+1]
            vertices_3d[2 * (i + 1) + 1] # bottom at curve[i+1]
        ])

        # Unfold this tetrahedron
        u = _tetsheet_2d(V, u_prev2, u_prev3)

        # Draw triangular faces
        is_last = (i == n - 1)
        _draw_triangles_2d(
            ax, u, is_last,
            facecolor=facecolors[i % len(facecolors)],
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Update positions for next tetrahedron
        u_prev2 = u[2]
        u_prev3 = u[3]

    # Set equal aspect and clean appearance
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.grid(True, alpha=0.2)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Kaleidocycle Paper Model (Unfolding Pattern)", fontsize=12, fontweight="bold")

    # Adjust limits to show all polygons
    ax.autoscale()
    ax.margins(0.05)

    return ax


def _draw_triangles_2d(
    ax: Axes,
    u: list[np.ndarray],
    is_last: bool,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
) -> None:
    """Draw triangular faces for the unfolded tetrahedron.

    Parameters
    ----------
    ax:
        Matplotlib 2D axes.
    u:
        List of 7 2D points.
    is_last:
        If True, omits some faces for the final connection.
    facecolor, edgecolor, linewidth, alpha:
        Styling parameters.

    References
    ----------
    Corresponds to drawtrig function in Maple code (line 365).
    """
    from matplotlib.patches import Polygon as MplPolygon

    # Always draw these two triangles
    # Triangle 1: [u[0], u[1], u[2]]
    poly1 = MplPolygon(
        [u[0], u[1], u[2]],
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(poly1)

    # Triangle 2: [u[3], u[1], u[4]]
    poly2 = MplPolygon(
        [u[3], u[1], u[4]],
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(poly2)

    # If not last, draw three more triangles
    if not is_last:
        # Triangle 3: [u[2], u[1], u[3]]
        poly3 = MplPolygon(
            [u[2], u[1], u[3]],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(poly3)

        # Triangle 4: [u[3], u[4], u[6]]
        poly4 = MplPolygon(
            [u[3], u[4], u[6]],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(poly4)

        # Triangle 5: [u[6], u[4], u[5]]
        poly5 = MplPolygon(
            [u[6], u[4], u[5]],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(poly5)


def plot_energy_comparison(
    samples: dict[str, Kaleidocycle],
    *,
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Create a bar chart comparing energies across multiple samples.

    Parameters
    ----------
    samples:
        Dictionary mapping sample names to Kaleidocycle objects.
    figsize:
        Figure size (width, height) in inches.

    Returns
    -------
    fig:
        The matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    from .energies import bending_energy, dipole_energy, torsion_energy

    names = list(samples.keys())
    bending = []
    torsion = []
    dipole = []

    for sample in samples.values():
        bending.append(bending_energy(sample.tangents))
        torsion.append(torsion_energy(sample.hinges, wrap=True))
        dipole.append(dipole_energy(sample.hinges, sample.curve))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    x = np.arange(len(names))
    width = 0.6

    axes[0].bar(x, bending, width, color="steelblue")
    axes[0].set_title("Bending Energy", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, torsion, width, color="forestgreen")
    axes[1].set_title("Torsion Energy", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, dipole, width, color="crimson")
    axes[2].set_title("Dipole Energy", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_vertex_values_2d(
    values: np.ndarray,
    ax: Axes | None = None,
    *,
    cmap: str = "viridis",
    linewidth: float = 3.0,
    markersize: float = 8.0,
    title: str | None = None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_axes: bool = True,
) -> Axes | tuple[Figure, object]:
    """Plot vertex values as 2D line plot with vertex index on horizontal axis.

    Parameters
    ----------
    values:
        1D array (static) or 2D array (animation) of vertex values.
    (other parameters same as plot_vertex_values)

    Returns
    -------
    ax or (fig, anim):
        Axes for static plot, or (figure, animation) for time evolution.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection

    # Determine if this is static (1D) or animation (2D)
    is_animation = values.ndim == 2

    if is_animation:
        # Animation case
        return _create_vertex_animation_2d(
            values,
            cmap=cmap,
            linewidth=linewidth,
            markersize=markersize,
            title=title,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            vmin=vmin,
            vmax=vmax,
            show_axes=show_axes,
        )

    # Static case: 1D array
    if values.ndim != 1:
        msg = f"expected 1D or 2D values array, got shape {values.shape}"
        raise ValueError(msg)

    n_vertices = len(values)
    vertex_indices = np.arange(n_vertices)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Determine color range
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    # Create colored line segments
    points = np.array([vertex_indices, values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection with colors
    segment_colors = []
    for i in range(len(segments)):
        # Use average value of segment endpoints for color
        avg_value = (values[i] + values[i + 1]) / 2
        segment_colors.append(colormap(norm(avg_value)))

    lc = LineCollection(segments, colors=segment_colors, linewidth=linewidth)
    ax.add_collection(lc)

    # Plot colored vertices
    scatter = ax.scatter(
        vertex_indices,
        values,
        c=values,
        cmap=cmap,
        s=markersize**2,
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=1,
        zorder=10,
    )

    # Set axis limits
    ax.set_xlim(-0.5, n_vertices - 0.5)
    y_range = vmax - vmin
    ax.set_ylim(vmin - 0.1 * y_range, vmax + 0.1 * y_range)

    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(values)
        cbar = plt.colorbar(mappable, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    if show_axes:
        ax.set_xlabel("Vertex Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()

    return ax


def _create_vertex_animation_2d(
    values: np.ndarray,
    *,
    cmap: str = "viridis",
    linewidth: float = 3.0,
    markersize: float = 8.0,
    title: str | None = None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_axes: bool = True,
    interval: int = 100,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, object]:
    """Create 2D line plot animation for time-evolving vertex values.

    Parameters
    ----------
    values:
        (n_frames, n_vertices) array of values over time.
    (other parameters same as plot_vertex_values)

    Returns
    -------
    fig, anim:
        Matplotlib figure and animation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection

    n_frames, n_vertices = values.shape
    vertex_indices = np.arange(n_vertices)

    # Determine global color range
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set axis limits (fixed throughout animation)
    ax.set_xlim(-0.5, n_vertices - 0.5)
    y_range = vmax - vmin
    ax.set_ylim(vmin - 0.1 * y_range, vmax + 0.1 * y_range)

    if show_axes:
        ax.set_xlabel("Vertex Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()

    # Add colorbar
    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(values)
        cbar = plt.colorbar(mappable, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10)

    # Initialize line collection and scatter
    lc = LineCollection([], linewidth=linewidth)
    ax.add_collection(lc)

    scatter = ax.scatter([], [], s=markersize**2, edgecolors="white", linewidths=1, zorder=10)

    # Title with frame counter
    if title:
        title_text = ax.set_title(f"{title} (frame 0/{n_frames})", fontsize=12, fontweight="bold")
    else:
        title_text = ax.set_title(f"Frame 0/{n_frames}", fontsize=12, fontweight="bold")

    def init():
        """Initialize animation."""
        lc.set_segments([])
        scatter.set_offsets(np.empty((0, 2)))
        return lc, scatter, title_text

    def update(frame):
        """Update function for animation."""
        current_values = values[frame]

        # Create segments
        points = np.array([vertex_indices, current_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Compute segment colors
        segment_colors = []
        for i in range(len(segments)):
            avg_value = (current_values[i] + current_values[i + 1]) / 2
            segment_colors.append(colormap(norm(avg_value)))

        lc.set_segments(segments)
        lc.set_colors(segment_colors)

        # Update scatter
        scatter.set_offsets(np.c_[vertex_indices, current_values])
        colors = colormap(norm(current_values))
        scatter.set_color(colors)

        # Update title
        if title:
            title_text.set_text(f"{title} (frame {frame}/{n_frames})")
        else:
            title_text.set_text(f"Frame {frame}/{n_frames}")

        return lc, scatter, title_text

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=interval,
        blit=True,
    )

    return fig, anim


def plot_vertex_values(
    values: np.ndarray | list,
    curve: np.ndarray | None = None,
    ax: Axes | None = None,
    *,
    cmap: str = "viridis",
    linewidth: float = 3.0,
    markersize: float = 8.0,
    title: str | None = None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_axes: bool = True,
) -> Axes | tuple[Figure, object]:
    """Plot vertex-based values (e.g., curvature, torsion) on a 3D curve or as a line plot.

    This function visualizes scalar values associated with each vertex. When a curve
    is provided, it colors the 3D curve. When no curve is given, it creates a 2D line
    plot with vertex index on the horizontal axis. If a 2D array is provided, it creates
    an animation showing the time evolution of the values.

    Parameters
    ----------
    values:
        Vertex values to visualize. Can be:
        - 1D array of shape (n,): Single snapshot, creates static plot
        - 2D array of shape (n_frames, n): Time evolution, creates animation
          where each row represents a snapshot for all vertices
        - List of lists: Interpreted as 2D array for time evolution
    curve:
        Optional (n, 3) array of 3D curve points. If None, creates a 2D line plot
        with vertex index on the horizontal axis instead of a 3D plot.
    ax:
        Matplotlib axes (2D or 3D depending on curve). If None, creates a new figure.
        Ignored when creating animations.
    cmap:
        Colormap name for coloring vertices.
    linewidth:
        Width of the line.
    markersize:
        Size of vertex markers.
    title:
        Optional title for the plot.
    show_colorbar:
        Whether to display a colorbar.
    colorbar_label:
        Label for the colorbar.
    vmin, vmax:
        Min and max values for color normalization. If None, uses data range.
    show_axes:
        Whether to display axis labels and grid.

    Returns
    -------
    ax:
        Matplotlib axes object (for static plots).
    (fig, anim):
        Tuple of figure and animation object (for time evolution).
        Use `anim.to_jshtml()` to display in notebooks.

    Examples
    --------
    Static plot of curvature on 3D curve:
    >>> from kaleidocycle.geometry import random_hinges, binormals_to_tangents
    >>> from kaleidocycle.geometry import tangents_to_curve, pairwise_curvature
    >>> hinges = random_hinges(6, seed=42).as_array()
    >>> tangents = binormals_to_tangents(hinges, normalize=False)
    >>> curve = tangents_to_curve(tangents)
    >>> curvature = pairwise_curvature(hinges)
    >>> ax = plot_vertex_values(curvature, curve, title="Curvature")

    Static line plot without curve:
    >>> curvature = pairwise_curvature(hinges)
    >>> ax = plot_vertex_values(curvature, title="Curvature vs Vertex Index")

    Animation of time-evolving torsion:
    >>> # torsion_evolution is shape (n_frames, n_vertices)
    >>> fig, anim = plot_vertex_values(torsion_evolution, curve, title="Torsion Evolution")
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Convert to numpy array
    values_array = np.asarray(values, dtype=float)

    # Check if curve is provided
    if curve is None:
        # 2D line plot mode
        return _plot_vertex_values_2d(
            values_array,
            ax=ax,
            cmap=cmap,
            linewidth=linewidth,
            markersize=markersize,
            title=title,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            vmin=vmin,
            vmax=vmax,
            show_axes=show_axes,
        )

    pts = np.asarray(curve)

    if pts.ndim != 2 or pts.shape[1] != 3:
        msg = f"expected (n, 3) curve array, got shape {pts.shape}"
        raise ValueError(msg)

    # Determine if this is static (1D) or animation (2D)
    is_animation = values_array.ndim == 2

    # Handle edge-based values (one less than vertices)
    # E.g., curvature is computed on edges, not vertices
    if is_animation:
        n_frames, n_values = values_array.shape
        if n_values == pts.shape[0] - 1:
            # Edge values: expand to vertex values by repeating edge values
            # [e0, e1, e2] -> [e0, e1, e2, e2] for closure
            values_array = np.column_stack([values_array, values_array[:, -1]])
        elif n_values != pts.shape[0]:
            msg = f"values shape {values_array.shape} doesn't match curve vertices {pts.shape[0]}"
            raise ValueError(msg)

    if is_animation:
        # Time evolution case: shape (n_frames, n_vertices)
        n_frames, n_vertices = values_array.shape

        # Create animation
        return _create_vertex_animation(
            values_array,
            pts,
            cmap=cmap,
            linewidth=linewidth,
            markersize=markersize,
            title=title,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            vmin=vmin,
            vmax=vmax,
            show_axes=show_axes,
        )
    else:
        # Static case: shape (n_vertices,)
        if values_array.ndim != 1:
            msg = f"expected 1D or 2D values array, got shape {values_array.shape}"
            raise ValueError(msg)

        # Handle edge-based values for static case
        if values_array.shape[0] == pts.shape[0] - 1:
            # Edge values: expand to vertex values by appending last value
            values_array = np.append(values_array, values_array[-1])
        elif values_array.shape[0] != pts.shape[0]:
            msg = f"values length {values_array.shape[0]} doesn't match curve vertices {pts.shape[0]}"
            raise ValueError(msg)

        # Create static plot
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        # Determine color range
        if vmin is None:
            vmin = np.min(values_array)
        if vmax is None:
            vmax = np.max(values_array)

        # Create colormap normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)

        # Plot colored line segments
        for i in range(len(pts) - 1):
            # Use average value of segment endpoints for color
            segment_value = (values_array[i] + values_array[i + 1]) / 2
            color = colormap(norm(segment_value))

            ax.plot(
                [pts[i, 0], pts[i + 1, 0]],
                [pts[i, 1], pts[i + 1, 1]],
                [pts[i, 2], pts[i + 1, 2]],
                "-",
                color=color,
                linewidth=linewidth,
            )

        # Plot colored vertices
        colors = colormap(norm(values_array))
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=values_array,
            cmap=cmap,
            s=markersize**2,
            vmin=vmin,
            vmax=vmax,
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

        if show_colorbar:
            mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
            mappable.set_array(values_array)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, pad=0.1)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=10)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")

        if show_axes:
            ax.set_xlabel("X", fontsize=10)
            ax.set_ylabel("Y", fontsize=10)
            ax.set_zlabel("Z", fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_axis_off()

        # Set equal aspect ratio
        _set_equal_aspect_3d(ax, pts)

        return ax


def _create_vertex_animation(
    values: np.ndarray,
    curve: np.ndarray,
    *,
    cmap: str = "viridis",
    linewidth: float = 3.0,
    markersize: float = 8.0,
    title: str | None = None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_axes: bool = True,
    interval: int = 100,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, object]:
    """Create animation for time-evolving vertex values.

    Parameters
    ----------
    values:
        (n_frames, n_vertices) array of values over time.
    curve:
        (n_vertices, 3) array of curve points.
    (other parameters same as plot_vertex_values)
    interval:
        Delay between frames in milliseconds.
    figsize:
        Figure size (width, height) in inches.

    Returns
    -------
    fig:
        Matplotlib figure.
    anim:
        Animation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import cm
    from matplotlib.colors import Normalize

    n_frames, n_vertices = values.shape
    pts = np.asarray(curve)

    # Determine global color range across all frames
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Set up axes limits (fixed throughout animation)
    _set_equal_aspect_3d(ax, pts)

    if show_axes:
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()

    # Add colorbar
    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(values)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, pad=0.1)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10)

    # Initialize line segments and scatter plot
    line_segments = []
    for i in range(n_vertices - 1):
        (line,) = ax.plot([], [], [], "-", linewidth=linewidth)
        line_segments.append(line)

    scatter = ax.scatter([], [], [], s=markersize**2, edgecolors="white", linewidths=1, zorder=10)

    # Title with frame counter
    if title:
        title_text = ax.set_title(f"{title} (frame 0/{n_frames})", fontsize=12, fontweight="bold")
    else:
        title_text = ax.set_title(f"Frame 0/{n_frames}", fontsize=12, fontweight="bold")

    def init():
        """Initialize animation."""
        for line in line_segments:
            line.set_data([], [])
            line.set_3d_properties([])
        scatter._offsets3d = ([], [], [])
        return line_segments + [scatter, title_text]

    def update(frame):
        """Update function for animation."""
        current_values = values[frame]

        # Update line segments with colors
        for i in range(n_vertices - 1):
            segment_value = (current_values[i] + current_values[i + 1]) / 2
            color = colormap(norm(segment_value))
            line_segments[i].set_data([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]])
            line_segments[i].set_3d_properties([pts[i, 2], pts[i + 1, 2]])
            line_segments[i].set_color(color)

        # Update scatter plot
        colors = colormap(norm(current_values))
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        scatter.set_color(colors)

        # Update title
        if title:
            title_text.set_text(f"{title} (frame {frame}/{n_frames})")
        else:
            title_text.set_text(f"Frame {frame}/{n_frames}")

        return line_segments + [scatter, title_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=interval,
        blit=False,  # blit=False needed for 3D animations
    )

    return fig, anim

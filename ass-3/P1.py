import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Parameters

np.random.seed(42)

N = 20
P_ER = 0.4
NAME = "PUSHKAR"

DT = 0.02
T_PER_LETTER = 4.0
STEPS_PER_LETTER = int(T_PER_LETTER / DT)

K_GAIN = 2.0

WORLD_SCALE = 6.0

FPS = 30
FRAME_STRIDE = 4


# Generate a connected Erdos-Renyi graph

def generate_graph(n, p, seed=1):
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=seed)

        if nx.is_connected(G):
            return G

        seed += 1


G = generate_graph(N, P_ER)
Adj = nx.to_numpy_array(G)

print(
    f"Graph: N={N}, p={P_ER}, "
    f"edges={G.number_of_edges()}, "
    f"connected={nx.is_connected(G)}"
)


# Random initial positions

x0 = np.random.uniform(
    -WORLD_SCALE,
    WORLD_SCALE,
    size=(N, 2)
)


# Letter definitions

LETTER_SEGMENTS = {

    "P": [
        ((-0.6, -1.0), (-0.6, 1.0)),
        ((-0.6, 1.0), (0.3, 1.0)),
        ((0.3, 1.0), (0.6, 0.7)),
        ((0.6, 0.7), (0.6, 0.3)),
        ((0.6, 0.3), (0.3, 0.0)),
        ((0.3, 0.0), (-0.6, 0.0))
    ],

    "U": [
        ((-0.6, 1.0), (-0.6, -0.5)),
        ((-0.6, -0.5), (-0.3, -1.0)),
        ((-0.3, -1.0), (0.3, -1.0)),
        ((0.3, -1.0), (0.6, -0.5)),
        ((0.6, -0.5), (0.6, 1.0))
    ],

    "S": [
        ((0.6, 0.8), (0.3, 1.0)),
        ((0.3, 1.0), (-0.4, 1.0)),
        ((-0.4, 1.0), (-0.6, 0.7)),
        ((-0.6, 0.7), (-0.6, 0.2)),
        ((-0.6, 0.2), (0.4, -0.2)),
        ((0.4, -0.2), (0.6, -0.5)),
        ((0.6, -0.5), (0.4, -1.0)),
        ((0.4, -1.0), (-0.4, -1.0)),
        ((-0.4, -1.0), (-0.6, -0.8))
    ],

    "H": [
        ((-0.6, -1.0), (-0.6, 1.0)),
        ((0.6, -1.0), (0.6, 1.0)),
        ((-0.6, 0.0), (0.6, 0.0))
    ],

    "K": [
        ((-0.6, -1.0), (-0.6, 1.0)),
        ((-0.6, 0.0), (0.6, 1.0)),
        ((-0.6, 0.0), (0.6, -1.0))
    ],

    "A": [
        ((-0.7, -1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.7, -1.0)),
        ((-0.4, 0.0), (0.4, 0.0))
    ],

    "R": [
        ((-0.6, -1.0), (-0.6, 1.0)),
        ((-0.6, 1.0), (0.3, 1.0)),
        ((0.3, 1.0), (0.6, 0.7)),
        ((0.6, 0.7), (0.6, 0.3)),
        ((0.6, 0.3), (0.3, 0.0)),
        ((0.3, 0.0), (-0.6, 0.0)),
        ((0.0, 0.0), (0.7, -1.0))
    ]
}


# Sample points along the letter segments

def sample_segment(p1, p2, n):
    p1 = np.array(p1)
    p2 = np.array(p2)

    t = np.linspace(0, 1, n)

    return np.outer(1 - t, p1) + np.outer(t, p2)


def letter_points(letter, n=N):

    segments = LETTER_SEGMENTS[letter]

    lengths = [
        np.linalg.norm(
            np.array(b) - np.array(a)
        )
        for a, b in segments
    ]

    total_length = sum(lengths)

    counts = [
        max(1, round(n * length / total_length))
        for length in lengths
    ]

    # Adjust so exactly N points are generated
    while sum(counts) > n:
        i = np.argmax(counts)

        if counts[i] > 1:
            counts[i] -= 1

    while sum(counts) < n:
        i = np.argmax(lengths)
        counts[i] += 1

    points = []

    for (p1, p2), count in zip(
        segments,
        counts
    ):
        points.append(
            sample_segment(p1, p2, count)
        )

    return np.vstack(points) * WORLD_SCALE


formation_targets = [
    letter_points(letter)
    for letter in NAME
]


# Assign agents to nearby target points

def assign_targets(x, targets):

    remaining = list(range(N))
    assigned = np.zeros_like(targets)

    for i in range(N):

        distances = np.linalg.norm(
            targets[remaining] - x[i],
            axis=1
        )

        nearest = np.argmin(distances)
        target_index = remaining.pop(nearest)

        assigned[i] = targets[target_index]

    return assigned


# Distributed formation-control law

def formation_control(x, target):

    dx = np.zeros_like(x)

    for i in range(N):

        neighbors = np.where(
            Adj[i] > 0
        )[0]

        if len(neighbors) == 0:
            continue

        error = (
            (x[i] - x[neighbors])
            -
            (target[i] - target[neighbors])
        )

        dx[i] = -K_GAIN * error.sum(axis=0)

    return dx


# Simulation

trajectory = [x0.copy()]
x = x0.copy()

for letter, target in zip(
    NAME,
    formation_targets
):

    print(f"Forming letter: {letter}")

    target = assign_targets(
        x,
        target
    )

    for _ in range(
        STEPS_PER_LETTER
    ):

        dx = formation_control(
            x,
            target
        )

        x += DT * dx

        trajectory.append(
            x.copy()
        )


trajectory = np.array(trajectory)

print(
    f"Simulation finished: "
    f"{len(trajectory)} states"
)


# Animation

fig, ax = plt.subplots(
    figsize=(8, 8)
)

margin = 2

ax.set_xlim(
    trajectory[:, :, 0].min() - margin,
    trajectory[:, :, 0].max() + margin
)

ax.set_ylim(
    trajectory[:, :, 1].min() - margin,
    trajectory[:, :, 1].max() + margin
)

ax.set_aspect("equal")

ax.set_title(
    f"Formation Control: {NAME} ({N} agents)"
)


# Agents

scat = ax.scatter(
    [],
    [],
    s=70,
    c="crimson",
    edgecolors="black",
    linewidths=0.5,
    zorder=2
)


# Communication graph

edge_lines = [
    ax.plot(
        [],
        [],
        color="gray",
        lw=0.5,
        alpha=0.5
    )[0]
    for _ in G.edges()
]


# Current letter

letter_label = ax.text(
    0.02,
    0.96,
    "",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)


frame_indices = np.arange(
    0,
    len(trajectory),
    FRAME_STRIDE
)


def current_letter(step):

    index = min(
        step // STEPS_PER_LETTER,
        len(NAME) - 1
    )

    return NAME[index]


def init():

    scat.set_offsets(
        np.zeros((N, 2))
    )

    for line in edge_lines:
        line.set_data([], [])

    letter_label.set_text("")

    return [
        scat,
        letter_label
    ] + edge_lines


def update(frame):

    step = frame_indices[frame]
    positions = trajectory[step]

    scat.set_offsets(positions)

    for line, (i, j) in zip(
        edge_lines,
        G.edges()
    ):

        line.set_data(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]]
        )

    letter_label.set_text(
        f"Target letter: {current_letter(step)}"
    )

    return [
        scat,
        letter_label
    ] + edge_lines


anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    init_func=init,
    blit=True,
    interval=1000 / FPS
)


# Save animation

try:

    writer = animation.FFMpegWriter(
        fps=FPS,
        bitrate=1800
    )

    anim.save(
        "pushkar_formation_control.mp4",
        writer=writer
    )

    print(
        "Saved: pushkar_formation_control.mp4"
    )

except Exception as e:

    print(
        f"FFmpeg unavailable: {e}"
    )

    anim.save(
        "pushkar_formation_control.gif",
        writer=animation.PillowWriter(
            fps=FPS
        )
    )

    print(
        "Saved: pushkar_formation_control.gif"
    )


plt.close(fig)


# Save graph topology

fig, ax = plt.subplots(
    figsize=(6, 6)
)

graph_pos = nx.spring_layout(
    G,
    seed=1
)

nx.draw(
    G,
    graph_pos,
    ax=ax,
    with_labels=True,
    node_color="crimson",
    edge_color="gray",
    font_color="white",
    font_size=8
)

ax.set_title(
    f"Erdos-Renyi Graph (N={N}, p={P_ER})"
)

fig.savefig(
    "erdos_renyi_graph.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "Saved: erdos_renyi_graph.png"
)
import argparse
import os
import re

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import yaml
from bioverse.collaters import LongCollater
from bioverse.factory import BenchmarkFactory, TransformFactory
from bioverse.trainer import Trainer
from bioverse.utilities import config as CONFIG
from omegaconf import OmegaConf
from torch_scatter import scatter_sum

import models

args = argparse.ArgumentParser()
args.add_argument("exp", type=str)
args = args.parse_args()
EXP = args.exp.split("=")[1]


CONFIG.workers = 1

MAX_EXAMPLES = 20 if EXP == "mnist" else 5

alphabet = {
    "mnist": None,
    "beta2d": "NClCFPOS",
    "beta3d": "NClCFPOS",
    "qm9aph": "HCNOF",
}[EXP]
CHANNEL_NAMES = re.findall(r"[A-Z][a-z]*", alphabet) if alphabet else None


HEX_TRIPLES = [
    ("#BA68C8", "#E0E0E0", "#DCE775"),
    ("#9575CD", "#E0E0E0", "#FFF176"),
    ("#7986CB", "#E0E0E0", "#FFD54F"),
    ("#64B5F6", "#E0E0E0", "#FFB74D"),
    ("#4FC3F7", "#E0E0E0", "#FF8A65"),
    ("#4DD0E1", "#E0E0E0", "#E57373"),
    ("#4DB6AC", "#E0E0E0", "#F06292"),
]


# Define custom color scales per input channel using (low, center, high) hex codes
def make_scale(low_hex: str, center_hex: str, high_hex: str):
    return [
        [0.0, low_hex],
        [0.5, center_hex],
        [1.0, high_hex],
    ]


COLOR_SCALES = [make_scale(lo, ce, hi) for (lo, ce, hi) in HEX_TRIPLES]

DIM = 2 if EXP in ["mnist", "beta2d"] else 3

config = OmegaConf.load("config.yaml")
config.exp = EXP
config = OmegaConf.to_container(config, resolve=True)
config["trainer"]["logger"] = None  # disable logging for this one
ckpt = torch.load(f"results/{EXP}/checkpoint.pt", weights_only=True)
model = getattr(models, config["model"].replace(".models.", ""))()
model.load_state_dict(ckpt["model"])
model.eval()
device = next(model.parameters()).device


# also plot raw filter weights of the first layer for the MNIST experiment
if EXP == "mnist":
    print("Plotting Neural Fields")
    layer = model.cosmo_layers[0]
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    field = layer.neural_field

    # High-resolution grid for visualization
    PLOT_RES = 50
    X_MIN, X_MAX = -model.radius, model.radius
    Y_MIN, Y_MAX = -model.radius, model.radius
    xs_t = torch.linspace(X_MIN, X_MAX, PLOT_RES)
    ys_t = torch.linspace(Y_MIN, Y_MAX, PLOT_RES)
    Yg, Xg = torch.meshgrid(ys_t, xs_t)
    vis_pts = torch.stack([Xg, Yg], dim=-1).view(-1, 2).to(device)

    # Evaluate field and reshape to (H, W, out, in)
    w_vis = field(vis_pts).reshape(PLOT_RES, PLOT_RES, out_channels, in_channels)
    W_vis = w_vis.detach().cpu().numpy()
    xs = xs_t.detach().cpu().numpy().tolist()
    ys = ys_t.detach().cpu().numpy().tolist()

    # Create a 1x7 subfigure for the first seven output channels (use input channel 0)
    from plotly.subplots import make_subplots

    n_show = min(7, out_channels)
    hspace = 0.002
    fig = make_subplots(rows=1, cols=n_show, horizontal_spacing=hspace)
    # Gather z range across selected channels for shared colorbar
    Z_list = []
    for o in range(n_show):
        Z_list.append(w_vis[:, :, o, 0].detach().cpu().numpy())
    zmin = float(np.min([np.min(Z) for Z in Z_list]))
    zmax = float(np.max([np.max(Z) for Z in Z_list]))
    # Make the color range symmetric around zero
    zabs = max(abs(zmin), abs(zmax))
    zmin, zmax = -zabs, zabs
    # Shared colorbar ticks with explicit formatting
    cb_vals_fields = np.linspace(zmin, zmax, 3).tolist()
    cb_text_fields = [f"{v:+.2f}" for v in cb_vals_fields]
    # Add heatmaps
    for o in range(n_show):
        show_cbar = o == n_show - 1  # single legend/colorbar on the far right
        fig.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=Z_list[o],
                zmin=zmin,
                zmax=zmax,
                zmid=0.0,
                colorscale="Fall_r",
                showscale=show_cbar,
                colorbar=(
                    dict(
                        x=1.02,
                        y=0.5,
                        yanchor="middle",
                        len=1.0,
                        thickness=12,
                        outlinewidth=0,
                        ticks="outside",
                        tickmode="array",
                        tickvals=cb_vals_fields,
                        ticktext=cb_text_fields,
                        tickfont=dict(size=12, family="Courier New, monospace"),
                        xpad=10,
                        ypad=0,
                    )
                    if show_cbar
                    else None
                ),
            ),
            row=1,
            col=o + 1,
        )
    # Hide axes/annotations, simple white template
    fig.update_xaxes(
        visible=False, showticklabels=False, showgrid=False, zeroline=False
    )
    fig.update_yaxes(
        visible=False, showticklabels=False, showgrid=False, zeroline=False
    )
    # Ensure each subplot is square by anchoring y to its corresponding x
    for o in range(n_show):
        ax_key = "" if o == 0 else f"{o+1}"
        fig.update_yaxes(scaleanchor=f"x{ax_key}", scaleratio=1, row=1, col=o + 1)
    # Compute figure width so each subplot domain is square in pixels
    fig_height = 80
    domain_w = (1.0 - hspace * (n_show - 1)) / n_show
    fig_width = int(fig_height / domain_w)
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        autosize=False,
        margin=dict(l=0, r=30, t=10, b=10),
        template="simple_white",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.write_image(
        f"results/{EXP}/mnist_fields.pdf",
        width=fig_width,
        height=fig_height,
        scale=3,
    )


benchmark = BenchmarkFactory(config["benchmark"])
transforms = TransformFactory(config["transforms"])
benchmark.apply(transforms)
trainer = Trainer(model, benchmark, **config["trainer"])
loader = benchmark.loader(
    split="train",
    collater=LongCollater(),
    batch_size=1,
    progress=False,
)
num_example = 0
# Accumulators for the single-output case (one combined figure across examples)
single_out_examples = []
single_out_global_min = float("inf")
single_out_global_max = float("-inf")
for (X, y), data in loader:
    true_class = data.y[0]
    if EXP in ["beta2d", "beta3d"] and true_class != 1:
        continue  # only plot positive examples
    if EXP in ["beta2d", "beta3d"]:
        molecule_name = data.molecule_name[0]
        # Normalize molecule name to Title Case words
        mol_title = re.sub(r"[_\\-]+", " ", str(molecule_name)).strip().title()
    print(f"Processing example {num_example}")
    trainer.backend.put_on_device(data)
    if EXP == "beta2d" or EXP == "beta3d":
        edge_index = data.molecule_edges.T
    elif EXP == "mnist":
        edge_index = gnn.radius_graph(data.atom_pos, model.eps, data.vertex2molecule)
    L = trainer.model.lift(
        data.atom_features.float(),
        data.atom_pos.float(),
        edge_index,
        data.vertex2molecule,
    )
    if EXP in ["mnist", "beta2d"]:
        L.ijk = L.ij
        L.jkl = L.jk
        L.tri2node = L.edge2node
    edge_adj = torch.sparse_coo_tensor(
        torch.stack([L.ijk, L.jkl]), torch.arange(L.ijk.shape[0]).to(L.ijk.device)
    )

    # get activations
    _, cosmo_features, max_indices = trainer.model(data)
    model_probs = F.softmax(cosmo_features, dim=1).detach().cpu().numpy()
    # save model probabilities to file
    with open(f"results/{EXP}/model_probs_{num_example}.yaml", "w") as f:
        yaml.dump(
            {"model_probs": model_probs[0].tolist(), "true_class": true_class.item()}, f
        )
    # if EXP == "mnist" and true_class == np.argmax(model_probs[0]):
    #     continue  # only plot misclassified examples
    output_maps = []
    for class_num, max_index in enumerate(max_indices[0]):
        edges = max_index.unsqueeze(0)
        layers = model.cosmo_layers
        fields = torch.eye(layers[-1].out_channels).unsqueeze(0).to(L.ijk.device)

        for i in range(len(layers), 0, -1):
            layer = layers[i - 1]
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            field = layer.neural_field

            src, dst = torch.index_select(edge_adj, 1, edges).coalesce().indices()
            R = L.bases[edges[dst]]
            hood = L.coords[L.tri2node[src]] - L.coords[L.tri2node[edges[dst]]]
            hood = torch.bmm(R, hood.unsqueeze(-1)).squeeze(-1)
            hood = hood / layer.radius
            w = field(hood).view(-1, out_channels, in_channels)
            deg = (
                scatter_sum(
                    torch.ones_like(dst, dtype=w.dtype),
                    dst,
                    dim=0,
                    dim_size=edges.shape[0],
                )[dst]
                .unsqueeze(-1)
                .unsqueeze(-1)
            )  # * in_channels.sqrt()
            fields = torch.einsum("boi,bij->boj", fields[dst], w / deg)
            edges = src
        # sum fields to edges
        fields = scatter_sum(
            fields, L.tri2node[edges], dim_size=data.num_vertices.sum(), dim=0
        )
        # collect per-node activation values for the current output channel across input channels (no averaging)
        vals_ic = fields[:, class_num, :]  # shape: num_nodes x in_channels
        output_maps.append(vals_ic.detach().cpu().numpy())
        del fields, edges

    # Determine global symmetric color range across all outputs and input channels
    num_out = len(output_maps)
    if num_out == 0:
        num_example += 1
        if num_example >= MAX_EXAMPLES:
            break
        continue
    num_in = output_maps[0].shape[1]
    all_vals = np.concatenate([v.ravel() for v in output_maps if v.size > 0])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    max_abs = float(max(abs(vmin), abs(vmax)))
    vmin, vmax = -max_abs, max_abs
    # coordinates: project to 2D if needed (use first two principal components)
    coords_np = L.coords.detach().cpu().numpy()
    if coords_np.shape[1] == 3:
        X = coords_np.astype(np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            R = Vt.T
            coords_view = (Xc @ R)[:, :2]
        except np.linalg.LinAlgError:
            coords_view = Xc[:, :2]
    else:
        coords_view = coords_np[:, :2]
    # Rotate 2D coordinates to maximally fit a square (minimize |width - height| of the bounding box)
    coords2 = coords_view.astype(np.float64)
    xs_c = coords2[:, 0]
    ys_c = coords2[:, 1]
    best_theta = 0.0
    best_score = float("inf")
    for theta in np.linspace(0.0, np.pi, 181, endpoint=False):
        cth = np.cos(theta)
        sth = np.sin(theta)
        xr = cth * xs_c - sth * ys_c
        yr = sth * xs_c + cth * ys_c
        w = float(xr.max() - xr.min())
        h = float(yr.max() - yr.min())
        score = abs(w - h)
        if score < best_score:
            best_score = score
            best_theta = theta
    # Apply best rotation
    cth = np.cos(best_theta)
    sth = np.sin(best_theta)
    x_rot = cth * xs_c - sth * ys_c
    y_rot = sth * xs_c + cth * ys_c
    coords_view = np.stack([x_rot, y_rot], axis=1)
    # Precompute edge line coordinates in 2D view
    # For MNIST, flip Y to use Cartesian coordinates (y up)
    if EXP == "mnist":
        coords_view[:, 1] = -coords_view[:, 1]
    x_lines = []
    y_lines = []
    src_idx = edge_index[0].detach().cpu().numpy()
    dst_idx = edge_index[1].detach().cpu().numpy()
    for s, d in zip(src_idx, dst_idx):
        x_lines += [coords_view[s, 0], coords_view[d, 0], None]
        y_lines += [coords_view[s, 1], coords_view[d, 1], None]

    if EXP == "mnist":
        # Subfigure of outputs: 5 columns, shared colorbar, top-1 channel only
        from plotly.subplots import make_subplots

        ncols = 5
        nrows = int(np.ceil(num_out / ncols))
        hspace = 0.03
        vspace = 0.08
        # Load per-class probabilities for this example to label subplots

        with open(f"results/{EXP}/model_probs_{num_example}.yaml", "r") as f:
            _probs_yaml = yaml.safe_load(f) or {}
            _probs = _probs_yaml.get("model_probs", [])
        # Build subplot titles "Class X, p=YY%"
        _total_cells = nrows * ncols
        _titles = []
        for _idx in range(_total_cells):
            if _idx < num_out and _idx < len(_probs):
                _pct = int(round(float(_probs[_idx]) * 100))
                _titles.append(f"Class {_idx} (p={_pct}%)")
            elif _idx < num_out:
                _titles.append(f"Class {_idx}")
            else:
                _titles.append("")
        grid_fig = make_subplots(
            rows=nrows,
            cols=ncols,
            horizontal_spacing=hspace,
            vertical_spacing=vspace,
            subplot_titles=_titles,
        )
        # Precompute size so each subplot domain is reasonably square
        target_side = 100
        domain_w = (1.0 - hspace * (ncols - 1)) / ncols
        domain_h = (1.0 - vspace * (nrows - 1)) / nrows
        grid_height = int(target_side * nrows / domain_h)
        grid_width = int(target_side * ncols / domain_w) / 3
        # Molecule diameter for this example (from rotated coords)
        diam_x = float(coords_view[:, 0].max() - coords_view[:, 0].min())
        diam_y = float(coords_view[:, 1].max() - coords_view[:, 1].min())
        mol_diam = max(diam_x, diam_y) if max(diam_x, diam_y) > 0 else 1.0
        size_k = 160.0
        size_min, size_max = 4.0, 24.0
        marker_size_example = float(np.clip(size_k / mol_diam, size_min, size_max))
        label_font_size_example = int(np.clip(marker_size_example * 0.9, 8.0, 36.0))
        # Shared colorbar ticks for this figure
        cb_vals_ex = np.linspace(vmin, vmax, 3).tolist()
        cb_text_ex = [f"{v:+.2f}" for v in cb_vals_ex]
        colorbar_added = False
        # Add subplots
        for o in range(num_out):
            row = o // ncols + 1
            col = o % ncols + 1
            vals_ic = output_maps[o]  # shape: num_nodes x num_in
            abs_vals = np.abs(vals_ic)
            top1_idx = np.argmax(abs_vals, axis=1)
            top1_val = vals_ic[np.arange(vals_ic.shape[0]), top1_idx]
            # Edges
            grid_fig.add_trace(
                go.Scatter(
                    x=x_lines,
                    y=y_lines,
                    mode="lines",
                    line=dict(color="rgb(50,50,50)", width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            # Points: only top-1; label channel only if abs value is in upper 50 percentile
            if num_in > 1:
                abs_top1 = np.abs(top1_val)
                thr = float(np.median(abs_top1))
                texts = [
                    (
                        f"<b>{(CHANNEL_NAMES[c] if CHANNEL_NAMES else str(c))}</b>"
                        if abs_top1[i] >= thr
                        else ""
                    )
                    for i, c in enumerate(top1_idx)
                ]
            else:
                texts = None
            grid_fig.add_trace(
                go.Scatter(
                    x=coords_view[:, 0],
                    y=coords_view[:, 1],
                    mode="markers+text" if num_in > 1 else "markers",
                    text=texts,
                    textposition="middle center",
                    textfont=dict(size=label_font_size_example, color="black"),
                    marker=dict(
                        size=12,
                        line=dict(color="Black", width=1),
                        color=top1_val,
                        colorscale="Fall_r",
                        cmin=vmin,
                        cmax=vmax,
                        opacity=1.0,
                        showscale=not colorbar_added,
                        colorbar=(
                            dict(
                                x=1.0,
                                len=0.7,
                                thickness=14,
                                outlinewidth=0,
                                ticks="outside",
                                tickmode="array",
                                tickvals=cb_vals_ex,
                                ticktext=cb_text_ex,
                                tickfont=dict(size=26, family="Courier New, monospace"),
                            )
                            if not colorbar_added
                            else None
                        ),
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            colorbar_added = True
        # Hide axes, anchor to be square per subplot
        for r in range(1, nrows + 1):
            for c in range(1, ncols + 1):
                ax_key = "" if (r == 1 and c == 1) else f"{(r - 1) * ncols + c}"
                grid_fig.update_xaxes(
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    row=r,
                    col=c,
                )
                grid_fig.update_yaxes(
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    scaleanchor=f"x{ax_key}",
                    scaleratio=1,
                    row=r,
                    col=c,
                )
        grid_fig.update_layout(
            width=grid_width,
            height=grid_height,
            autosize=False,
            margin=dict(l=10, r=10, t=20, b=0),
            template="simple_white",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        grid_fig.write_image(
            f"results/{EXP}/{EXP}_example_{num_example}.pdf",
            width=grid_width,
            height=grid_height,
        )
        grid_fig.write_image(
            f"results/{EXP}/{EXP}_example_{num_example}.svg",
            width=grid_width,
            height=grid_height,
        )

    if EXP == "beta2d":
        # Single-output: accumulate for a combined figure after loop
        vals_ic = output_maps[0]
        abs_vals = np.abs(vals_ic)
        top1_idx = np.argmax(abs_vals, axis=1)
        top1_val = vals_ic[np.arange(vals_ic.shape[0]), top1_idx]
        # Remove disconnected nodes (nodes not incident to any edge)
        node_connected_mask = np.zeros(coords_view.shape[0], dtype=bool)
        node_connected_mask[src_idx] = True
        node_connected_mask[dst_idx] = True
        coords_view_plot = coords_view[node_connected_mask]
        top1_idx_plot = top1_idx[node_connected_mask]
        top1_val_plot = top1_val[node_connected_mask]
        single_out_examples.append(
            dict(
                x_lines=x_lines,
                y_lines=y_lines,
                coords_view=coords_view_plot.copy(),
                top1_idx=top1_idx_plot.copy(),
                top1_val=top1_val_plot.copy(),
                num_in=num_in,
                title=(mol_title if "mol_title" in locals() else ""),
            )
        )
        # Update global range using the actual plotted values
        if top1_val.size > 0:
            single_out_global_min = min(
                single_out_global_min, float(np.nanmin(top1_val))
            )
            single_out_global_max = max(
                single_out_global_max, float(np.nanmax(top1_val))
            )

    num_example += 1
    if num_example >= MAX_EXAMPLES:
        break

# If we are in the single-output case, render one combined figure with examples as subplots
if EXP == "beta2d":
    from plotly.subplots import make_subplots

    ncols = max(5, len(single_out_examples))
    nrows = int(np.ceil(len(single_out_examples) / ncols))
    hspace = 0.03
    vspace = 0.0
    # Build subplot titles from molecule names (Title Case)
    _total_cells = nrows * ncols
    _titles = []
    for _i in range(_total_cells):
        if _i < len(single_out_examples):
            _titles.append(single_out_examples[_i].get("title", ""))
        else:
            _titles.append("")
    grid_fig = make_subplots(
        rows=nrows,
        cols=ncols,
        horizontal_spacing=hspace,
        vertical_spacing=vspace,
        subplot_titles=_titles,
    )
    target_side = 200
    domain_w = (1.0 - hspace * (ncols - 1)) / ncols
    domain_h = (1.0 - vspace * (nrows - 1)) / nrows
    grid_height = int(target_side * nrows / domain_h)
    grid_width = int(target_side * ncols / domain_w) / 4
    # Global symmetric range across examples (based on plotted top1 values)
    max_abs = float(
        max(
            abs(single_out_global_min if np.isfinite(single_out_global_min) else 0.0),
            abs(single_out_global_max if np.isfinite(single_out_global_max) else 0.0),
        )
    )
    cmin = -max_abs
    cmax = max_abs
    colorbar_added = False
    # Shared colorbar ticks for combined examples figure
    cb_vals_ex2 = np.linspace(cmin, cmax, 3).tolist()
    cb_text_ex2 = [f"{v:+.2f}" for v in cb_vals_ex2]
    for idx, ex in enumerate(single_out_examples):
        row = idx // ncols + 1
        col = idx % ncols + 1
        x_lines = ex["x_lines"]
        y_lines = ex["y_lines"]
        coords_view = ex["coords_view"]
        top1_idx = ex["top1_idx"]
        top1_val = ex["top1_val"]
        num_in = ex["num_in"]
        # Edges
        grid_fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color="rgb(50,50,50)", width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        # Points with labels if multiple input channels; label only if abs value in upper 50 percentile
        if num_in > 1:
            abs_top1 = np.abs(top1_val)
            thr = np.max(abs_top1) * 0.2  # float(np.quantile(abs_top1, 0.8))
            texts = [
                (
                    f"<b>{(CHANNEL_NAMES[c] if CHANNEL_NAMES else str(c))}</b>"
                    if abs_top1[i] >= thr
                    else ""
                )
                for i, c in enumerate(top1_idx)
            ]
        else:
            texts = None
        # Molecule diameter for this example (from rotated coords)
        diam_x = float(coords_view[:, 0].max() - coords_view[:, 0].min())
        diam_y = float(coords_view[:, 1].max() - coords_view[:, 1].min())
        mol_diam = max(diam_x, diam_y) if max(diam_x, diam_y) > 0 else 1.0
        size_k = 160.0
        size_min, size_max = 4.0, 24.0
        marker_size_example = float(np.clip(size_k / mol_diam, size_min, size_max))
        label_font_size_example = int(np.clip(marker_size_example * 0.7, 8.0, 32.0))
        grid_fig.add_trace(
            go.Scatter(
                x=coords_view[:, 0],
                y=coords_view[:, 1],
                mode="markers+text" if num_in > 1 else "markers",
                text=texts,
                textposition="middle center",
                textfont=dict(size=label_font_size_example, color="white"),
                marker=dict(
                    size=marker_size_example,
                    line=dict(color="Black", width=1),
                    color=top1_val,
                    colorscale="Fall_r",
                    cmin=cmin,
                    cmax=cmax,
                    opacity=1.0,
                    showscale=not colorbar_added,
                    colorbar=(
                        dict(
                            x=1.0,
                            len=0.7,
                            thickness=14,
                            outlinewidth=0,
                            ticks="outside",
                            tickmode="array",
                            tickvals=cb_vals_ex2,
                            ticktext=cb_text_ex2,
                            tickfont=dict(size=24, family="Courier New, monospace"),
                        )
                        if not colorbar_added
                        else None
                    ),
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        colorbar_added = True
    # Hide axes and anchor square
    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            ax_key = "" if (r == 1 and c == 1) else f"{(r - 1) * ncols + c}"
            grid_fig.update_xaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=r,
                col=c,
            )
            grid_fig.update_yaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                scaleanchor=f"x{ax_key}",
                scaleratio=1,
                row=r,
                col=c,
            )
    grid_fig.update_layout(
        width=grid_width,
        height=grid_height,
        autosize=False,
        margin=dict(l=5, r=5, t=25, b=10),
        template="simple_white",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    # Save one combined figure across all examples
    os.makedirs(f"results/{EXP}", exist_ok=True)
    grid_fig.write_image(
        f"results/{EXP}/{EXP}_examples.pdf",
        width=grid_width,
        height=grid_height,
    )
    grid_fig.write_image(
        f"results/{EXP}/{EXP}_examples.svg",
        width=grid_width,
        height=grid_height,
    )
    grid_fig.write_image(
        f"results/{EXP}/{EXP}_examples.png",
        width=grid_width,
        height=grid_height,
    )

import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from jinja2 import Template
from typing import List, Dict, Optional
import importlib.resources
import base64
import io

from .report import (
    compute_per_position_ic,
    _plot_weights,
    tomtomlite_dataframe,
    generate_tomtom_dataframe,
)
from memelite.io import read_meme


def plot_to_base64(array, figsize=(10, 3), clamp=True, highest_nucleotide=None):
    """Plot weights as a sequence logo and return as base64 string."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    import pandas

    df = pandas.DataFrame(array, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    import logomaker

    crp_logo = logomaker.Logo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    if clamp:
        plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

    if highest_nucleotide is not None:
        start = highest_nucleotide - 0.5
        height = df.sum(axis=1).max()
        height_correction = height * 0.1  # Make it a bit higher
        rec = Rectangle(
            (start, 0 - height_correction),
            1,
            height + (height_correction * 2),
            facecolor="#9AA7B8",
            alpha=0.2,
            clip_on=False,
            zorder=0,
        )
        ax.add_patch(rec)

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def plot_spatial_distributions_to_base64(data, bins=30, figsize=(8, 12)):

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    for ax in axes.flatten():
        ax.spines[["top", "right"]].set_visible(False)

    # Calculate seqlet center positions
    seqlet_centers = (data["seqlet_starts"] + data["seqlet_ends"]) / 2
    track_centers = data["track_lengths"] / 2

    # Get position of seqlet relative to center and end of input sequences
    relative_to_center = seqlet_centers - track_centers
    relative_to_ends = data["seqlet_ends"] - data["track_lengths"]

    # Set bounds based on global region size with center at 0
    lim = max(np.abs(relative_to_center)) + 1
    center_xlim = (-lim, lim)

    sns.histplot(
        relative_to_center,
        alpha=0.7,
        ax=axes[0],
        stat="density",
        kde=True,
        line_kws={"color": "black"},
        bins=bins,
        binrange=center_xlim,
    )
    axes[0].set_xlabel("Nucleotide")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distance from Sequence Centers")
    axes[0].set_xlim(center_xlim)

    max_value = max(max(data["seqlet_starts"]), max(abs(relative_to_ends)))
    sns.histplot(
        data["seqlet_starts"],
        alpha=0.7,
        ax=axes[1],
        stat="density",
        kde=True,
        line_kws={"color": "black"},
        bins=bins,
        binrange=(0, max_value),
    )
    axes[1].set_xlabel("Nucleotide")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distance from Sequence Starts")
    axes[1].set_xlim((0, max_value + 1))

    sns.histplot(
        relative_to_ends,
        alpha=0.7,
        ax=axes[2],
        stat="density",
        kde=True,
        line_kws={"color": "black"},
        bins=bins,
        binrange=(-max_value, 0),
    )
    axes[2].set_xlabel("Nucleotide")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Distance from Sequence Ends")
    axes[2].set_xlim((-max_value - 1, 0))

    fig.tight_layout()

    # Save to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def plot_histogram_to_base64(
    data,
    bins=30,
    xlabel="",
    ylabel="Density",
    title="",
    figsize=(8, 4),
    xlim=None,
):
    """Create histogram plot and return as base64 string."""
    _, ax = plt.subplots(figsize=figsize)

    if max(data) - min(data) > bins:
        bins = max(data) - min(data)

    sns.histplot(
        data,
        alpha=0.7,
        edgecolor="black",
        stat="density",
        kde=True,
        discrete=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

    if xlim is not None:
        ax.set_xlim(xlim)

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def plot_entropy_to_base64(
    data,
    title="",
    figsize=(8, 4),
    xlim=None,
):
    """Create histogram plot and return as base64 string."""
    _, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        data,
        x="values",
        hue="distribution" if "null" in data["distribution"].values else None,
        alpha=0.7,
        edgecolor="black",
        stat="density",
        bins=20,
        kde=True,
    )
    ax.set_xlabel("Entropy scores")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

    if xlim is not None:
        ax.set_xlim(xlim)

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def plot_pie_to_base64(data, figsize):
    """
    Create a pie chart with given data and save to base64.
    """
    frame_labels = [0, 1, 2]
    counts = [(data == frame).sum() for frame in frame_labels]
    frame_label = ["+" + str(i) for i in frame_labels]

    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        counts,
        colors=sns.color_palette("Set2"),
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        pctdistance=1.3,
    )
    ax.legend(
        wedges,
        frame_label,
        title="Frames",
        loc="center left",
        bbox_to_anchor=(1.2, 0, 0.5, 1),
    )

    # save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def extract_seqlet_data(modisco_h5py: str, pattern_groups: List[str]) -> Dict:
    """Extract seqlet data for descriptive analysis."""
    with h5py.File(modisco_h5py, "r") as modisco_results:
        patterns_data = {}

        for contribution_dir_name in pattern_groups:
            if contribution_dir_name not in modisco_results.keys():
                continue

            metacluster = modisco_results[contribution_dir_name]

            def sort_key(x):
                return int(x[0].split("_")[-1])

            for pattern_name, pattern in sorted(metacluster.items(), key=sort_key):  # type: ignore
                pattern_tag = f"{contribution_dir_name}.{pattern_name}"

                # Extract basic pattern data
                ppm = np.array(pattern["sequence"][:])
                cwm = np.array(pattern["contrib_scores"][:])
                hcwm = np.array(pattern["hypothetical_contribs"][:])

                # Extract seqlet information
                seqlets_grp = pattern["seqlets"]
                n_seqlets = seqlets_grp["n_seqlets"][:][0]

                entropy_scores = np.array(seqlets_grp["entropy_scores"][:])

                # Get seqlet positions and data if available
                padding_size = np.array(seqlets_grp.get("padding", []))
                seqlet_starts = np.array(seqlets_grp.get("start", []))
                seqlet_ends = np.array(seqlets_grp.get("end", []))
                track_lengths = np.array(seqlets_grp.get("track_lengths", []))
                seqlet_example_idx = np.array(seqlets_grp.get("example_idx", []))
                track_frames = np.array(seqlets_grp.get("frames", []))

                # Get seqlet contribution scores and calculate importance
                seqlet_contribs = []
                seqlet_importance = []
                if "contrib_scores" in seqlets_grp:
                    seqlet_contribs = np.array(seqlets_grp["contrib_scores"][:])

                    # Calculate total importance as sum of absolute contribution scores
                    for seqlet_contrib in seqlet_contribs:
                        importance = np.sum(seqlet_contrib)
                        seqlet_importance.append(importance)

                # Calculate pattern statistics
                gc_content = np.mean(ppm[:, [1, 2]])  # C and G positions
                avg_importance = np.mean(np.sum(cwm, axis=1))
                avg_entropy = np.mean(entropy_scores)
                std_entropy = np.std(entropy_scores)

                # Calculate standard deviations for seqlet statistics
                std_importance = np.nan
                if len(seqlet_importance) > 0:
                    std_importance = np.std(seqlet_importance)

                # Get frames of the highest contributing nucleotide
                highest_contributing_position = np.argmax(cwm)
                seqlet_frames = (
                    track_frames
                    + seqlet_starts
                    + padding_size
                    + highest_contributing_position
                ) % 3

                # Store seqlet positions for global region size calculation
                # Also correct the coordinate by removing the right padding of the input
                # sequences
                padding_size = np.array(padding_size) if len(padding_size) > 0 else 0

                seqlet_starts = (
                    seqlet_starts - padding_size
                    if len(seqlet_starts) > 0
                    else np.array([])
                )
                seqlet_ends = (
                    seqlet_ends - padding_size if len(seqlet_ends) > 0 else np.array([])
                )
                track_lengths = (
                    track_lengths - (padding_size * 2)
                    if len(track_lengths) > 0
                    else np.array([])
                )

                # Distance are being attributed np.nan values as placeholder.
                # Real values will be computed globablly later
                patterns_data[pattern_tag] = {
                    "ppm": ppm,
                    "cwm": cwm,
                    "hcwm": hcwm,
                    "n_seqlets": n_seqlets,
                    "gc_content": gc_content,
                    "avg_importance": avg_importance,
                    "std_importance": std_importance,
                    "avg_entropy": avg_entropy,
                    "std_entropy": std_entropy,
                    "median_abs_distance_from_center": np.nan,
                    "std_distance_from_center": np.nan,
                    "median_distance_from_start": np.nan,
                    "std_distance_from_start": np.nan,
                    "median_distance_from_end": np.nan,
                    "std_distance_from_end": np.nan,
                    "seqlet_starts": seqlet_starts,
                    "seqlet_ends": seqlet_ends,
                    "seqlet_frames": seqlet_frames,
                    "track_lengths": track_lengths,
                    "padding_sizes": padding_size,
                    "entropy_scores": entropy_scores,
                    "seqlet_example_idx": np.array(seqlet_example_idx)
                    if len(seqlet_example_idx) > 0
                    else np.array([]),
                    "seqlet_importance": np.array(seqlet_importance)
                    if len(seqlet_importance) > 0
                    else np.array([]),
                    "seqlet_contribs": seqlet_contribs,
                }

    return patterns_data


def compute_distances(patterns_data: Dict) -> Dict:
    """
    Compute the spatial distribution data.
    """
    # Update patterns data with global information and compute distances from center
    updated_patterns_data = patterns_data.copy()
    for pattern_tag, data in updated_patterns_data.items():
        # Compute median absolute distance from global center and standard deviation
        if len(data["seqlet_starts"]) > 0 and len(data["seqlet_ends"]) > 0:
            # Calculate seqlet center positions
            seqlet_centers = (data["seqlet_starts"] + data["seqlet_ends"]) / 2
            track_centers = data["track_lengths"] / 2

            # Calculate distances from local center
            distances_from_center = np.abs(seqlet_centers - track_centers)
            data["median_abs_distance_from_center"] = np.median(distances_from_center)
            data["std_distance_from_center"] = np.std(distances_from_center)

            # Calculate distance from start
            data["median_distance_from_start"] = np.median(data["seqlet_starts"])
            data["std_distance_from_start"] = np.std(data["seqlet_starts"])

            # Calculate distances from end
            distances_from_end = data["seqlet_ends"] - data["track_lengths"]
            # Correct distances for seqlet which were extanded in the padding
            if any(distances_from_end > data["padding_sizes"]):
                raise ValueError("Some seqlets have coordinate going over the padding.")
            distances_from_end = np.where(distances_from_end > 0, 0, distances_from_end)

            data["median_distance_from_end"] = np.median(distances_from_end)
            data["std_distance_from_end"] = np.std(distances_from_end)

        else:
            data["median_abs_distance_from_center"] = np.nan
            data["std_distance_from_center"] = np.nan
            data["median_distance_from_start"] = np.nan
            data["std_distance_from_start"] = np.nan
            data["median_distance_from_end"] = np.nan
            data["std_distance_from_end"] = np.nan

    return updated_patterns_data


def create_logos(
    patterns_data: Dict, output_dir: str, trim_threshold: float = 0.3
) -> Dict:
    """Create logo visualizations for each pattern as both files and base64 data."""
    logo_dir = os.path.join(output_dir, "logos")
    os.makedirs(logo_dir, exist_ok=True)

    logo_data = {}

    for pattern_tag, data in patterns_data.items():
        pattern_dir = os.path.join(logo_dir, pattern_tag)
        os.makedirs(pattern_dir, exist_ok=True)

        cwm = data["cwm"]
        hcwm = data["hcwm"]
        ppm = data["ppm"]

        highest_nucleotide = np.argmax(cwm.sum(axis=1))

        # Calculate trimmed version
        score = np.sum(cwm, axis=1)
        trim_thresh = np.max(score) * trim_threshold
        pass_inds = np.where(score >= trim_thresh)[0]

        if len(pass_inds) > 0:
            start_trim = max(np.min(pass_inds) - 2, 0)
            end_trim = min(np.max(pass_inds) + 3, len(score))
            trimmed_cwm = cwm[start_trim:end_trim]
            trimmed_highest_nucleotide = np.argmax(trimmed_cwm.sum(axis=1))
        else:
            trimmed_cwm = cwm
            trimmed_highest_nucleotide = highest_nucleotide

        # Generate logos as both files and base64
        logos = {}

        # CWM Logo
        cwm_path = os.path.join(pattern_dir, "cwm_logo.png")
        _plot_weights(cwm, cwm_path, figsize=(12, 3))
        logos["cwm"] = plot_to_base64(
            cwm, figsize=(12, 3), highest_nucleotide=highest_nucleotide
        )
        logos["cwm_path"] = cwm_path

        # hCWM Logo
        hcwm_path = os.path.join(pattern_dir, "hcwm_logo.png")
        _plot_weights(hcwm, hcwm_path, figsize=(12, 3), clamp=False)
        logos["hcwm"] = plot_to_base64(
            hcwm, figsize=(12, 3), clamp=False, highest_nucleotide=highest_nucleotide
        )
        logos["hcwm_path"] = hcwm_path

        # IC-scaled PPM Logo (information-weighted PPM)
        background = np.array([0.25, 0.25, 0.25, 0.25])
        ic = compute_per_position_ic(ppm, background, 0.001)
        ic_ppm_path = os.path.join(pattern_dir, "ic_ppm_logo.png")
        _plot_weights(ppm * ic[:, None], ic_ppm_path, figsize=(12, 3))
        logos["ic_ppm"] = plot_to_base64(
            ppm * ic[:, None], figsize=(12, 3), highest_nucleotide=highest_nucleotide
        )
        logos["ic_ppm_path"] = ic_ppm_path

        # Keep PWM alias for backwards compatibility
        logos["pwm"] = logos["ic_ppm"]
        logos["pwm_path"] = logos["ic_ppm_path"]

        # Trimmed CWM Logo (Forward)
        trimmed_path = os.path.join(pattern_dir, "trimmed_cwm_fwd_logo.png")
        _plot_weights(trimmed_cwm, trimmed_path, figsize=(10, 3))
        logos["trimmed_cwm_fwd"] = plot_to_base64(
            trimmed_cwm, figsize=(10, 3), highest_nucleotide=trimmed_highest_nucleotide
        )
        logos["trimmed_cwm_fwd_path"] = trimmed_path

        # Trimmed CWM Logo (Reverse)
        cwm_rev = cwm[::-1, ::-1]
        score_rev = np.sum(np.abs(cwm_rev), axis=1)
        trim_thresh_rev = np.max(score_rev) * trim_threshold
        pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

        if len(pass_inds_rev) > 0:
            start_trim_rev = max(np.min(pass_inds_rev) - 2, 0)
            end_trim_rev = min(np.max(pass_inds_rev) + 3, len(score_rev))
            trimmed_cwm_rev = cwm_rev[start_trim_rev:end_trim_rev]
        else:
            trimmed_cwm_rev = cwm_rev

        trimmed_rev_path = os.path.join(pattern_dir, "trimmed_cwm_rev_logo.png")
        _plot_weights(trimmed_cwm_rev, trimmed_rev_path, figsize=(10, 3))
        logos["trimmed_cwm_rev"] = plot_to_base64(trimmed_cwm_rev, figsize=(10, 3))
        logos["trimmed_cwm_rev_path"] = trimmed_rev_path

        # Keep original for backwards compatibility
        logos["trimmed_cwm"] = logos["trimmed_cwm_fwd"]
        logos["trimmed_cwm_path"] = logos["trimmed_cwm_fwd_path"]

        logo_data[pattern_tag] = logos

    return logo_data


def create_distribution_plots(
    patterns_data: Dict, output_dir: str, null_entropy_distribution: np.ndarray | None
) -> Dict:
    """Create seqlet importance and spatial distribution plots as base64 data."""
    distribution_data = {}

    for pattern_tag, data in patterns_data.items():
        plots = {}

        # Seqlet importance distribution
        if len(data["seqlet_importance"]) > 0:
            plots["importance"] = plot_histogram_to_base64(
                data["seqlet_importance"],
                bins=30,
                xlabel="Seqlet Total Contribution Score",
                ylabel="Density",
                title=f"Seqlet Contribution Score Distribution - {pattern_tag}",
                figsize=(8, 4),
            )

            plots["spatial_distributions"] = plot_spatial_distributions_to_base64(
                data,
                bins=30,
                figsize=(8, 10),
            )

            plots["frames_proportion"] = plot_pie_to_base64(
                data["seqlet_frames"],
                figsize=(8, 4),
            )

            if null_entropy_distribution is not None:
                entropy_df = pd.concat(
                    [
                        pd.DataFrame(
                            {"values": data["entropy_scores"], "distribution": "true"}
                        ),
                        pd.DataFrame(
                            {
                                "values": null_entropy_distribution,
                                "distribution": "null",
                            }
                        ),
                    ]
                )
            else:
                entropy_df = pd.DataFrame(
                    {"values": data["entropy_scores"], "distribution": "true"}
                )
            plots["entropy_scores"] = plot_entropy_to_base64(
                entropy_df,
                title="Seqlet Entropy Score Distribution",
                figsize=(8, 4),
            )

        distribution_data[pattern_tag] = plots

    return distribution_data


def create_seqlet_example_logos(
    patterns_data: Dict, output_dir: str, n_examples: int = 10
) -> Dict:
    """Create logo plots of seqlets from importance score quantiles to show distribution."""
    examples_dir = os.path.join(output_dir, "seqlet_examples")
    os.makedirs(examples_dir, exist_ok=True)

    examples_data = {}

    for pattern_tag, data in patterns_data.items():
        if len(data["seqlet_importance"]) == 0 or len(data["seqlet_contribs"]) == 0:
            continue

        # Get the position of the highest score nucleotide in the motif
        highest_nucleotide = np.argmax(data["cwm"].sum(axis=1))
        pattern_dir = os.path.join(examples_dir, pattern_tag)
        os.makedirs(pattern_dir, exist_ok=True)

        # Get seqlets representing quantiles of importance distribution
        importance_scores = data["seqlet_importance"]
        seqlet_contribs = data["seqlet_contribs"]

        # Adjust n_examples if we have fewer seqlets than requested
        actual_n_examples = min(n_examples, len(importance_scores))

        # Calculate quantile percentiles (10%, 20%, ..., 100%)
        quantiles = np.linspace(10, 100, actual_n_examples)
        quantile_indices = []
        used_indices = set()

        for quantile in quantiles:
            percentile_value = np.percentile(importance_scores, quantile)
            # Find all seqlets close to this percentile value
            distances = np.abs(importance_scores - percentile_value)
            sorted_indices = np.argsort(distances)

            # Find the closest unused index
            closest_idx = None
            for idx in sorted_indices:
                if idx not in used_indices:
                    closest_idx = idx
                    break

            if closest_idx is not None:
                quantile_indices.append(closest_idx)
                used_indices.add(closest_idx)

        example_logos = []

        # Ensure we have the right number of quantile_indices
        for i in range(len(quantile_indices)):
            if i < len(quantiles):
                idx = quantile_indices[i]
                if idx < len(seqlet_contribs) and idx < len(importance_scores):
                    # Extract individual seqlet contribution scores
                    seqlet_cwm = seqlet_contribs[idx]
                    importance = importance_scores[idx]
                    quantile_pct = quantiles[i]

                    # Create logo file for backwards compatibility
                    logo_path = os.path.join(
                        pattern_dir, f"quantile_{int(quantile_pct)}.png"
                    )
                    _plot_weights(seqlet_cwm, logo_path, figsize=(8, 1.2))

                    # Also create base64 data
                    base64_data = plot_to_base64(
                        seqlet_cwm,
                        figsize=(8, 1.2),
                        highest_nucleotide=highest_nucleotide,
                    )

                    example_logos.append(
                        {
                            "rank": i + 1,
                            "quantile": int(quantile_pct),
                            "path": logo_path,
                            "base64": base64_data,
                            "importance": float(importance),
                        }
                    )

        examples_data[pattern_tag] = list(reversed(example_logos))

    return examples_data


def create_tomtom_match_logos(
    tomtom_data: Dict, output_dir: str, meme_motif_db: str, top_n_matches: int
) -> Dict:
    """Create logo plots for Tomtom matches."""
    tomtom_logos_dir = os.path.join(output_dir, "tomtom_logos")
    os.makedirs(tomtom_logos_dir, exist_ok=True)

    # Read the motif database
    motifs = read_meme(meme_motif_db)
    motifs = {name.split()[0]: pwm.T for name, pwm in motifs.items()}

    tomtom_logos = {}
    background = np.array([0.25, 0.25, 0.25, 0.25])

    for pattern_tag, matches in tomtom_data.items():
        tomtom_logos[pattern_tag] = {}
        for i in range(top_n_matches):
            match_key = f"match_{i}"
            if match_key in matches and matches[match_key]:
                match_name = matches[match_key].strip()
                if match_name in motifs:
                    # Create logo for this match
                    ppm = motifs[match_name]
                    ic = compute_per_position_ic(ppm, background, 0.001)

                    # Create file for backwards compatibility
                    logo_path = os.path.join(
                        tomtom_logos_dir, f"{pattern_tag}_match_{i}.png"
                    )
                    _plot_weights(ppm * ic[:, None], logo_path, figsize=(8, 2))
                    tomtom_logos[pattern_tag][f"match_{i}_logo"] = logo_path

                    # Also create base64 data
                    base64_data = plot_to_base64(ppm * ic[:, None], figsize=(8, 2))
                    tomtom_logos[pattern_tag][f"match_{i}_base64"] = base64_data

    return tomtom_logos


def create_descriptive_names(tomtom_data: Dict, top_n_matches: int = 3) -> Dict:
    """Create descriptive names for motifs based on Tomtom matches."""
    descriptive_names = {}

    for pattern_tag, matches in tomtom_data.items():
        # Collect first 10 characters of each match
        name_parts = []
        for i in range(min(top_n_matches, 3)):  # Use max 3 matches for name
            match_key = f"match_{i}"
            if match_key in matches and matches[match_key]:
                match_name = matches[match_key].strip()
                # Take first 10 characters
                name_parts.append(match_name[:10])

        if name_parts:
            descriptive_names[pattern_tag] = ";".join(name_parts)
        else:
            # Fallback to pattern tag if no matches
            descriptive_names[pattern_tag] = pattern_tag

    return descriptive_names


def generate_descriptive_report(
    modisco_h5py: str,
    output_dir: str,
    null_entropy_distribution: Optional[np.ndarray] | None = None,
    img_path_suffix: str = "./",
    meme_motif_db: Optional[str] = None,
    top_n_matches: int = 3,
    ttl: bool = False,
    n_examples: int = 10,
    trim_threshold: float = 0.3,
):
    """Generate descriptive HTML report."""

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    pattern_groups = ["pos_patterns", "neg_patterns"]

    # Extract seqlets
    patterns_data = extract_seqlet_data(modisco_h5py, pattern_groups)

    # Compute global region size and update distances
    patterns_data = compute_distances(patterns_data)

    # Create visualizations
    logo_paths = create_logos(patterns_data, output_dir, trim_threshold)
    distribution_paths = create_distribution_plots(
        patterns_data, output_dir, null_entropy_distribution
    )
    examples_data = create_seqlet_example_logos(patterns_data, output_dir, n_examples)

    # Get Tomtom matches if database provided
    tomtom_data = {}
    tomtom_logos = {}
    descriptive_names = {}
    if meme_motif_db is not None:
        from pathlib import Path

        if ttl:
            tomtom_df = tomtomlite_dataframe(
                Path(modisco_h5py),
                Path(output_dir),
                Path(meme_motif_db) if meme_motif_db else None,
                pattern_groups=pattern_groups,
                top_n_matches=top_n_matches,
                trim_threshold=trim_threshold,
            )
        else:
            tomtom_df = generate_tomtom_dataframe(
                Path(modisco_h5py),
                Path(output_dir),
                Path(meme_motif_db) if meme_motif_db else None,
                is_writing_tomtom_matrix=False,
                pattern_groups=pattern_groups,
                top_n_matches=top_n_matches,
                trim_threshold=trim_threshold,
            )

        # Convert to dictionary format
        for i, (pattern_tag, _) in enumerate(patterns_data.items()):
            if i < len(tomtom_df):
                tomtom_data[pattern_tag] = {}
                for j in range(top_n_matches):
                    match_col = f"match{j}"
                    pval_col = f"pval{j}" if ttl else f"qval{j}"
                    if match_col in tomtom_df.columns:
                        tomtom_data[pattern_tag][f"match_{j}"] = tomtom_df.iloc[i][
                            match_col
                        ]
                        tomtom_data[pattern_tag][f"pval_{j}"] = tomtom_df.iloc[i][
                            pval_col
                        ]

        # Create TOMTOM match logos and descriptive names
        tomtom_logos = create_tomtom_match_logos(
            tomtom_data, output_dir, meme_motif_db, top_n_matches
        )
        descriptive_names = create_descriptive_names(tomtom_data, top_n_matches)

    from . import templates

    # Generate HTML report
    template_str = (
        importlib.resources.files(templates)
        .joinpath("descriptive_report.html")
        .read_text()
    )
    template = Template(template_str)
    html_content = template.render(
        patterns_data=patterns_data,
        logo_paths=logo_paths,
        distribution_paths=distribution_paths,
        examples_data=examples_data,
        tomtom_data=tomtom_data,
        tomtom_logos=tomtom_logos,
        descriptive_names=descriptive_names,
        img_path_suffix=img_path_suffix,
        meme_motif_db=meme_motif_db,
        top_n_matches=top_n_matches,
        ttl=ttl,
    )

    # Write HTML report
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"Report generated: {report_path}")
    return report_path

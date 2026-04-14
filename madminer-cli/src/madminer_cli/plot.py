from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from madminer_cli import LOGGER

_FOURVEC_SUFFIXES = ("_E", "_px", "_py", "_pz")
_JET_PREFIX_RE = re.compile(r"^(j\d+)_(E|pt)$")


def _convert_to_ptetaphi(
    x: np.ndarray, obs_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Replace each ``<prefix>_{E,px,py,pz}`` quartet with ``<prefix>_{pt,eta,phi}``.

    Columns that don't form a complete quartet are passed through unchanged.
    """
    name_to_idx = {n: i for i, n in enumerate(obs_names)}
    consumed = set()
    new_cols: List[np.ndarray] = []
    new_names: List[str] = []

    for i, name in enumerate(obs_names):
        if i in consumed:
            continue
        if not name.endswith("_E"):
            new_cols.append(x[:, i])
            new_names.append(name)
            continue
        prefix = name[:-2]
        quartet = [name_to_idx.get(prefix + s) for s in _FOURVEC_SUFFIXES]
        if any(q is None for q in quartet):
            new_cols.append(x[:, i])
            new_names.append(name)
            continue

        iE, ipx, ipy, ipz = quartet
        E = x[:, iE]
        px, py, pz = x[:, ipx], x[:, ipy], x[:, ipz]
        # Missing jets/leptons are padded as zeros (or NaN); treat E <= 0 as absent
        # so the derived pt/eta/phi stay NaN and downstream masks reject the row.
        missing = ~np.isfinite(E) | (E <= 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = np.sqrt(px * px + py * py)
            p = np.sqrt(pt * pt + pz * pz)
            ratio = np.where(p > 0, pz / p, np.nan)
            ratio = np.clip(ratio, -1.0 + 1e-15, 1.0 - 1e-15)
            eta = np.arctanh(ratio)
            phi = np.arctan2(py, px)
        pt = np.where(missing, np.nan, pt)
        eta = np.where(missing, np.nan, eta)
        phi = np.where(missing, np.nan, phi)

        new_cols.extend([pt, eta, phi])
        new_names.extend([f"{prefix}_pt", f"{prefix}_eta", f"{prefix}_phi"])
        consumed.update(quartet)

    return np.column_stack(new_cols), new_names


def _jet_multiplicity(x: np.ndarray, obs_names: List[str]) -> Optional[np.ndarray]:
    """Per-event jet count, inferred from finite ``j<i>_E`` (or ``j<i>_pt``) columns."""
    jets = {}
    for i, name in enumerate(obs_names):
        m = _JET_PREFIX_RE.match(name)
        if m:
            jets[m.group(1)] = i
    if not jets:
        return None
    cols = np.column_stack([x[:, i] for i in jets.values()])
    # A jet is "present" if its E (or pt) is finite and strictly positive;
    # missing jets are zero-padded in the raw madminer output.
    present = np.isfinite(cols) & (cols > 0)
    return np.sum(present, axis=1).astype(int)


def _plot_jet_multiplicity_grid(
    njets: np.ndarray,
    weights_per_group: np.ndarray,
    group_labels: List[str],
    out_path: Path,
    title: str,
    normalize: bool,
    log: bool,
) -> None:
    n_max = int(njets.max()) if njets.size else 0
    bins = np.arange(-0.5, n_max + 1.5, 1.0)
    centers = np.arange(0, n_max + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    width = 0.8 / max(1, weights_per_group.shape[1])
    for b, label in enumerate(group_labels):
        w = weights_per_group[:, b]
        mask = np.isfinite(w)
        hist, _ = np.histogram(
            njets[mask], bins=bins, weights=w[mask], density=normalize
        )
        offset = (b - (weights_per_group.shape[1] - 1) / 2.0) * width
        ax.bar(centers + offset, hist, width=width, label=label, alpha=0.85)

    ax.set_xticks(centers)
    ax.set_xlabel("number of jets per event")
    ax.set_ylabel("normalized" if normalize else "Σ weights")
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _weighted_quantile_range(
    values: np.ndarray, weights: np.ndarray, qs: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """Return (lo, hi) weighted quantiles, or None if not computable."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() < 2:
        return None
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if not np.isfinite(total) or total <= 0:
        return None
    cdf = cw / total
    lo = float(np.interp(qs[0], cdf, v))
    hi = float(np.interp(qs[1], cdf, v))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return lo, hi


def _plot_one(
    ax,
    values: np.ndarray,
    weights_per_bench: np.ndarray,
    bench_labels: List[str],
    n_bins: int,
    log: bool,
    normalize: bool,
    quantiles_for_range: Tuple[float, float],
    name: str,
) -> bool:
    """Plot one observable on ``ax``. Returns True on success."""
    ranges = []
    for b in range(weights_per_bench.shape[1]):
        r = _weighted_quantile_range(values, weights_per_bench[:, b], quantiles_for_range)
        if r is not None:
            ranges.append(r)
    if not ranges:
        return False
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    if hi <= lo:
        return False

    bins = np.linspace(lo, hi, n_bins + 1)
    finite = np.isfinite(values)
    for b, label in enumerate(bench_labels):
        w = weights_per_bench[:, b]
        mask = finite & np.isfinite(w)
        if not mask.any():
            continue
        hist, edges = np.histogram(
            values[mask], bins=bins, weights=w[mask], density=normalize
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, hist, where="mid", label=label, linewidth=1.3)

    ax.set_xlabel(name)
    ax.set_ylabel("normalized" if normalize else "dσ / bin [pb]")
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=7, loc="best")
    return True


def _plot_observable_grid(
    x: np.ndarray,
    obs_idx: List[int],
    obs_names: List[str],
    weights_per_group: np.ndarray,
    group_labels: List[str],
    out_path: Path,
    title: str,
    n_bins: int,
    n_cols: int,
    log: bool,
    normalize: bool,
    quantiles_for_range: Tuple[float, float],
) -> List[str]:
    n = len(obs_idx)
    if n == 0:
        raise ValueError("No observables selected for plotting")

    n_cols_eff = max(1, min(n_cols, n))
    n_rows = math.ceil(n / n_cols_eff)
    fig, axes = plt.subplots(
        n_rows, n_cols_eff, figsize=(5 * n_cols_eff, 3.5 * n_rows), squeeze=False
    )

    skipped: List[str] = []
    for i, (oi, name) in enumerate(zip(obs_idx, obs_names)):
        ax = axes[i // n_cols_eff][i % n_cols_eff]
        ok = _plot_one(
            ax=ax,
            values=x[:, oi],
            weights_per_bench=weights_per_group,
            bench_labels=group_labels,
            n_bins=n_bins,
            log=log,
            normalize=normalize,
            quantiles_for_range=quantiles_for_range,
            name=name,
        )
        if not ok:
            ax.text(0.5, 0.5, f"{name}\n(no valid range)", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            skipped.append(name)

    for j in range(n, n_rows * n_cols_eff):
        axes[j // n_cols_eff][j % n_cols_eff].axis("off")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return skipped


def _infer_observable_names_for_npy(
    x_file: Path, n_features: int, logger
) -> List[str]:
    h5_candidates = sorted(x_file.parent.glob("*.h5"))
    if len(h5_candidates) != 1:
        if len(h5_candidates) > 1:
            logger.warning(
                f"{x_file.name}: found multiple .h5 files in {x_file.parent}; "
                "using generic observable names"
            )
        return [f"x{i}" for i in range(n_features)]

    h5_file = h5_candidates[0]
    try:
        from madminer.analysis import DataAnalyzer

        analyzer = DataAnalyzer(str(h5_file), disable_morphing=True)
        names = list(analyzer.observables)
    except Exception as ex:
        logger.warning(
            f"{x_file.name}: could not load observables from {h5_file.name} "
            f"({type(ex).__name__}: {ex}); using generic observable names"
        )
        return [f"x{i}" for i in range(n_features)]

    if len(names) != n_features:
        logger.warning(
            f"{x_file.name}: feature dimension {n_features} does not match "
            f"{h5_file.name} ({len(names)} observables); using generic names"
        )
        return [f"x{i}" for i in range(n_features)]

    return names


def _format_class_label(value: float) -> str:
    as_float = float(value)
    if as_float.is_integer():
        return f"class {int(as_float)}"
    return f"class {as_float:.3g}"


def _group_weights_for_npy(
    x_file: Path,
    n_events: int,
    benchmarks: Optional[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    suffix = x_file.stem[2:] if x_file.stem.startswith("x_") else x_file.stem
    y_file = x_file.with_name(f"y_{suffix}.npy")

    if y_file.exists():
        y = np.asarray(np.load(y_file)).reshape(-1)
        if y.shape[0] != n_events:
            raise ValueError(
                f"{x_file.name}: {y_file.name} has {y.shape[0]} rows, expected {n_events}"
            )

        finite = np.isfinite(y)
        if not finite.any():
            raise ValueError(f"{x_file.name}: {y_file.name} has no finite labels")

        classes = np.sort(np.unique(y[finite]))
        weights = np.column_stack([(y == cls).astype(float) for cls in classes])

        if benchmarks is not None:
            if len(benchmarks) != len(classes):
                raise ValueError(
                    f"{x_file.name}: got {len(benchmarks)} labels via --benchmarks, "
                    f"but found {len(classes)} classes in {y_file.name}"
                )
            group_labels = benchmarks
        elif len(classes) == 2 and suffix.endswith("ratio"):
            group_labels = ["theta0", "theta1"]
        else:
            group_labels = [_format_class_label(float(cls)) for cls in classes]

        return weights, group_labels

    weights = np.ones((n_events, 1), dtype=float)
    if benchmarks is not None:
        if len(benchmarks) != 1:
            raise ValueError(
                f"{x_file.name}: expected exactly one --benchmarks label when "
                f"{y_file.name} is missing, got {len(benchmarks)}"
            )
        group_labels = benchmarks
    else:
        group_labels = [x_file.stem]

    return weights, group_labels


def plot_process_distributions(
    input_files: List[Path],
    outdir: Path,
    dataset: str,
    level: str,
    observables: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    normalize: bool = False,
    log: bool = False,
    n_bins: int = 50,
    n_cols: int = 3,
    uncertainties: str = "none",  # kept for CLI compatibility; unused here
    quantiles_for_range: Tuple[float, float] = (0.025, 0.975),
    convert_to_ptetaphi: bool = False,
    plot_jet_multiplicity: bool = False,
) -> List[Path]:
    """Plot observable distributions for each input file.

    Supports analyzed ``.h5`` files and augmented ``x_*.npy`` files.
    One composite figure per input file is written to ``outdir/dataset/level/<stem>.png``.
    Observables whose weighted range is degenerate/NaN are skipped.
    """
    logger = LOGGER.getChild(f"{__name__}.plot_process_distributions")
    _ = uncertainties

    target_dir = outdir / dataset / level
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for input_file in input_files:
        suffix = input_file.suffix.lower()

        if suffix == ".h5":
            from madminer.analysis import DataAnalyzer

            logger.info(f"Loading {input_file}")
            analyzer = DataAnalyzer(str(input_file), disable_morphing=True)

            all_obs_names = list(analyzer.observables)

            all_bench_names = list(analyzer.benchmarks)
            bench_names = benchmarks if benchmarks else all_bench_names
            missing_b = [b for b in bench_names if b not in all_bench_names]
            if missing_b:
                raise ValueError(f"Benchmarks not in file {input_file}: {missing_b}")
            bench_idx = [all_bench_names.index(b) for b in bench_names]

            x, w_all = analyzer.weighted_events()
            w = w_all[:, bench_idx]

            if convert_to_ptetaphi:
                x, all_obs_names = _convert_to_ptetaphi(x, all_obs_names)

            obs_names = observables if observables else all_obs_names
            missing = [o for o in obs_names if o not in all_obs_names]
            if missing:
                raise ValueError(f"Observables not in file {input_file}: {missing}")
            obs_idx = [all_obs_names.index(o) for o in obs_names]

            out_path = target_dir / f"{input_file.stem}.png"
            skipped = _plot_observable_grid(
                x=x,
                obs_idx=obs_idx,
                obs_names=obs_names,
                weights_per_group=w,
                group_labels=bench_names,
                out_path=out_path,
                title=input_file.stem,
                n_bins=n_bins,
                n_cols=n_cols,
                log=log,
                normalize=normalize,
                quantiles_for_range=quantiles_for_range,
            )
            if skipped:
                logger.warning(
                    f"{input_file.name}: skipped {len(skipped)} observables: {skipped}"
                )
            logger.info(f"Wrote {out_path}")
            saved.append(out_path)

            if plot_jet_multiplicity:
                njets = _jet_multiplicity(x, all_obs_names)
                if njets is None:
                    logger.warning(
                        f"{input_file.name}: no j<i>_E or j<i>_pt observables found, "
                        "skipping jet multiplicity plot"
                    )
                else:
                    nj_path = target_dir / f"{input_file.stem}_njets.png"
                    _plot_jet_multiplicity_grid(
                        njets=njets,
                        weights_per_group=w,
                        group_labels=bench_names,
                        out_path=nj_path,
                        title=f"{input_file.stem} — jet multiplicity",
                        normalize=normalize,
                        log=log,
                    )
                    logger.info(f"Wrote {nj_path}")
                    saved.append(nj_path)
            continue

        if suffix == ".npy":
            logger.info(f"Loading {input_file}")
            x = np.asarray(np.load(input_file))
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if x.ndim != 2:
                raise ValueError(
                    f"{input_file}: expected a 1D or 2D array, got shape {x.shape}"
                )
            if x.shape[0] == 0:
                raise ValueError(f"{input_file}: empty input array")

            all_obs_names = _infer_observable_names_for_npy(
                x_file=input_file,
                n_features=x.shape[1],
                logger=logger,
            )

            if convert_to_ptetaphi:
                x, all_obs_names = _convert_to_ptetaphi(x, all_obs_names)

            obs_names = observables if observables else all_obs_names
            missing = [o for o in obs_names if o not in all_obs_names]
            if missing:
                raise ValueError(f"Observables not in file {input_file}: {missing}")
            obs_idx = [all_obs_names.index(o) for o in obs_names]

            weights_per_group, group_labels = _group_weights_for_npy(
                x_file=input_file,
                n_events=x.shape[0],
                benchmarks=benchmarks,
            )

            out_path = target_dir / f"{input_file.stem}.png"
            skipped = _plot_observable_grid(
                x=x,
                obs_idx=obs_idx,
                obs_names=obs_names,
                weights_per_group=weights_per_group,
                group_labels=group_labels,
                out_path=out_path,
                title=input_file.stem,
                n_bins=n_bins,
                n_cols=n_cols,
                log=log,
                normalize=normalize,
                quantiles_for_range=quantiles_for_range,
            )
            if skipped:
                logger.warning(
                    f"{input_file.name}: skipped {len(skipped)} observables: {skipped}"
                )
            logger.info(f"Wrote {out_path}")
            saved.append(out_path)

            if plot_jet_multiplicity:
                njets = _jet_multiplicity(x, all_obs_names)
                if njets is None:
                    logger.warning(
                        f"{input_file.name}: no j<i>_E or j<i>_pt observables found, "
                        "skipping jet multiplicity plot"
                    )
                else:
                    nj_path = target_dir / f"{input_file.stem}_njets.png"
                    _plot_jet_multiplicity_grid(
                        njets=njets,
                        weights_per_group=weights_per_group,
                        group_labels=group_labels,
                        out_path=nj_path,
                        title=f"{input_file.stem} — jet multiplicity",
                        normalize=normalize,
                        log=log,
                    )
                    logger.info(f"Wrote {nj_path}")
                    saved.append(nj_path)
            continue

        raise ValueError(
            f"Unsupported input file type for plotting: {input_file} "
            "(expected .h5 or .npy)"
        )

    return saved

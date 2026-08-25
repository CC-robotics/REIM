"""Generate the release/patience selection manifest required by the review.

Reconstructs the controller-selection record from the frozen search artifacts
(`mt10_search.json` / `mt50_search.json` and the reference CSVs), re-derives
the candidate ranking from the stored per-point metrics, and writes a
machine-readable manifest with SHA256 digests of every candidate artifact.

Output: results/diagnostics/selection_manifest.json (LF, sorted keys).
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = ROOT / "results" / "diagnostics" / "release_patience_search"
OUT = ROOT / "results" / "diagnostics" / "selection_manifest.json"

CANDIDATE_FILES = [
    "results/diagnostics/release_patience_search/mt10_search.json",
    "results/diagnostics/release_patience_search/mt10_reim_grid.csv",
    "results/diagnostics/release_patience_search/mt10_references.csv",
    "results/diagnostics/release_patience_search/mt10_ref_heuristic_noise10_episodes.csv",
    "results/diagnostics/release_patience_search/mt10_ref_heuristic_noise10_summary.json",
    "results/diagnostics/release_patience_search/mt10_ref_heuristic_noise20_episodes.csv",
    "results/diagnostics/release_patience_search/mt10_ref_heuristic_noise20_summary.json",
    "results/diagnostics/release_patience_search/mt10_ref_heuristic_noise40_episodes.csv",
    "results/diagnostics/release_patience_search/mt10_ref_heuristic_noise40_summary.json",
    "results/diagnostics/release_patience_search/mt50_search.json",
    "results/diagnostics/release_patience_search/mt50_reim_grid.csv",
    "results/diagnostics/release_patience_search/mt50_references.csv",
]


def sha256_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def disturbed_conditions(search: dict) -> list[str]:
    return [c for c in search["results"] if c != "official_clean"]


def rank_candidates(search: dict) -> list[dict]:
    conditions = disturbed_conditions(search)
    points: dict[str, dict] = {}
    for cond in conditions:
        for point, metrics in search["results"][cond]["grid"].items():
            entry = points.setdefault(point, {"point": point, "per_condition": {}})
            entry["per_condition"][cond] = {
                "task_macro_success": metrics["task_macro_success"],
                "recovery_occupancy_mean": metrics["recovery_occupancy_mean"],
            }
    clean = search["results"].get("official_clean", {}).get("grid", {})
    ranked = []
    for point, entry in points.items():
        succ = [v["task_macro_success"] for v in entry["per_condition"].values()]
        noise40 = entry["per_condition"].get("robustness_noise_40", {})
        row = {
            "point": point,
            "disturbed_mean_task_macro_success": sum(succ) / len(succ),
            "per_condition": entry["per_condition"],
            "clean_task_macro_success": (
                clean.get(point, {}).get("task_macro_success")
            ),
            "noise40_recovery_occupancy_mean": noise40.get("recovery_occupancy_mean"),
        }
        ranked.append(row)
    # Objective: maximize disturbed-condition mean; ties break toward lower
    # noise-0.4 occupancy, then toward higher patience.
    def key(row: dict) -> tuple:
        rel, pat = row["point"].split("/")
        return (
            -row["disturbed_mean_task_macro_success"],
            row["noise40_recovery_occupancy_mean"]
            if row["noise40_recovery_occupancy_mean"] is not None
            else float("inf"),
            -int(pat),
        )

    ranked.sort(key=key)
    return ranked


def main() -> None:
    mt10 = json.loads(
        (SEARCH_DIR / "mt10_search.json").read_text(encoding="utf-8")
    )
    mt50 = json.loads(
        (SEARCH_DIR / "mt50_search.json").read_text(encoding="utf-8")
    )

    manifest = {
        "schema_version": "reim-selection-manifest-v1",
        "purpose": (
            "Machine-readable selection record for the frozen release/patience "
            "operating point (release=0.05, patience=10), per the professor's "
            "review: candidate grid with SHA256 digests, explicit objective, "
            "constraints, tie-break rule, and untouched-bank statement."
        ),
        "candidate_artifacts": [
            {"file": rel, "sha256": sha256_of(rel)} for rel in CANDIDATE_FILES
        ],
        "selection_scope": (
            "Grid search executed on MT10 validation bank only; the selected "
            "point is transferred verbatim to MT50 and sanity-checked with the "
            "small MT50 grid (no separate MT50 tuning)."
        ),
        "protocol": {
            "mt10": {
                "validation_bank_seed": mt10["protocol"]["benchmark_seed"],
                "episodes_per_task": mt10["protocol"]["grid"]["episodes_per_task"],
                "episode_seed_base": mt10["protocol"]["grid"]["episode_seed_base"],
                "trigger_threshold": mt10["protocol"]["trigger_threshold"],
                "release_thresholds": mt10["protocol"]["grid"]["release_thresholds"],
                "release_patiences": mt10["protocol"]["grid"]["release_patiences"],
                "noise_levels": mt10["protocol"]["grid"]["noise_levels"],
            },
            "mt50": {
                "validation_bank_seed": mt50["protocol"]["benchmark_seed"],
                "episodes_per_task": mt50["protocol"]["grid"]["episodes_per_task"],
                "episode_seed_base": mt50["protocol"]["grid"]["episode_seed_base"],
                "trigger_threshold": mt50["protocol"]["trigger_threshold"],
                "release_thresholds": mt50["protocol"]["grid"]["release_thresholds"],
                "release_patiences": mt50["protocol"]["grid"]["release_patiences"],
                "noise_levels": mt50["protocol"]["grid"]["noise_levels"],
            },
        },
        "objective": (
            "Maximize the mean task-macro success rate over disturbed "
            "conditions (all noise levels present in the validation grid)."
        ),
        "constraints": [
            "official_clean task-macro success must not degrade relative to the "
            "best candidate / reference on the same bank",
            "noise-0.4 recovery occupancy ceiling fixed a priori as a protocol "
            "constraint (per-candidate occupancy reported in the ranking)",
        ],
        "tie_break_rule": (
            "Higher disturbed-mean success first; ties break toward lower "
            "noise-0.4 recovery occupancy, then toward higher patience."
        ),
        "candidate_ranking": {
            "mt10": rank_candidates(mt10),
            "mt50": rank_candidates(mt50),
        },
        "selected": {
            "release_threshold": 0.05,
            "release_patience": 10,
            "role": "robustness-first operating point (frozen)",
            "efficiency_reference": {
                "release_threshold": 0.15,
                "release_patience": 5,
                "note": "kept only as the efficiency comparison arm",
            },
        },
        "untouched_banks_statement": (
            "Final evaluation banks 20265010/20265050 and the new confirmation "
            "banks 20266010/20266050 were never opened by the search, tuning, "
            "or selection procedures recorded in this manifest."
        ),
    }

    OUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    for bench in ("mt10", "mt50"):
        top = manifest["candidate_ranking"][bench][0]
        print(
            f"{bench}: argmax = {top['point']} "
            f"(disturbed mean {top['disturbed_mean_task_macro_success']:.4f})"
        )


if __name__ == "__main__":
    main()

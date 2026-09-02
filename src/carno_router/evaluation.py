from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from .calibration import PlattCalibrator
from .bundle import build_model_bundle
from .dataset import BenchmarkExample, SplitBundle, file_sha256, group_time_split, load_examples
from .model import SparseLogisticRegression, TfidfVectorizer, sigmoid
from .pricing import PricingSnapshot


def _rounded(value: float | None, digits: int = 8) -> float | None:
    return None if value is None else round(value, digits)


def _decimal_string(value: Decimal) -> str:
    rendered = format(value.quantize(Decimal("0.00000001")), "f")
    return rendered.rstrip("0").rstrip(".") or "0"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cost_for(
    row: BenchmarkExample, selected: str, pricing: PricingSnapshot
) -> Decimal:
    if selected == "cheap":
        return pricing.estimate(
            "cheap", row.cheap_input_tokens, row.cheap_output_tokens
        )
    if selected == "strong":
        return pricing.estimate(
            "strong", row.strong_input_tokens, row.strong_output_tokens
        )
    raise ValueError(f"unknown selection: {selected}")


def _latency_for(row: BenchmarkExample, selected: str) -> float | None:
    return row.cheap_latency_ms if selected == "cheap" else row.strong_latency_ms


def _quality_for(row: BenchmarkExample, selected: str) -> float:
    return row.cheap_quality if selected == "cheap" else row.strong_quality


def _selection_metrics(
    rows: tuple[BenchmarkExample, ...],
    selections: dict[str, str],
    pricing: PricingSnapshot,
) -> dict[str, object]:
    if not rows:
        raise ValueError("metrics require rows")
    qualities: list[float] = []
    costs: list[Decimal] = []
    latencies: list[float] = []
    latency_complete = True
    correct = 0
    strong_needed = 0
    strong_needed_selected = 0
    weak_opportunities = 0
    weak_opportunities_selected = 0
    strong_calls = 0
    for row in rows:
        selected = selections[row.example_id]
        qualities.append(_quality_for(row, selected))
        costs.append(_cost_for(row, selected, pricing))
        if selected == "strong":
            strong_calls += 1
        target = "strong" if row.strong_is_better else "cheap"
        correct += int(selected == target)
        if row.strong_is_better:
            strong_needed += 1
            strong_needed_selected += int(selected == "strong")
        else:
            weak_opportunities += 1
            weak_opportunities_selected += int(selected == "cheap")
        latency = _latency_for(row, selected)
        if latency is None:
            latency_complete = False
        else:
            latencies.append(latency)
    total_cost = sum(costs, Decimal(0))
    mean_cost = total_cost / Decimal(len(rows))
    payload: dict[str, object] = {
        "rows": len(rows),
        "mean_quality": _rounded(sum(qualities) / len(qualities)),
        "total_cost": _decimal_string(total_cost),
        "mean_cost": _decimal_string(mean_cost),
        "strong_call_rate": _rounded(strong_calls / len(rows)),
        "preference_accuracy": _rounded(correct / len(rows)),
        "strong_needed_recall": _rounded(
            strong_needed_selected / strong_needed if strong_needed else 1.0
        ),
        "weak_opportunity_specificity": _rounded(
            weak_opportunities_selected / weak_opportunities
            if weak_opportunities
            else 1.0
        ),
        "latency_complete": latency_complete,
    }
    if latency_complete:
        payload["latency_ms"] = {
            "mean": _rounded(sum(latencies) / len(latencies)),
            "p50": _rounded(_percentile(latencies, 0.50)),
            "p95": _rounded(_percentile(latencies, 0.95)),
            "p99": _rounded(_percentile(latencies, 0.99)),
        }
    else:
        payload["latency_ms"] = None
    return payload


def _threshold_selections(
    rows: tuple[BenchmarkExample, ...],
    probabilities: list[float],
    threshold: float,
) -> dict[str, str]:
    return {
        row.example_id: "strong" if probability >= threshold else "cheap"
        for row, probability in zip(rows, probabilities)
    }


def _threshold_curve(
    rows: tuple[BenchmarkExample, ...],
    probabilities: list[float],
    pricing: PricingSnapshot,
) -> list[dict[str, object]]:
    curve = []
    for step in range(102):
        threshold = step / 100.0 if step <= 100 else 1.000000001
        metrics = _selection_metrics(
            rows, _threshold_selections(rows, probabilities, threshold), pricing
        )
        curve.append({"threshold": _rounded(threshold), **metrics})
    return curve


def _choose_quality_constrained_threshold(
    curve: list[dict[str, object]],
    always_strong_quality: float,
    max_quality_drop: float,
) -> dict[str, object]:
    minimum_quality = always_strong_quality - max_quality_drop
    eligible = [
        row for row in curve if float(row["mean_quality"]) >= minimum_quality - 1e-12
    ]
    if not eligible:
        raise RuntimeError("always-strong threshold is unexpectedly absent")
    return min(
        eligible,
        key=lambda row: (
            Decimal(str(row["mean_cost"])),
            -float(row["mean_quality"]),
            -float(row["threshold"]),
        ),
    )


def _risk_controlled_threshold_curve(
    rows: tuple[BenchmarkExample, ...],
    probabilities: list[float],
    pricing: PricingSnapshot,
    *,
    max_quality_drop: float,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Attach a calibration-only one-sided bootstrap risk bound to each policy."""

    if replicates < 200:
        raise ValueError("risk-control bootstrap requires at least 200 replicates")
    base_curve = _threshold_curve(rows, probabilities, pricing)
    group_rows: dict[str, list[BenchmarkExample]] = {}
    strata: dict[str, list[str]] = {}
    for row in rows:
        group_rows.setdefault(row.group_id, []).append(row)
    for group_id, members in group_rows.items():
        pairs = sorted({row.model_pair or "unspecified" for row in members})
        majority_label = int(
            sum(row.strong_is_better for row in members) * 2 >= len(members)
        )
        strata.setdefault(f"{'+'.join(pairs)}|label={majority_label}", []).append(group_id)

    rng = random.Random(seed)
    bootstrap_group_samples: list[list[str]] = []
    for _ in range(replicates):
        sampled_groups = []
        for stratum in sorted(strata):
            groups = sorted(strata[stratum])
            sampled_groups.extend(rng.choice(groups) for _slot in groups)
        bootstrap_group_samples.append(sampled_groups)

    probability_by_id = {
        row.example_id: probability for row, probability in zip(rows, probabilities)
    }
    preliminary_curve = []
    replicate_matrix: list[list[float]] = []
    for point in base_curve:
        threshold = float(point["threshold"])
        group_quality_drops: dict[str, tuple[float, int]] = {}
        for group_id, members in group_rows.items():
            routed_quality = 0.0
            always_strong_quality = 0.0
            for row in members:
                selection = (
                    "strong"
                    if probability_by_id[row.example_id] >= threshold
                    else "cheap"
                )
                routed_quality += _quality_for(row, selection)
                always_strong_quality += row.strong_quality
            group_quality_drops[group_id] = (
                always_strong_quality - routed_quality,
                len(members),
            )
        replicate_drops = []
        for sampled_groups in bootstrap_group_samples:
            drop_sum = sum(group_quality_drops[group_id][0] for group_id in sampled_groups)
            row_count = sum(group_quality_drops[group_id][1] for group_id in sampled_groups)
            replicate_drops.append(drop_sum / row_count)
        quality_drop = (
            float(base_curve[0]["mean_quality"]) - float(point["mean_quality"])
        )
        upper_bound = _percentile(replicate_drops, 0.95)
        preliminary_curve.append(
            {
                **point,
                "quality_drop_vs_always_strong": _rounded(quality_drop),
                "quality_drop_upper_pointwise_one_sided_95": _rounded(upper_bound),
            }
        )
        replicate_matrix.append(replicate_drops)
    max_deviations = [
        max(
            replicate_matrix[threshold_index][replicate_index]
            - float(preliminary_curve[threshold_index]["quality_drop_vs_always_strong"])
            for threshold_index in range(len(preliminary_curve))
        )
        for replicate_index in range(replicates)
    ]
    simultaneous_margin = _percentile(max_deviations, 0.95)
    controlled_curve = []
    for point in preliminary_curve:
        exact_always_strong = float(point["strong_call_rate"]) == 1.0
        simultaneous_upper = (
            0.0
            if exact_always_strong
            else float(point["quality_drop_vs_always_strong"]) + simultaneous_margin
        )
        controlled_curve.append(
            {
                **point,
                "quality_drop_upper_simultaneous_one_sided_95": _rounded(
                    simultaneous_upper
                ),
                "risk_constraint_pass": simultaneous_upper
                <= max_quality_drop + 1e-12,
            }
        )
    eligible = [row for row in controlled_curve if row["risk_constraint_pass"]]
    if not eligible:
        raise RuntimeError("always-strong policy must satisfy the risk constraint")
    selected = min(
        eligible,
        key=lambda row: (
            Decimal(str(row["mean_cost"])),
            float(row["quality_drop_upper_simultaneous_one_sided_95"]),
            -float(row["mean_quality"]),
            -float(row["threshold"]),
        ),
    )
    receipt = {
        "schema": "CarnoCalibrationRiskControlReceiptV1",
        "selection_data": "calibration_only",
        "test_data_used": False,
        "method": "simultaneous_one_sided_95_percent_max_deviation_prompt_group_bootstrap_stratified_by_model_pair_and_label",
        "replicates": replicates,
        "seed": seed,
        "quality_drop_budget": max_quality_drop,
        "candidate_thresholds": len(controlled_curve),
        "eligible_thresholds": len(eligible),
        "simultaneous_max_deviation_margin": _rounded(simultaneous_margin),
        "multiplicity_control": "familywise band across all candidate thresholds",
        "selected": selected,
    }
    return controlled_curve, receipt


def _choose_safety_threshold(curve: list[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in curve if float(row["strong_needed_recall"]) >= 0.95]
    return max(eligible, key=lambda row: float(row["threshold"])) if eligible else curve[0]


def _choose_balanced_threshold(curve: list[dict[str, object]]) -> dict[str, object]:
    return max(
        curve,
        key=lambda row: (
            float(row["preference_accuracy"]),
            -float(row["strong_call_rate"]),
            float(row["threshold"]),
        ),
    )


def _random_selections(
    rows: tuple[BenchmarkExample, ...], rate: float, seed: int
) -> dict[str, str]:
    selections = {}
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{row.example_id}".encode("utf-8")).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64)
        selections[row.example_id] = "strong" if draw < rate else "cheap"
    return selections


@dataclass(frozen=True)
class PromptModelSpec:
    name: str
    max_features: int
    word_ngram_max: int
    char_ngram_min: int | None
    char_ngram_max: int | None
    l2: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_features": self.max_features,
            "word_ngram_max": self.word_ngram_max,
            "char_ngram_min": self.char_ngram_min,
            "char_ngram_max": self.char_ngram_max,
            "l2": self.l2,
        }


MODEL_CANDIDATES = (
    PromptModelSpec("word_1_2__2048__l2_0_02", 2048, 2, None, None, 0.02),
    PromptModelSpec("hybrid__2048__l2_0_02", 2048, 2, 3, 5, 0.02),
    PromptModelSpec("hybrid__4096__l2_0_02", 4096, 2, 3, 5, 0.02),
    PromptModelSpec("hybrid__4096__l2_0_08", 4096, 2, 3, 5, 0.08),
)


def _fit_prompt_model(
    rows: tuple[BenchmarkExample, ...],
    specification: PromptModelSpec,
) -> tuple[TfidfVectorizer, SparseLogisticRegression]:
    vectorizer = TfidfVectorizer(
        max_features=specification.max_features,
        word_ngram_max=specification.word_ngram_max,
        char_ngram_min=specification.char_ngram_min,
        char_ngram_max=specification.char_ngram_max,
    ).fit([row.prompt for row in rows])
    vectors = vectorizer.transform([row.prompt for row in rows])
    model = SparseLogisticRegression(l2=specification.l2).fit(
        vectors,
        [row.strong_is_better for row in rows],
        len(vectorizer.vocabulary or {}),
    )
    return vectorizer, model


def _group_stratified_oof_logits(
    rows: tuple[BenchmarkExample, ...],
    fold_count: int,
    seed: int,
    specification: PromptModelSpec,
) -> tuple[list[float], dict[str, object]]:
    if fold_count < 2:
        raise ValueError("out-of-fold calibration requires at least two folds")
    group_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group_indices.setdefault(row.group_id, []).append(index)
    if len(group_indices) < 2 * fold_count:
        raise ValueError("too few prompt groups for requested calibration folds")

    strata: dict[str, list[str]] = {}
    for group_id, indices in group_indices.items():
        labels = [rows[index].strong_is_better for index in indices]
        pairs = sorted({rows[index].model_pair or "unspecified" for index in indices})
        majority_label = int(sum(labels) * 2 >= len(labels))
        stratum = f"{'+'.join(pairs)}|label={majority_label}"
        strata.setdefault(stratum, []).append(group_id)

    group_fold: dict[str, int] = {}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda group_id: hashlib.sha256(
                f"{seed}:{stratum}:{group_id}".encode("utf-8")
            ).hexdigest(),
        )
        offset = int(
            hashlib.sha256(f"{seed}:{stratum}".encode("utf-8")).hexdigest()[:8],
            16,
        ) % fold_count
        for position, group_id in enumerate(ordered):
            group_fold[group_id] = (offset + position) % fold_count

    oof_logits: list[float | None] = [None] * len(rows)
    fold_receipts = []
    for fold in range(fold_count):
        held_out_indices = [
            index for index, row in enumerate(rows) if group_fold[row.group_id] == fold
        ]
        fit_indices = [
            index for index, row in enumerate(rows) if group_fold[row.group_id] != fold
        ]
        if not held_out_indices:
            raise ValueError(f"calibration fold {fold} contains no rows")
        fit_rows = tuple(rows[index] for index in fit_indices)
        if set(row.strong_is_better for row in fit_rows) != {0, 1}:
            raise ValueError(f"calibration fold {fold} training complement lacks both classes")
        vectorizer, model = _fit_prompt_model(fit_rows, specification)
        fold_logits = model.decision_function(
            vectorizer.transform([rows[index].prompt for index in held_out_indices])
        )
        for index, logit in zip(held_out_indices, fold_logits):
            oof_logits[index] = logit
        fit_groups = {rows[index].group_id for index in fit_indices}
        held_out_groups = {rows[index].group_id for index in held_out_indices}
        if fit_groups & held_out_groups:
            raise RuntimeError("prompt group crossed an out-of-fold calibration boundary")
        fold_receipts.append(
            {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "held_out_rows": len(held_out_indices),
                "fit_groups": len(fit_groups),
                "held_out_groups": len(held_out_groups),
                "group_overlap": 0,
            }
        )
    if any(logit is None for logit in oof_logits):
        raise RuntimeError("out-of-fold prediction coverage is incomplete")
    return [float(logit) for logit in oof_logits], {
        "schema": "CarnoGroupStratifiedOOFReceiptV1",
        "fold_count": fold_count,
        "seed": seed,
        "strategy": "model_pair_and_majority_label_stratified_prompt_groups",
        "row_count": len(rows),
        "group_count": len(group_indices),
        "prediction_coverage": 1.0,
        "group_overlap": 0,
        "model_specification": specification.to_dict(),
        "folds": fold_receipts,
    }


def _select_prompt_model(
    rows: tuple[BenchmarkExample, ...], fold_count: int, seed: int
) -> tuple[PromptModelSpec, list[float], dict[str, object], dict[str, object]]:
    labels = [row.strong_is_better for row in rows]
    candidates = []
    cached: dict[str, tuple[list[float], dict[str, object]]] = {}
    for specification in MODEL_CANDIDATES:
        logits, receipt = _group_stratified_oof_logits(
            rows, fold_count=fold_count, seed=seed, specification=specification
        )
        probabilities = [sigmoid(logit) for logit in logits]
        metrics = _calibration_metrics(labels, probabilities)
        candidates.append(
            {
                "specification": specification.to_dict(),
                "oof_metrics_before_calibration": {
                    key: metrics[key]
                    for key in ("roc_auc", "pr_auc", "brier", "log_loss")
                },
            }
        )
        cached[specification.name] = (logits, receipt)
    selected_row = max(
        candidates,
        key=lambda row: (
            float(row["oof_metrics_before_calibration"]["roc_auc"] or -1.0),
            -float(row["oof_metrics_before_calibration"]["brier"] or 1.0),
            -float(row["oof_metrics_before_calibration"]["log_loss"] or 99.0),
            str(row["specification"]["name"]),
        ),
    )
    selected_name = str(selected_row["specification"]["name"])
    selected_specification = next(
        specification
        for specification in MODEL_CANDIDATES
        if specification.name == selected_name
    )
    logits, oof_receipt = cached[selected_name]
    selection_receipt = {
        "schema": "CarnoTrainOnlyModelSelectionReceiptV1",
        "selection_data": "group_stratified_out_of_fold_train_predictions_only",
        "test_data_used": False,
        "calibration_data_used": False,
        "objective": "maximize_roc_auc_then_minimize_brier_then_log_loss",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected_row,
    }
    return selected_specification, logits, oof_receipt, selection_receipt


def _prompt_length_vectors(
    fit_rows: tuple[BenchmarkExample, ...],
    target_rows: tuple[BenchmarkExample, ...],
) -> tuple[list[dict[int, float]], dict[str, float]]:
    fit_components = [
        (math.log1p(len(row.prompt.split())), math.log1p(len(row.prompt)))
        for row in fit_rows
    ]
    means = [sum(values[index] for values in fit_components) / len(fit_components) for index in range(2)]
    scales = []
    for index in range(2):
        variance = sum(
            (values[index] - means[index]) ** 2 for values in fit_components
        ) / len(fit_components)
        scales.append(max(math.sqrt(variance), 1e-12))
    vectors = []
    for row in target_rows:
        values = (math.log1p(len(row.prompt.split())), math.log1p(len(row.prompt)))
        vectors.append(
            {index: (values[index] - means[index]) / scales[index] for index in range(2)}
        )
    return vectors, {
        "word_log_mean": _rounded(means[0]),
        "word_log_scale": _rounded(scales[0]),
        "character_log_mean": _rounded(means[1]),
        "character_log_scale": _rounded(scales[1]),
    }


def _normalized_cost_sensitivity(strong_call_rate: float) -> dict[str, object]:
    sensitivity = {}
    for ratio in (2, 5, 10, 20):
        normalized_cost = 1.0 + (ratio - 1.0) * strong_call_rate
        sensitivity[str(ratio)] = {
            "normalized_cost": _rounded(normalized_cost),
            "savings_vs_always_strong": _rounded((ratio - normalized_cost) / ratio),
        }
    return sensitivity


def _oracle_selections(rows: tuple[BenchmarkExample, ...]) -> dict[str, str]:
    return {
        row.example_id: "strong" if row.strong_quality > row.cheap_quality else "cheap"
        for row in rows
    }


def _roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = [probabilities[index] for index, label in enumerate(labels) if label]
    negatives = [probabilities[index] for index, label in enumerate(labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _average_precision(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(
        zip(probabilities, labels, range(len(labels))),
        key=lambda row: (-row[0], row[2]),
    )
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label, _) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _calibration_metrics(labels: list[int], probabilities: list[float]) -> dict[str, object]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("calibration metrics require aligned rows")
    epsilon = 1e-12
    brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)
    log_loss = -sum(
        label * math.log(max(probability, epsilon))
        + (1 - label) * math.log(max(1.0 - probability, epsilon))
        for label, probability in zip(labels, probabilities)
    ) / len(labels)
    ece = 0.0
    bins = []
    for index in range(5):
        low = index / 5.0
        high = (index + 1) / 5.0
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities)
            if low <= probability < high or (index == 4 and probability == 1.0)
        ]
        if members:
            observed = sum(label for label, _ in members) / len(members)
            predicted = sum(probability for _, probability in members) / len(members)
            ece += len(members) / len(labels) * abs(observed - predicted)
        else:
            observed = predicted = None
        bins.append(
            {
                "low": low,
                "high": high,
                "count": len(members),
                "observed": _rounded(observed),
                "predicted": _rounded(predicted),
            }
        )
    return {
        "roc_auc": _rounded(_roc_auc(labels, probabilities)),
        "pr_auc": _rounded(_average_precision(labels, probabilities)),
        "brier": _rounded(brier),
        "log_loss": _rounded(log_loss),
        "ece_5": _rounded(ece),
        "bins": bins,
    }


def _distribution_summary(rows: tuple[BenchmarkExample, ...]) -> dict[str, object]:
    word_lengths = sorted(len(row.prompt.split()) for row in rows)
    pair_counts: dict[str, int] = {}
    for row in rows:
        pair = row.model_pair or "unspecified"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return {
        "rows": len(rows),
        "prompt_groups": len({row.group_id for row in rows}),
        "strong_better_prevalence": _rounded(
            sum(row.strong_is_better for row in rows) / len(rows)
        ),
        "prompt_word_count": {
            "p25": word_lengths[len(word_lengths) // 4],
            "p50": word_lengths[len(word_lengths) // 2],
            "p75": word_lengths[(3 * len(word_lengths)) // 4],
            "p90": word_lengths[(9 * len(word_lengths)) // 10],
            "max": word_lengths[-1],
        },
        "model_pair_counts": dict(sorted(pair_counts.items())),
    }


def _categorical_total_variation(
    left: tuple[BenchmarkExample, ...], right: tuple[BenchmarkExample, ...]
) -> float:
    categories = {
        row.model_pair or "unspecified" for row in left + right
    }
    return 0.5 * sum(
        abs(
            sum((row.model_pair or "unspecified") == category for row in left)
            / len(left)
            - sum((row.model_pair or "unspecified") == category for row in right)
            / len(right)
        )
        for category in categories
    )


def _shift_diagnostics(split: SplitBundle) -> dict[str, object]:
    train = _distribution_summary(split.train)
    calibration = _distribution_summary(split.calibration)
    test = _distribution_summary(split.test)
    pair_tv_train_test = _categorical_total_variation(split.train, split.test)
    prevalence_delta = abs(
        float(train["strong_better_prevalence"])
        - float(test["strong_better_prevalence"])
    )
    warnings = []
    if pair_tv_train_test >= 0.20:
        warnings.append("model_pair_mix_shift_train_to_test")
    if prevalence_delta >= 0.10:
        warnings.append("label_prevalence_shift_train_to_test")
    if int(test["rows"]) < 200:
        warnings.append("small_test_cohort")
    return {
        "schema": "CarnoDistributionShiftReceiptV1",
        "train": train,
        "calibration": calibration,
        "test": test,
        "train_to_test": {
            "model_pair_total_variation": _rounded(pair_tv_train_test),
            "strong_better_prevalence_absolute_delta": _rounded(prevalence_delta),
        },
        "warning_thresholds": {
            "model_pair_total_variation": 0.20,
            "prevalence_absolute_delta": 0.10,
            "minimum_test_rows": 200,
        },
        "warnings": warnings,
    }


def _slice_diagnostics(
    train_rows: tuple[BenchmarkExample, ...],
    test_rows: tuple[BenchmarkExample, ...],
    probabilities: list[float],
    selections: dict[str, str],
    pricing: PricingSnapshot,
) -> list[dict[str, object]]:
    train_lengths = sorted(len(row.prompt.split()) for row in train_rows)
    q25 = train_lengths[len(train_lengths) // 4]
    q75 = train_lengths[(3 * len(train_lengths)) // 4]
    ordered_test = sorted(test_rows, key=lambda row: (row.timestamp, row.group_id))
    midpoint_time = ordered_test[len(ordered_test) // 2].timestamp
    probability_by_id = {
        row.example_id: probability for row, probability in zip(test_rows, probabilities)
    }
    slices: dict[tuple[str, str], list[BenchmarkExample]] = {}
    for row in test_rows:
        word_count = len(row.prompt.split())
        length_slice = "short" if word_count <= q25 else "long" if word_count >= q75 else "medium"
        entries = (
            ("prompt_length", length_slice),
            ("temporal_half", "early" if row.timestamp < midpoint_time else "late"),
            ("model_pair", row.model_pair or "unspecified"),
        )
        for key in entries:
            slices.setdefault(key, []).append(row)
    diagnostics = []
    for (dimension, value), members in sorted(slices.items()):
        member_tuple = tuple(members)
        labels = [row.strong_is_better for row in member_tuple]
        member_probabilities = [probability_by_id[row.example_id] for row in member_tuple]
        metric = _selection_metrics(
            member_tuple,
            {row.example_id: selections[row.example_id] for row in member_tuple},
            pricing,
        )
        calibration = _calibration_metrics(labels, member_probabilities)
        diagnostics.append(
            {
                "dimension": dimension,
                "value": value,
                "rows": len(member_tuple),
                "interpretation": "directional" if len(member_tuple) < 30 else "supported",
                "strong_better_prevalence": _rounded(sum(labels) / len(labels)),
                "roc_auc": calibration["roc_auc"],
                "brier": calibration["brier"],
                "preference_accuracy": metric["preference_accuracy"],
                "strong_call_rate": metric["strong_call_rate"],
                "strong_needed_recall": metric["strong_needed_recall"],
            }
        )
    return diagnostics


def _error_analysis(
    rows: tuple[BenchmarkExample, ...],
    probabilities: list[float],
    selections: dict[str, str],
) -> dict[str, object]:
    errors = {
        "missed_strong_need": 0,
        "unnecessary_strong_call": 0,
        "high_confidence_error": 0,
    }
    by_pair: dict[str, dict[str, int]] = {}
    for row, probability in zip(rows, probabilities):
        selected = selections[row.example_id]
        expected = "strong" if row.strong_is_better else "cheap"
        pair = row.model_pair or "unspecified"
        pair_row = by_pair.setdefault(pair, {"rows": 0, "errors": 0})
        pair_row["rows"] += 1
        if selected != expected:
            pair_row["errors"] += 1
            if selected == "cheap":
                errors["missed_strong_need"] += 1
            else:
                errors["unnecessary_strong_call"] += 1
            if probability <= 0.20 or probability >= 0.80:
                errors["high_confidence_error"] += 1
    return {
        "schema": "CarnoErrorAnalysisReceiptV1",
        **errors,
        "model_pair": {
            pair: {
                **counts,
                "error_rate": _rounded(counts["errors"] / counts["rows"]),
            }
            for pair, counts in sorted(by_pair.items())
        },
        "raw_prompts_in_analysis": False,
    }


def _augment_relative_metrics(
    metrics: dict[str, object],
    always_strong: dict[str, object],
    oracle: dict[str, object],
) -> dict[str, object]:
    strong_cost = Decimal(str(always_strong["total_cost"]))
    current_cost = Decimal(str(metrics["total_cost"]))
    savings = (
        float((strong_cost - current_cost) / strong_cost) if strong_cost else 0.0
    )
    strong_call_rate = float(metrics["strong_call_rate"])
    return {
        **metrics,
        "quality_delta_vs_always_strong": _rounded(
            float(metrics["mean_quality"]) - float(always_strong["mean_quality"])
        ),
        "modeled_cost_savings_vs_always_strong": _rounded(savings),
        "quality_regret_vs_oracle": _rounded(
            float(oracle["mean_quality"]) - float(metrics["mean_quality"])
        ),
        "normalized_cost_sensitivity": _normalized_cost_sensitivity(
            strong_call_rate
        ),
    }


def _bootstrap_intervals(
    rows: tuple[BenchmarkExample, ...],
    routed: dict[str, str],
    pricing: PricingSnapshot,
    replicates: int,
    seed: int,
    comparators: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    if replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    strata: dict[str, list[str]] = {}
    group_rows: dict[str, list[BenchmarkExample]] = {}
    for row in rows:
        group_rows.setdefault(row.group_id, []).append(row)
    for group_id, members in group_rows.items():
        pairs = sorted({row.model_pair or "unspecified" for row in members})
        stratum = "+".join(pairs)
        strata.setdefault(stratum, []).append(group_id)
    rng = random.Random(seed)
    quality_deltas: list[float] = []
    savings_values: list[float] = []
    strong_rates: list[float] = []
    preference_accuracies: list[float] = []
    strong_needed_recalls: list[float] = []
    comparator_accuracy_deltas: dict[str, list[float]] = {
        name: [] for name in (comparators or {})
    }
    for _ in range(replicates):
        sampled: list[BenchmarkExample] = []
        routed_sample: dict[str, str] = {}
        comparator_samples: dict[str, dict[str, str]] = {
            name: {} for name in (comparators or {})
        }
        counter = 0
        for stratum in sorted(strata):
            groups = sorted(strata[stratum])
            for _slot in groups:
                selected_group = rng.choice(groups)
                for row in group_rows[selected_group]:
                    synthetic_id = f"bootstrap-{counter}-{row.example_id}"
                    counter += 1
                    clone = BenchmarkExample(
                        example_id=synthetic_id,
                        group_id=f"bootstrap-{counter}-{row.group_id}",
                        timestamp=row.timestamp,
                        prompt=row.prompt,
                        cheap_quality=row.cheap_quality,
                        strong_quality=row.strong_quality,
                        cheap_input_tokens=row.cheap_input_tokens,
                        cheap_output_tokens=row.cheap_output_tokens,
                        strong_input_tokens=row.strong_input_tokens,
                        strong_output_tokens=row.strong_output_tokens,
                        cheap_latency_ms=row.cheap_latency_ms,
                        strong_latency_ms=row.strong_latency_ms,
                        model_pair=row.model_pair,
                    )
                    sampled.append(clone)
                    routed_sample[synthetic_id] = routed[row.example_id]
                    for name, selections in (comparators or {}).items():
                        comparator_samples[name][synthetic_id] = selections[row.example_id]
        sample_tuple = tuple(sampled)
        routed_metrics = _selection_metrics(sample_tuple, routed_sample, pricing)
        strong_metrics = _selection_metrics(
            sample_tuple,
            {row.example_id: "strong" for row in sample_tuple},
            pricing,
        )
        quality_deltas.append(
            float(routed_metrics["mean_quality"])
            - float(strong_metrics["mean_quality"])
        )
        strong_cost = Decimal(str(strong_metrics["total_cost"]))
        routed_cost = Decimal(str(routed_metrics["total_cost"]))
        savings_values.append(
            float((strong_cost - routed_cost) / strong_cost) if strong_cost else 0.0
        )
        strong_rates.append(float(routed_metrics["strong_call_rate"]))
        preference_accuracies.append(float(routed_metrics["preference_accuracy"]))
        strong_needed_recalls.append(float(routed_metrics["strong_needed_recall"]))
        for name, selections in comparator_samples.items():
            comparator_metrics = _selection_metrics(sample_tuple, selections, pricing)
            comparator_accuracy_deltas[name].append(
                float(routed_metrics["preference_accuracy"])
                - float(comparator_metrics["preference_accuracy"])
            )

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "lower_95": round(_percentile(values, 0.025), 8),
            "median": round(_percentile(values, 0.5), 8),
            "upper_95": round(_percentile(values, 0.975), 8),
        }

    return {
        "method": "paired_prompt_group_bootstrap_stratified_by_model_pair",
        "replicates": replicates,
        "seed": seed,
        "quality_delta_vs_always_strong": interval(quality_deltas),
        "modeled_cost_savings_vs_always_strong": interval(savings_values),
        "strong_call_rate": interval(strong_rates),
        "preference_accuracy": interval(preference_accuracies),
        "strong_needed_recall": interval(strong_needed_recalls),
        "preference_accuracy_delta_vs": {
            name: interval(values)
            for name, values in sorted(comparator_accuracy_deltas.items())
        },
    }


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 1729
    bootstrap_replicates: int = 500
    max_quality_drop: float = 0.02
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    selection_bootstrap_replicates: int = 1000

    def __post_init__(self) -> None:
        if not 0 <= self.max_quality_drop <= 1:
            raise ValueError("max_quality_drop must be in [0,1]")
        if self.selection_bootstrap_replicates < 200:
            raise ValueError("selection bootstrap requires at least 200 replicates")


def run_benchmark(
    data_path: str,
    pricing_path: str,
    config: BenchmarkConfig,
) -> dict[str, object]:
    rows = load_examples(data_path)
    split: SplitBundle = group_time_split(
        rows, config.train_fraction, config.calibration_fraction
    )
    pricing = PricingSnapshot.load(pricing_path)
    if not {"cheap", "strong"}.issubset(pricing.models):
        raise ValueError("benchmark pricing must define cheap and strong")

    selected_specification, oof_logits, oof_receipt, model_selection_receipt = (
        _select_prompt_model(split.train, fold_count=5, seed=config.seed)
    )
    calibrator = PlattCalibrator().fit(
        oof_logits,
        [row.strong_is_better for row in split.train],
    )
    vectorizer, model = _fit_prompt_model(split.train, selected_specification)
    calibration_logits = model.decision_function(
        vectorizer.transform([row.prompt for row in split.calibration])
    )
    calibration_probabilities = calibrator.predict(calibration_logits)
    test_logits = model.decision_function(
        vectorizer.transform([row.prompt for row in split.test])
    )
    test_probabilities = calibrator.predict(test_logits)

    validation_curve = _threshold_curve(
        split.calibration, calibration_probabilities, pricing
    )
    validation_always_strong = _selection_metrics(
        split.calibration,
        {row.example_id: "strong" for row in split.calibration},
        pricing,
    )
    point_estimate_choice = _choose_quality_constrained_threshold(
        validation_curve,
        float(validation_always_strong["mean_quality"]),
        config.max_quality_drop,
    )
    risk_control_curve, risk_control_receipt = _risk_controlled_threshold_curve(
        split.calibration,
        calibration_probabilities,
        pricing,
        max_quality_drop=config.max_quality_drop,
        replicates=config.selection_bootstrap_replicates,
        seed=config.seed + 10_000,
    )
    primary_choice = dict(risk_control_receipt["selected"])
    safety_choice = _choose_safety_threshold(validation_curve)
    balanced_choice = _choose_balanced_threshold(validation_curve)
    selected_threshold = float(primary_choice["threshold"])
    routed_selections = _threshold_selections(
        split.test, test_probabilities, selected_threshold
    )
    point_estimate_selections = _threshold_selections(
        split.test, test_probabilities, float(point_estimate_choice["threshold"])
    )
    safety_selections = _threshold_selections(
        split.test, test_probabilities, float(safety_choice["threshold"])
    )
    balanced_selections = _threshold_selections(
        split.test, test_probabilities, float(balanced_choice["threshold"])
    )

    length_train_vectors, length_scaling = _prompt_length_vectors(
        split.train, split.train
    )
    length_model = SparseLogisticRegression().fit(
        length_train_vectors,
        [row.strong_is_better for row in split.train],
        2,
    )
    length_calibration_vectors, _ = _prompt_length_vectors(
        split.train, split.calibration
    )
    length_validation_probabilities = [
        sigmoid(logit)
        for logit in length_model.decision_function(length_calibration_vectors)
    ]
    length_validation_curve = _threshold_curve(
        split.calibration, length_validation_probabilities, pricing
    )
    length_choice = _choose_balanced_threshold(length_validation_curve)
    length_test_vectors, _ = _prompt_length_vectors(split.train, split.test)
    length_test_probabilities = [
        sigmoid(logit) for logit in length_model.decision_function(length_test_vectors)
    ]
    length_selections = _threshold_selections(
        split.test, length_test_probabilities, float(length_choice["threshold"])
    )
    always_cheap_selection = {row.example_id: "cheap" for row in split.test}
    always_strong_selection = {row.example_id: "strong" for row in split.test}
    oracle_selection = _oracle_selections(split.test)

    raw_metrics = {
        "always_cheap": _selection_metrics(
            split.test, always_cheap_selection, pricing
        ),
        "always_strong": _selection_metrics(
            split.test, always_strong_selection, pricing
        ),
        "oracle_ceiling": _selection_metrics(split.test, oracle_selection, pricing),
        "routed": _selection_metrics(split.test, routed_selections, pricing),
        "routed_point_estimate": _selection_metrics(
            split.test, point_estimate_selections, pricing
        ),
        "routed_safety_recall_95": _selection_metrics(
            split.test, safety_selections, pricing
        ),
        "routed_balanced_accuracy": _selection_metrics(
            split.test, balanced_selections, pricing
        ),
        "prompt_length_logistic": _selection_metrics(
            split.test, length_selections, pricing
        ),
    }
    random_selection = _random_selections(
        split.test, float(raw_metrics["routed"]["strong_call_rate"]), config.seed
    )
    raw_metrics["deterministic_random_matched_rate"] = _selection_metrics(
        split.test, random_selection, pricing
    )
    policies = {
        name: _augment_relative_metrics(
            metrics, raw_metrics["always_strong"], raw_metrics["oracle_ceiling"]
        )
        for name, metrics in raw_metrics.items()
    }

    test_curve = _threshold_curve(split.test, test_probabilities, pricing)
    frontier: list[dict[str, object]] = []
    best_quality = -1.0
    for point in sorted(
        test_curve,
        key=lambda row: (Decimal(str(row["mean_cost"])), -float(row["mean_quality"])),
    ):
        if float(point["mean_quality"]) > best_quality + 1e-12:
            frontier.append(
                {
                    "threshold": point["threshold"],
                    "mean_cost": point["mean_cost"],
                    "mean_quality": point["mean_quality"],
                    "strong_call_rate": point["strong_call_rate"],
                }
            )
            best_quality = float(point["mean_quality"])

    bootstrap = _bootstrap_intervals(
        split.test,
        routed_selections,
        pricing,
        config.bootstrap_replicates,
        config.seed,
        comparators={
            "deterministic_random_matched_rate": random_selection,
            "prompt_length_logistic": length_selections,
            "routed_point_estimate": point_estimate_selections,
        },
    )
    shift = _shift_diagnostics(split)
    slices = _slice_diagnostics(
        split.train, split.test, test_probabilities, routed_selections, pricing
    )
    errors = _error_analysis(split.test, test_probabilities, routed_selections)
    predictions = [
        {
            "example_sha256": hashlib.sha256(row.example_id.encode()).hexdigest(),
            "group_sha256": hashlib.sha256(row.group_id.encode()).hexdigest(),
            "strong_needed_probability": _rounded(probability),
            "selected_model": routed_selections[row.example_id],
            "failure_type": (
                "correct"
                if routed_selections[row.example_id]
                == ("strong" if row.strong_is_better else "cheap")
                else "missed_strong_need"
                if routed_selections[row.example_id] == "cheap"
                else "unnecessary_strong_call"
            ),
            "strong_is_better": bool(row.strong_is_better),
            "cheap_quality": row.cheap_quality,
            "strong_quality": row.strong_quality,
            "model_pair": row.model_pair,
        }
        for row, probability in zip(split.test, test_probabilities)
    ]
    split_manifest = []
    for split_name, split_rows in (
        ("train", split.train),
        ("calibration", split.calibration),
        ("test", split.test),
    ):
        split_manifest.extend(
            {
                "example_sha256": hashlib.sha256(row.example_id.encode()).hexdigest(),
                "group_sha256": hashlib.sha256(row.group_id.encode()).hexdigest(),
                "split": split_name,
                "model_pair": row.model_pair,
            }
            for row in split_rows
        )

    test_auc = _calibration_metrics(
        [row.strong_is_better for row in split.test], test_probabilities
    )["roc_auc"]
    test_quality_drop_upper = -float(
        bootstrap["quality_delta_vs_always_strong"]["lower_95"]
    )
    deployment_blockers = [
        "offline historical proxy does not represent current provider quality or price",
    ]
    if len(split.test) < 200:
        deployment_blockers.append("test cohort is below the predeclared 200-row minimum")
    if test_auc is None or float(test_auc) < 0.65:
        deployment_blockers.append("held-out discrimination is below the 0.65 ROC-AUC gate")
    if test_quality_drop_upper > config.max_quality_drop:
        deployment_blockers.append(
            "held-out 95% quality-loss bound exceeds the configured quality budget"
        )
    if shift["warnings"]:
        deployment_blockers.append("distribution-shift warnings require a fresh evaluation")

    report = {
        "schema": "CarnoCostQualityBenchmarkV2",
        "benchmark_id": "carno_offline_cost_quality_v2",
        "status": "illustrative_synthetic_smoke"
        if "synthetic" in data_path.lower()
        else "offline_proxy_not_provider_claim",
        "dataset": {
            "sha256": file_sha256(data_path),
            "row_count": len(rows),
            "prompt_group_count": len({row.group_id for row in rows}),
            "raw_prompts_in_report": False,
        },
        "pricing": {
            "sha256": file_sha256(pricing_path),
            "snapshot_id": pricing.snapshot_id,
            "currency": pricing.currency,
            "claims_boundary": pricing.claims_boundary,
        },
        "configuration": {
            "seed": config.seed,
            "bootstrap_replicates": config.bootstrap_replicates,
            "max_quality_drop": config.max_quality_drop,
            "selection_bootstrap_replicates": config.selection_bootstrap_replicates,
            "route_strong_when": "calibrated_probability >= threshold",
        },
        "split": split.receipt,
        "model": {
            "family": "train_selected_prompt_only_sparse_logistic",
            "selected_specification": selected_specification.to_dict(),
            "feature_count": len(vectorizer.vocabulary or {}),
            "fingerprint": model.fingerprint(vectorizer),
            "network_access": False,
            "selection_receipt": model_selection_receipt,
        },
        "calibration": {
            "method": "five_fold_group_stratified_oof_platt_scaling",
            "fit_data": "out_of_fold_train_predictions_only",
            "oof_receipt": oof_receipt,
            "fingerprint": calibrator.fingerprint(),
            "slope": _rounded(calibrator.slope),
            "intercept": _rounded(calibrator.intercept),
            "test_metrics": _calibration_metrics(
                [row.strong_is_better for row in split.test], test_probabilities
            ),
        },
        "threshold_selection": {
            "primary_policy": "minimum_validation_cost_subject_to_one_sided_95_percent_quality_risk_bound",
            "selected": primary_choice,
            "point_estimate": point_estimate_choice,
            "safety_recall_95": safety_choice,
            "balanced_accuracy": balanced_choice,
            "test_data_used_for_selection": False,
            "risk_control_receipt": risk_control_receipt,
        },
        "prompt_length_baseline": {
            "family": "two_feature_log_length_logistic",
            "features": ["log1p_word_count", "log1p_character_count"],
            "training_data": "train_only",
            "threshold_selection": "highest_validation_accuracy_then_lower_strong_call_rate",
            "selected": length_choice,
            "scaling": length_scaling,
            "network_access": False,
        },
        "test_policies": policies,
        "test_cost_quality_frontier": frontier,
        "uncertainty": bootstrap,
        "distribution_shift": shift,
        "slice_diagnostics": slices,
        "error_analysis": errors,
        "deployment_decision": {
            "schema": "CarnoDeploymentDecisionV1",
            "offline_research_artifact": "PASS",
            "portfolio_evidence": "PASS_WITH_EXPLICIT_LIMITS",
            "live_autorouting": "BLOCKED",
            "shadow_mode": "REQUIRES_FRESH_CURRENT_PROVIDER_EVALUATION",
            "blockers": deployment_blockers,
        },
        "claim_boundaries": [
            "All savings are modeled under the exact pinned pricing snapshot.",
            "Synthetic-fixture results are engineering smoke evidence, not provider performance.",
            "Offline proxy preference is not absolute model quality and cannot establish production parity.",
            "Provider latency, historical spend, deployment reliability, and customer adoption are not measured.",
            "The cohort uses historical model generations and is not evidence about current frontier models.",
            "No demographic fairness conclusion is supported because protected-attribute labels are absent.",
        ],
    }
    model_bundle = build_model_bundle(
        vectorizer=vectorizer,
        model=model,
        calibrator=calibrator,
        model_fingerprint=model.fingerprint(vectorizer),
        benchmark_id=report["benchmark_id"],
        dataset_sha256=report["dataset"]["sha256"],
        split_receipt=split.receipt,
        threshold_selection=report["threshold_selection"],
    )
    return {
        "report": report,
        "model_bundle": model_bundle,
        "predictions": predictions,
        "split_manifest": split_manifest,
        "validation_threshold_curve": risk_control_curve,
        "bootstrap_intervals": bootstrap,
    }

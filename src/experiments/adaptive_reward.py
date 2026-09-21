"""PPA-guided reward-weight adaptation for EXPlace experiments.

This module deliberately has no dependency on the placement environment or
PyTorch so the outer-loop optimizer can be tested in isolation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeightUpdate:
    """One observation made by the reward-weight learner."""

    observation_count: int
    ppa_penalty: float
    updated: bool
    correlations: dict
    weights: dict


class PPACorrelationWeightLearner:
    """Adapt proxy-reward weights using their correlation with measured PPA.

    A lower PPA penalty is better. A proxy cost receives more weight when its
    historical values are positively correlated with the downstream penalty.
    Negative correlations are clipped because optimizing such a proxy in the
    current direction would be misleading. Updates are smoothed and mixed with
    the initial prior to avoid unstable weight collapse on short histories.
    """

    def __init__(
        self,
        mask_names,
        initial_weights,
        learnable_masks,
        min_observations=4,
        update_rate=0.25,
        temperature=0.25,
        prior_strength=0.25,
        min_weight=0.02,
        hpwl_weight=1.0,
        tns_weight=1.0,
        wns_weight=1.0,
    ):
        self.mask_names = list(mask_names)
        initial_weights = np.asarray(initial_weights, dtype=np.float64)
        if len(self.mask_names) != len(initial_weights):
            raise ValueError("mask_names and initial_weights must have equal length")
        if np.any(initial_weights < 0) or initial_weights.sum() <= 0:
            raise ValueError("initial_weights must be non-negative with a positive sum")

        unknown = set(learnable_masks) - set(self.mask_names)
        if unknown:
            raise ValueError(f"unknown learnable masks: {sorted(unknown)}")
        self.learnable_indices = [
            idx for idx, name in enumerate(self.mask_names) if name in learnable_masks
        ]
        if not self.learnable_indices:
            raise ValueError("at least one reward mask must be learnable")
        if min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if not 0 < update_rate <= 1:
            raise ValueError("update_rate must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= prior_strength <= 1:
            raise ValueError("prior_strength must be in [0, 1]")
        if min_weight < 0:
            raise ValueError("min_weight must be non-negative")

        self.weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        self.learnable_budget = float(initial_weights[self.learnable_indices].sum())
        if min_weight * len(self.learnable_indices) > self.learnable_budget:
            raise ValueError("min_weight is too large for the learnable weight budget")
        self.prior = initial_weights[self.learnable_indices] / self.learnable_budget
        self.min_observations = min_observations
        self.update_rate = update_rate
        self.temperature = temperature
        self.prior_strength = prior_strength
        self.min_weight = min_weight
        self.hpwl_weight = hpwl_weight
        self.tns_weight = tns_weight
        self.wns_weight = wns_weight
        self.cost_history = []
        self.ppa_history = []
        self.last_correlations = np.zeros(len(self.learnable_indices), dtype=np.float64)

    def observe(self, costs, gp_hpwl, tns=0.0, wns=0.0):
        """Add one downstream evaluation and possibly update reward weights."""
        cost_vector = np.asarray(
            [float(costs[name]) for name in self.mask_names], dtype=np.float64
        )
        if not np.all(np.isfinite(cost_vector)):
            raise ValueError("all proxy costs must be finite")

        ppa_penalty = self._ppa_penalty(gp_hpwl, tns, wns)
        self.cost_history.append(cost_vector)
        self.ppa_history.append(ppa_penalty)
        updated = len(self.ppa_history) >= self.min_observations

        if updated:
            costs_array = np.asarray(self.cost_history)[:, self.learnable_indices]
            penalty_array = np.asarray(self.ppa_history)
            correlations = np.asarray(
                [self._rank_correlation(costs_array[:, idx], penalty_array)
                 for idx in range(costs_array.shape[1])]
            )
            self.last_correlations = correlations
            useful = np.clip(correlations, 0.0, None)
            if np.allclose(useful, 0.0):
                learned_distribution = self.prior.copy()
            else:
                logits = useful / self.temperature
                logits -= logits.max()
                learned_distribution = np.exp(logits)
                learned_distribution /= learned_distribution.sum()

            distribution = (
                (1.0 - self.prior_strength) * learned_distribution
                + self.prior_strength * self.prior
            )
            remaining_budget = (
                self.learnable_budget - self.min_weight * len(self.learnable_indices)
            )
            target = self.min_weight + remaining_budget * distribution
            current = self.weights[self.learnable_indices]
            new_weights = (1.0 - self.update_rate) * current + self.update_rate * target
            # Correct floating-point drift while preserving fixed-mask weights.
            new_weights *= self.learnable_budget / new_weights.sum()
            self.weights[self.learnable_indices] = new_weights

        return RewardWeightUpdate(
            observation_count=len(self.ppa_history),
            ppa_penalty=ppa_penalty,
            updated=updated,
            correlations=dict(zip(self.learnable_names, self.last_correlations.tolist())),
            weights=self.weight_dict,
        )

    @property
    def learnable_names(self):
        return [self.mask_names[idx] for idx in self.learnable_indices]

    @property
    def weight_dict(self):
        return dict(zip(self.mask_names, self.weights.tolist()))

    def _ppa_penalty(self, gp_hpwl, tns, wns):
        metrics = np.asarray([gp_hpwl, tns, wns], dtype=np.float64)
        if not np.all(np.isfinite(metrics)):
            raise ValueError("PPA metrics must be finite")
        return float(
            self.hpwl_weight * np.log1p(max(float(gp_hpwl), 0.0))
            + self.tns_weight * np.log1p(max(-float(tns), 0.0))
            + self.wns_weight * np.log1p(max(-float(wns), 0.0))
        )

    @staticmethod
    def _rank_correlation(values, targets):
        """Spearman-style correlation without requiring SciPy."""
        if np.ptp(values) <= 1e-12 or np.ptp(targets) <= 1e-12:
            return 0.0
        value_ranks = PPACorrelationWeightLearner._rank_with_ties(values)
        target_ranks = PPACorrelationWeightLearner._rank_with_ties(targets)
        correlation = np.corrcoef(value_ranks, target_ranks)[0, 1]
        return float(correlation) if np.isfinite(correlation) else 0.0

    @staticmethod
    def _rank_with_ties(values):
        """Return zero-based average ranks so tied observations stay neutral."""
        values = np.asarray(values)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks


def build_reward_weight_learner(args):
    """Construct the configured learner, or return None for the baseline."""
    mode = getattr(args, "reward_weight_mode", "fixed")
    if mode == "fixed":
        return None
    if mode != "ppa_correlation":
        raise ValueError(f"unsupported reward_weight_mode: {mode}")

    mask_names = list(args.used_masks[:-1])
    return PPACorrelationWeightLearner(
        mask_names=mask_names,
        initial_weights=args.trade_off_coeff,
        learnable_masks=getattr(args, "learnable_masks", mask_names),
        min_observations=getattr(args, "reward_weight_min_observations", 4),
        update_rate=getattr(args, "reward_weight_update_rate", 0.25),
        temperature=getattr(args, "reward_weight_temperature", 0.25),
        prior_strength=getattr(args, "reward_weight_prior_strength", 0.25),
        min_weight=getattr(args, "reward_weight_min", 0.02),
        hpwl_weight=getattr(args, "ppa_hpwl_weight", 1.0),
        tns_weight=getattr(args, "ppa_tns_weight", 1.0),
        wns_weight=getattr(args, "ppa_wns_weight", 1.0),
    )

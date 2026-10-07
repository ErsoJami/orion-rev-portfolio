"""Exact contextual-bandit risk for finite actions; not a Jev RLCD reproduction."""

import torch


def expected_decision_cost(logits, targets, execute_options):
    """Cost0 correct,1 incorrect,5 false EXECUTE when gold has no action.

    execute_options has the same [views, options] axes as logits. No sampling
    variance is necessary: every possible action's reward is known from gold.
    """
    assert logits.ndim == 2 and logits.shape == execute_options.shape
    assert targets.shape == logits.shape[:1] and execute_options.dtype == torch.bool
    probabilities = logits.softmax(-1)
    selected = probabilities.gather(1, targets[:, None]).squeeze(1)
    gold_execute = execute_options.gather(1, targets[:, None]).squeeze(1)
    execution_mass = (probabilities * execute_options).sum(-1)
    cost = 1 - selected + 4 * (~gold_execute) * execution_mass
    return cost.mean()


def execute_mask(candidates, orders, device):
    return torch.tensor(
        [[candidates[j]["decision"] == "EXECUTE" for j in order] for order in orders],
        dtype=torch.bool,
        device=device,
    )

import torch
from torch_geometric.utils import scatter


def pprint_pyg_obj(batch):
    """For printing out the pytorch geometric attributes in a readable way."""
    for key in vars(batch)["_store"].keys():
        if key.startswith("_"):
            continue
        print(f"{key}: {batch[key].shape}")


def safe_divide_loss(loss, num_choices):
    """Divide `loss` by `num_choices`, but guard against `num_choices` being 0."""
    return loss / max(num_choices, 1.0)


SMALL_NUMBER, BIG_NUMBER = 1e-7, 1e7


def compute_neglogprob_for_multihot_objective(
    logprobs,
    multihot_labels,
    per_decision_num_correct_choices,
):
    # Normalise by number of correct choices and mask out entries for wrong decisions:
    return -(
        (logprobs + torch.log(per_decision_num_correct_choices + SMALL_NUMBER))
        * multihot_labels
        / (per_decision_num_correct_choices + SMALL_NUMBER)
    )


def traced_unsorted_segment_log_softmax(
    logits,  # edge_candidate_logits
    segment_ids,  # edge_candidate_to_graph_map
    num_segments,  # num_graphs_in_batch
):
    """Basically compute the log softmax for an array that contains a mix of a few different
    groups of logits. The final result is that log softmax is applied to each individual group
    of logits."""
    max_per_segment = scatter(logits, segment_ids, reduce="max")
    scattered_maxes = max_per_segment[segment_ids]
    recentered_scores = logits - scattered_maxes
    exped_recentered_scores = torch.exp(recentered_scores)

    per_segment_sums = scatter(exped_recentered_scores, segment_ids, reduce="sum")
    per_segment_normalization_consts = torch.log(per_segment_sums)

    log_probs = recentered_scores - per_segment_normalization_consts[segment_ids]
    return log_probs


import numpy as np

def sample_indices_from_logprobs(
    num_samples, sampling_mode, logprobs
):
    """Samples indices (without replacement) given the log-likelihoods.

    Args:
        num_samples: intended number of samples
        sampling_mode: sampling method (greedy
        logprobs: log-probabilities of selecting appropriate entries

    Returns:
        indices of picked values, shape (n,), where n = min(num_samples, available_samples)

    Note:
        the ordering of returned indices is arbitrary
    """
    num_choices = logprobs.shape[0]
    indices = np.arange(num_choices)
    num_samples = min(num_samples, num_choices)  # Handle cases where we only have few candidates
    if sampling_mode == 'greedy':
        # Note that this will return the top num_samples indices, but not in order:
        picked_indices = np.argpartition(logprobs, -num_samples)[-num_samples:]
    # elif sampling_mode == DecoderSamplingMode.SAMPLING:
    #     p = np.exp(logprobs)  # Convert to probabilities
    #     # We can only sample values with non-zero probabilities
    #     num_choices = np.sum(p > 0)
    #     num_samples = min(num_samples, num_choices)
    #     picked_indices = np.random.choice(
    #         indices,
    #         size=(num_samples,),
    #         replace=False,
    #         p=p,
    #     )
    else:
        raise ValueError(f"Sampling method {sampling_mode} not known.")

    return picked_indices
from torch.utils.data import Sampler


class DuplicatedIndicesSamplerWrapper(Sampler):
    """
    Samples randomly, and then duplicates each index by a fixed frequency given by a frequency_mapping.
    """

    def __init__(
        self,
        sampler,
        frequency_mapping,
    ):
        self.frequency_mapping = frequency_mapping
        self.sampler = sampler

    def __iter__(self):
        indices = list(self.sampler)
        return (
            idx for idx in indices for _ in range(self.frequency_mapping.get(idx, 1))
        )

import torch


class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        positive_number = int(self.num_samples * self.positive_fraction)
        positive_number = min(positive.numel(), positive_number)
        negative_number = self.num_samples - positive_number
        negative_number = min(negative.numel(), negative_number)

        random_positive = torch.randperm(positive.numel(), device=positive.device)[:positive_number]
        random_negative = torch.randperm(negative.numel(), device=negative.device)[:negative_number]

        pos_idx = positive[random_positive]
        neg_idx = negative[random_negative]

        return pos_idx, neg_idx

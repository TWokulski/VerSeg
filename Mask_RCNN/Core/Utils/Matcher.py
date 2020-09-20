import torch


class Matcher:
    def __init__(self, high_th, low_th, low_quality_matches=False):
        self.high_threshold = high_th
        self.low_threshold = low_th
        self.allow_low_quality = low_quality_matches

    def __call__(self, quality_matrix):
        matched_value, matched_idx = quality_matrix.max(dim=0)
        label = torch.full((quality_matrix.shape[1],), -1, dtype=torch.float, device=quality_matrix.device)

        label[matched_value >= self.high_threshold] = 1
        label[matched_value < self.low_threshold] = 0

        if self.allow_low_quality:
            highest_quality = quality_matrix.max(dim=1)[0]
            get_prediction_pairs = torch.where(quality_matrix == highest_quality[:, None])[1]
            label[get_prediction_pairs] = 1

        return label, matched_idx

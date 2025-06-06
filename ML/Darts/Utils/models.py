from darts.utils.likelihood_models import GaussianLikelihood
import torch

class PositiveGaussianLikelihood(GaussianLikelihood):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)

        # Ensure that std is always non-negative before taking the exp
        if torch.any(result.std < 0):
            result.std = torch.abs(result.std)  # Take absolute value to prevent negative std

        result.std = torch.exp(result.std)  # Ensures a positive std after exp
        return result


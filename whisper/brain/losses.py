import torch
import torch.nn.functional as F





def alignment_loss(brain_embeddings, audio_embeddings):
    # mean square error loss
    mse_loss = F.mse_loss(brain_embeddings, audio_embeddings)
    # cosine similarity along the time axis
    brain_norm = F.normalize(brain_embeddings, dim=1)
    audio_norm = F.normalize(audio_embeddings, dim=1)
    cosine_sim = torch.einsum('nct,ncp->ntp', brain_norm, audio_norm)
    # mean of the diagonal
    time_alignment = cosine_sim.diagonal(dim1=1, dim2=2).mean()
    # cosine similarity along the feature axis
    brain_norm = F.normalize(brain_embeddings, dim=2)
    audio_norm = F.normalize(audio_embeddings, dim=2)
    cosine_sim = torch.einsum('nct,npt->nctp', brain_norm, audio_norm)
    # mean of the diagonal
    feature_alignment = cosine_sim.diagonal(dim1=2, dim2=3).mean()
    return mse_loss, time_alignment, feature_alignment


def gaussian_kernel(iqr):
    sigma = iqr / 1.349  # Calculate sigma from IQR
    window_size = int(2 * sigma) * 2 + 1
    x = torch.arange(window_size) - window_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize the kernel
    return kernel, sigma

def temporal_gaussian_infonce_loss(brain_embeddings, audio_embeddings, temperature=0.07, iqr=20, neg_sample_prop=None):
    N, C, T = brain_embeddings.shape

    # Normalize the embeddings
    brain_norm = F.normalize(brain_embeddings, dim=1)
    audio_norm = F.normalize(audio_embeddings, dim=1)

    # Calculate the pairwise cosine similarities
    cosine_sim = torch.einsum('nct,ncp->ntp', brain_norm, audio_norm) / temperature

    # Create a Gaussian kernel based on the provided IQR
    kernel, sigma = gaussian_kernel(iqr)
    kernel = kernel.to(brain_embeddings.device)
    window_size = len(kernel)

    # Apply the Gaussian kernel to the similarities
    gaussian_weights = F.pad(kernel, (0, T - window_size), mode='constant', value=0)
    gaussian_weights = gaussian_weights.unsqueeze(0).expand(T, -1)
    weighted_cosine_sim = torch.zeros_like(cosine_sim)

    for t in range(T):
        weighted_cosine_sim[:, t, :] = cosine_sim[:, t, :] * gaussian_weights[t]

    # InfoNCE loss calculation
    pos_sim = torch.exp(weighted_cosine_sim.diagonal(dim1=1, dim2=2))  # Positive pairs

    # Mask out the positive pairs and nearby pairs for negative samples
    mask_radius = int(2 * sigma)
    mask = torch.eye(T, device=brain_embeddings.device).unsqueeze(0).expand(N, -1, -1)
    for i in range(1, mask_radius + 1):
        mask += torch.eye(T, device=brain_embeddings.device).unsqueeze(0).expand(N, -1, -1).roll(shifts=i, dims=1)
        mask += torch.eye(T, device=brain_embeddings.device).unsqueeze(0).expand(N, -1, -1).roll(shifts=-i, dims=1)

    neg_sim = torch.exp(weighted_cosine_sim) * (1 - mask)

    # Optionally retain only a proportion or a fixed number of negative samples
    if neg_sample_prop is not None:
        if isinstance(neg_sample_prop, float) and 0 < neg_sample_prop < 1:
            num_neg_samples = int(neg_sample_prop * T)
        elif isinstance(neg_sample_prop, int) and neg_sample_prop > 1:
            num_neg_samples = neg_sample_prop
        else:
            raise ValueError("neg_sample_prop should be a float between 0 and 1, or an integer greater than 1")

        neg_indices = torch.randperm(T)[:num_neg_samples].to(brain_embeddings.device)
        neg_sim = neg_sim[:, :, neg_indices]

    loss = -torch.log(pos_sim / neg_sim.sum(dim=-1)).mean()

    return loss

def mse_adjusted_temporal_gaussian_infonce_loss(brain_embeddings, audio_embeddings, sequence_length, alpha=0.5, iqr=20, temperature=0.07, neg_sample_prop=None):
    mse_loss = F.mse_loss(brain_embeddings, audio_embeddings)
    brain_embeddings = brain_embeddings[:, :, :sequence_length//2]
    audio_embeddings = audio_embeddings[:, :, :sequence_length//2]
    contrastive_loss = temporal_gaussian_infonce_loss(brain_embeddings, audio_embeddings, temperature, iqr=iqr, neg_sample_prop=neg_sample_prop)

    # Combine the losses using a static alpha
    combined_loss = alpha * contrastive_loss + (1 - alpha) * mse_loss

    return combined_loss


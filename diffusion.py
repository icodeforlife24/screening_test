import torch

def sample_ddpm(
    model,
    alphas,
    alpha_bars,
    betas,
    T,
    device,
    n_samples=200,
    latent_dim=32
):

    model.eval()

    x = torch.randn(
        n_samples,
        latent_dim,
        device=device
    )

    with torch.no_grad():

        for t in reversed(range(T)):

            t_batch = torch.full(
                (n_samples,),
                t,
                device=device,
                dtype=torch.long
            )

            pred_noise = model(
                x,
                t_batch
            )

            alpha = alphas[t]
            alpha_bar = alpha_bars[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                (1/torch.sqrt(alpha))
                *
                (
                    x
                    -
                    (
                        (1-alpha)
                        /
                        torch.sqrt(1-alpha_bar)
                    )
                    *
                    pred_noise
                )
                +
                torch.sqrt(beta)
                *
                noise
            )

    return x.cpu()
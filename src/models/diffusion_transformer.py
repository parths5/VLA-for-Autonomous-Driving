import torch
from torch import nn, Tensor
import math
from typing import Tuple, Optional

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])

class VectorInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, input_size] and returns an output of shape [B, 1, token_size]
    Args:
        input_size: Expected feature dimension of input tensor.
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """

    def __init__(
        self,
        input_size: int,
        token_size: int = 258,
        hidden_size: int = 64,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        # store args
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.norm_layer = norm_layer
        # networks
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, token_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, input_size]

        Returns:
            Output with dims [B, 1, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x).unsqueeze(1)
        return x
    

class PolicyDiffusionTransformer(nn.Module):

    def __init__(
        self,
        num_transformer_layers,
        image_encoding_dim,
        act_dim=2,  # each waypoint is (x, y)
        hidden_size=2048,
        n_transformer_heads=8,
        device="cpu",
        target="diffusion_policy",
    ):
        super(PolicyDiffusionTransformer, self).__init__()
        assert target in [
            "diffusion_policy",
            "value_model",
        ], f"target must be either 'diffusion_policy' or 'value_model', but got {target}"
        # saving constants
        self.num_transformer_layers = num_transformer_layers
        self.image_encoding_dim = image_encoding_dim  # flattened: 250 * 2560 = 640000
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.n_transformer_heads = n_transformer_heads
        self.device = device

        # fixed sinusoidal timestep embeddings for diffusion noise level
        self.sinusoidal_timestep_embeddings = (
            self.get_all_sinusoidal_timestep_embeddings(
                self.hidden_size, max_period=10000
            )
        )
        self.sinusoidal_linear_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # embed state, action, overall_return_left into hidden_size
        self.image_embedding = nn.Linear(image_encoding_dim, self.hidden_size)
        self.velocity_embedding = VectorInputAdaptor(1, self.hidden_size, 64)
        self.act_embedding = nn.Linear(self.act_dim, self.hidden_size)

        # transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=0.01,  # 0.05
            activation="gelu",
            norm_first=True,  # apply layernorm before attention (see https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm)
            batch_first=True,  # batch size comes first in input tensors
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_transformer_layers,
        )

        # decode results into final actions
        if target == "diffusion_policy":
            self.predict_noise = nn.Sequential(
                nn.Linear(self.hidden_size, self.act_dim),
            )
        # decode results into single value for value model
        else:
            self.predict_noise = nn.Sequential(
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

        # set device
        self.to(self.device)

    # from https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L103
    def get_all_sinusoidal_timestep_embeddings(
        self, dim, max_period=10000, num_timesteps=1000
    ):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        timesteps = torch.arange(0, num_timesteps, device=self.device)
        half = dim // 2
        logs = -math.log(max_period)
        arange = torch.arange(start=0, end=half, dtype=torch.float32)
        logfreqs = logs * arange / half
        freqs = torch.exp(logfreqs).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(
        self,
        image_encodings,
        velocities,
        noisy_actions,
        noise_timesteps,
        image_mask=None,
        velocity_mask=None,
        actions_padding_mask=None,
    ):
        """
        forward pass of the model

        Args:
            image_encodings (torch.Tensor): flattened image encodings, shape (batch_size, 1, image_encoding_dim)
            velocities (torch.Tensor): current velocity, shape (batch_size, 1, 1)
            noisy_actions (torch.Tensor): noisy actions to be denoised via noise prediction, shape (batch_size, num_future_actions, act_dim)
            noise_timesteps (torch.Tensor): noise timesteps for diffusion (higher timesteps implies more noisy action), shape (batch_size, 1)
            image_mask (torch.Tensor): mask for image encodings, shape (batch_size, 1)
            velocity_mask (torch.Tensor): mask for velocity, shape (batch_size, 1)
            actions_padding_mask (torch.Tensor): mask for noisy actions, shape (batch_size, num_future_actions)
        """

        # get batch size and action sequence length
        batch_size, num_future_actions = noisy_actions.shape[0], noisy_actions.shape[1]

        # embed image encodings and velocity (both have seq_length=1)
        # image_encodings: (B, 1, 640000) -> (B, 1, hidden_size)
        image_embeddings = self.image_embedding(image_encodings)

        # velocities: (B, 1, 1) -> (B, 1, hidden_size)
        velocity_embeddings = self.velocity_embedding(velocities.squeeze(-1))

        # noisy_actions: (B, num_future_actions, act_dim) -> (B, num_future_actions, hidden_size)
        noisy_actions_embeddings = self.act_embedding(noisy_actions)

        # get fixed timestep embeddings for diffusion noise level
        # noise_timesteps: (B, 1) -> (B, 1, hidden_size)
        noise_timestep_embeddings = self.sinusoidal_timestep_embeddings[noise_timesteps]

        # apply linear layer to sinusoidal timestep embeddings
        noise_timestep_embeddings = self.sinusoidal_linear_layer(
            noise_timestep_embeddings
        )

        # concatenate conditioning: [noise_timestep, image_encoding, velocity]
        # All have seq_length=1, so concatenate along sequence dimension
        # (B, 1, hidden_size) + (B, 1, hidden_size) + (B, 1, hidden_size) -> (B, 3, hidden_size)
        conditioning = torch.cat(
            (noise_timestep_embeddings, image_embeddings, velocity_embeddings), dim=1
        )

        # Create padding masks
        # If no mask is given for conditioning, all are valid (False = not masked)
        if image_mask is None:
            image_mask = torch.zeros(batch_size, 1, device=self.device).bool()
        if velocity_mask is None:
            velocity_mask = torch.zeros(batch_size, 1, device=self.device).bool()

        # Conditioning mask: [noise_timestep (always valid), image, velocity]
        conditioning_mask = torch.cat(
            (
                torch.zeros(batch_size, 1, device=self.device).bool(),  # noise timestep always valid
                image_mask,
                velocity_mask,
            ),
            dim=1,
        )

        # If actions mask is None, all actions are valid
        if actions_padding_mask is None:
            actions_padding_mask = torch.zeros(
                batch_size, num_future_actions, device=self.device
            ).bool()

        # Causal mask: future actions only depend on past actions
        # This ensures autoregressive generation during diffusion
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            num_future_actions
        ).to(self.device)

        # Run transformer decoder
        # tgt: noisy actions to denoise (B, num_future_actions, hidden_size)
        # memory: conditioning (B, 3, hidden_size) - [noise_timestep, image, velocity]
        output = self.decoder(
            tgt=noisy_actions_embeddings,
            memory=conditioning,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=actions_padding_mask,
            memory_key_padding_mask=conditioning_mask,
        )

        # predict noise: (B, num_future_actions, hidden_size) -> (B, num_future_actions, act_dim)
        noise_preds = self.predict_noise(output)
        return noise_preds

import torch
from torch import nn
import math


class ImageEmbeddingProjector(nn.Module):
    def __init__(self, image_embedding_tokens, image_embedding_dim, image_hidden_dim, transformer_hidden_dim):
        super().__init__()
        self.image_to_hidden = nn.Sequential(
            nn.Linear(image_embedding_dim, image_hidden_dim),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.ReLU(),
            nn.Linear(image_embedding_tokens * image_hidden_dim, transformer_hidden_dim)
        )
    def forward(self, x):
        return self.image_to_hidden(x)

class StateProjector(nn.Module):
    def __init__(self, state_dim, transformer_hidden_dim):
        super().__init__()
        self.velocity_projection = nn.Sequential(
            nn.Linear(state_dim, transformer_hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.velocity_projection(x)

class OutputProjector(nn.Module):
    def __init__(self, output_dim, transformer_hidden_dim):
        super().__init__()
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, transformer_hidden_dim)
        )
    def forward(self, x):
        return self.output_projection(x)

class ProjectionToOutput(nn.Module):
    def __init__(self, transformer_hidden_dim, output_dim):
        super().__init__()
        self.projection_to_output = nn.Sequential(
            nn.Linear(transformer_hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection_to_output(x)

class TimeEncoding(nn.Module):
    def __init__(self, time_hidden_dim, transformer_hdden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_encoding = nn.Sequential(
            nn.Linear(1, time_hidden_dim),
            nn.ReLU(),
            nn.Linear(time_hidden_dim, transformer_hdden_dim)
        )
    def forward(self, x):
        return self.time_encoding(x)
    
class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Recreate original forward function but also store attention weights

        # Self attention
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=True
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2, cross_attn_weights = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=True
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Return attention weights for layer
        attn_weights = [self_attn_weights, cross_attn_weights]
        return tgt, attn_weights

class DriveFusion(nn.Module):
    def __init__(self, state_dim, time_hidden_dim, image_embeddings_tokens, image_embedding_dim, image_hidden_dim, output_dim, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, transformer_num_feedforward, transformer_dropout):
        super().__init__()
        self.transformer_num_layers = transformer_num_layers
        self.image_embedding_projector = ImageEmbeddingProjector(image_embeddings_tokens, image_embedding_dim, image_hidden_dim, transformer_hidden_dim)
        self.state_projector = StateProjector(state_dim, transformer_hidden_dim)
        self.output_in_projection = OutputProjector(output_dim, transformer_hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [CustomDecoderLayer(
                transformer_hidden_dim, transformer_num_heads, transformer_num_feedforward, transformer_dropout, batch_first=True
            ) for _ in range(transformer_num_layers)]
        )
        self.projection_to_output = ProjectionToOutput(transformer_hidden_dim, output_dim)
        self.time_encoding = TimeEncoding(time_hidden_dim, transformer_hidden_dim)
        self.sinusoidal_timestep_embeddings = (
            self.get_all_sinusoidal_timestep_embeddings(
                transformer_hidden_dim, max_period=10000
            )
        )
        self.sinusoidal_linear_layer = nn.Linear(transformer_hidden_dim, transformer_hidden_dim)


    def forward(self, image_embeddings, states, outputs_in, history_ts, denoising_step):

        seq_len = outputs_in.shape[1]
        projected_image_embeddings = self.image_embedding_projector(image_embeddings)
        projected_states = self.state_projector(states)
        projected_times = self.time_encoding(history_ts)

        denoising_timestep_embedding = self.sinusoidal_timestep_embeddings[denoising_step]

        # apply linear layer to sinusoidal timestep embeddings
        denoising_timestep_projection = self.sinusoidal_linear_layer(
            denoising_timestep_embedding
        )

        projected_outputs_in = self.output_in_projection(outputs_in)

        memory = torch.cat([projected_image_embeddings + projected_times, projected_states + projected_times, denoising_timestep_projection], axis=1)

        # Infer device from input tensor
        device = outputs_in.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        hidden = projected_outputs_in   
        attn_weights = {}     
        for i, layer in enumerate(self.transformer_blocks):
            hidden, attn_weights_layer = layer(
                tgt=hidden, memory=memory, tgt_mask=tgt_mask, memory_mask=None,  
            )
            attn_weights[f'self_attn_layer_{i}'] = attn_weights_layer[0]
            attn_weights[f'cross_attn_layer_{i}'] = attn_weights_layer[1]
        
        return self.projection_to_output(hidden), attn_weights
    
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
 
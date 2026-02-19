import os
import sys
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusion_transformer import PolicyDiffusionTransformer
from diffusers import DDPMScheduler

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataloaders.bench2drive_dataset import Bench2DriveDataset

try:
    import wandb
except ImportError:
    wandb = None


class TrainDiffusionPolicy:
    """
    Training class for diffusion policy on Bench2Drive dataset.

    Input: Flattened image encodings (640,000-dim) + current velocity (1-dim)
    Output: 10 future waypoints, each (x, y) = 20-dim total
    """

    def __init__(
        self,
        model: PolicyDiffusionTransformer,
        optimizer: torch.optim.Optimizer,
        train_dataloader,
        val_dataloader=None,
        device="cpu",
        num_train_diffusion_timesteps=30,
        clip_sample_range=1.0,
        stats_file="checkpoints/diffusion_policy/normalization_stats.pt",
        config=None,
    ):
        """
        Initialize the TrainDiffusionPolicy class.

        Args:
            model: PolicyDiffusionTransformer to train
            optimizer: Optimizer for training
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            device: Device to use for training
            num_train_diffusion_timesteps: Number of diffusion timesteps
            clip_sample_range: Range to clip samples during diffusion
            stats_file: Path to save/load normalization statistics
            config: Dictionary of hyperparameters for wandb logging
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_train_diffusion_timesteps = num_train_diffusion_timesteps
        self.clip_sample_range = clip_sample_range
        self.config = config or {}

        # Move model to device
        self.model.set_device(self.device)

        # Compute/load normalization statistics from training data
        print("Computing normalization statistics...")
        # Create directory for stats file if it doesn't exist
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        self._compute_normalization_stats(stats_file=stats_file)

        # Training and inference schedulers for diffusion
        self.training_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler.alphas_cumprod = (
            self.inference_scheduler.alphas_cumprod.to(self.device)
        )

    def _compute_normalization_stats(self, stats_file="normalization_stats.pt"):
        """
        Compute mean and std for normalization from training data.
        We need to normalize:
        - Image embeddings: (640,000-dim)
        - Velocity: (1-dim)
        - Waypoints: (10, 2)

        Args:
            stats_file: Path to save/load normalization statistics
        """
        # Check if stats file exists
        if os.path.exists(stats_file):
            print(f"  Loading normalization statistics from {stats_file}...")
            stats = torch.load(stats_file)
            self.embedding_mean = stats['embedding_mean'].to(self.device)
            self.embedding_std = stats['embedding_std'].to(self.device)
            self.velocity_mean = stats['velocity_mean'].to(self.device)
            self.velocity_std = stats['velocity_std'].to(self.device)
            self.waypoint_mean = stats['waypoint_mean'].to(self.device)
            self.waypoint_std = stats['waypoint_std'].to(self.device)

            print(f"  Loaded statistics from cache")
            print(f"  Embedding stats - mean: {self.embedding_mean.mean():.4f}, std: {self.embedding_std.mean():.4f}")
            print(f"  Velocity stats - mean: {self.velocity_mean:.4f}, std: {self.velocity_std:.4f}")
            print(f"  Waypoint stats - mean: {self.waypoint_mean.mean():.4f}, std: {self.waypoint_std.mean():.4f}")
            return

        print("  Collecting statistics from training data...")

        # Collect samples for statistics
        embeddings_list = []
        velocities_list = []
        waypoints_list = []

        # Sample a subset for efficiency (max 1000 samples)
        max_samples = min(1000, len(self.train_dataloader.dataset))
        indices = np.random.choice(len(self.train_dataloader.dataset), max_samples, replace=False)

        for idx in tqdm(indices, desc="  Computing stats"):
            sample = self.train_dataloader.dataset[idx]
            # Flatten embeddings: (250, 2560) -> (640000,)
            embeddings_list.append(sample['embeddings'].flatten().float().numpy())
            velocities_list.append(sample['speed'])
            waypoints_list.append(sample['waypoints'].flatten())  # (10, 2) -> (20,)

        # Stack and compute statistics
        embeddings_array = np.stack(embeddings_list, axis=0)  # (N, 640000)
        velocities_array = np.array(velocities_list)  # (N,)
        waypoints_array = np.stack(waypoints_list, axis=0)  # (N, 20)

        # Compute mean and std
        self.embedding_mean = torch.tensor(embeddings_array.mean(axis=0), dtype=torch.float32).to(self.device)
        self.embedding_std = torch.tensor(embeddings_array.std(axis=0) + 1e-6, dtype=torch.float32).to(self.device)

        self.velocity_mean = torch.tensor(velocities_array.mean(), dtype=torch.float32).to(self.device)
        self.velocity_std = torch.tensor(velocities_array.std() + 1e-6, dtype=torch.float32).to(self.device)

        self.waypoint_mean = torch.tensor(waypoints_array.mean(axis=0), dtype=torch.float32).to(self.device)
        self.waypoint_std = torch.tensor(waypoints_array.std(axis=0) + 1e-6, dtype=torch.float32).to(self.device)

        # Save statistics to file
        print(f"  Saving normalization statistics to {stats_file}...")
        torch.save({
            'embedding_mean': self.embedding_mean.cpu(),
            'embedding_std': self.embedding_std.cpu(),
            'velocity_mean': self.velocity_mean.cpu(),
            'velocity_std': self.velocity_std.cpu(),
            'waypoint_mean': self.waypoint_mean.cpu(),
            'waypoint_std': self.waypoint_std.cpu(),
        }, stats_file)

        print(f"  Embedding stats - mean: {self.embedding_mean.mean():.4f}, std: {self.embedding_std.mean():.4f}")
        print(f"  Velocity stats - mean: {self.velocity_mean:.4f}, std: {self.velocity_std:.4f}")
        print(f"  Waypoint stats - mean: {self.waypoint_mean.mean():.4f}, std: {self.waypoint_std.mean():.4f}")

    def normalize_state(self, embeddings, velocity):
        """
        Normalize state (embeddings + velocity).

        Args:
            embeddings: (B, 250, 2560) bfloat16
            velocity: (B,) float32

        Returns:
            normalized_embeddings: (B, 1, 640000) float32
            normalized_velocity: (B, 1, 1) float32
        """
        # Flatten embeddings: (B, 250, 2560) -> (B, 640000)
        B = embeddings.shape[0]
        flat_embeddings = embeddings.reshape(B, -1).float()

        # Normalize embeddings
        normalized_embeddings = (flat_embeddings - self.embedding_mean) / self.embedding_std

        # Add sequence dimension: (B, 640000) -> (B, 1, 640000)
        normalized_embeddings = normalized_embeddings.unsqueeze(1)

        # Normalize velocity
        normalized_velocity = (velocity - self.velocity_mean) / self.velocity_std

        # Reshape: (B,) -> (B, 1, 1)
        normalized_velocity = normalized_velocity.unsqueeze(1).unsqueeze(1)

        return normalized_embeddings, normalized_velocity

    def normalize_actions(self, waypoints):
        """
        Normalize waypoints (actions).

        Args:
            waypoints: (B, 10, 2) float32

        Returns:
            normalized_waypoints: (B, 10, 2) float32
        """
        B = waypoints.shape[0]
        # Flatten: (B, 10, 2) -> (B, 20)
        flat_waypoints = waypoints.reshape(B, -1)

        # Normalize
        normalized_flat = (flat_waypoints - self.waypoint_mean) / self.waypoint_std

        # Reshape back: (B, 20) -> (B, 10, 2)
        normalized_waypoints = normalized_flat.reshape(B, 10, 2)

        return normalized_waypoints

    def denormalize_actions(self, normalized_waypoints):
        """
        Denormalize waypoints (actions).

        Args:
            normalized_waypoints: (B, 10, 2) float32

        Returns:
            waypoints: (B, 10, 2) float32
        """
        B = normalized_waypoints.shape[0]
        # Flatten
        normalized_flat = normalized_waypoints.reshape(B, -1)

        # Denormalize
        flat_waypoints = normalized_flat * self.waypoint_std + self.waypoint_mean

        # Reshape back
        waypoints = flat_waypoints.reshape(B, 10, 2)

        return waypoints

    def training_step(self, batch):
        """
        Runs a single training step on a batch.

        Args:
            batch: Dictionary from dataloader with keys:
                - embeddings: (B, 250, 2560)
                - speed: (B,)
                - waypoints: (B, 10, 2)

        Returns:
            dict: Dictionary containing loss and other metrics
        """
        # Move batch to device
        embeddings = batch['embeddings'].to(self.device)
        velocity = batch['speed'].to(self.device)
        waypoints = batch['waypoints'].to(self.device)

        batch_size = embeddings.shape[0]

        # Normalize states and actions
        normalized_embeddings, normalized_velocity = self.normalize_state(embeddings, velocity)
        normalized_waypoints = self.normalize_actions(waypoints)

        # Sample random noise
        eps = torch.randn_like(normalized_waypoints)

        # Sample random timesteps (avoid t=0 for stability)
        t = torch.randint(
            1, self.num_train_diffusion_timesteps, (batch_size,), device=self.device
        )

        # Add noise to waypoints
        noisy_waypoints = self.training_scheduler.add_noise(normalized_waypoints, eps, t)

        # Predict noise
        noise_preds = self.model(
            image_encodings=normalized_embeddings,
            velocities=normalized_velocity,
            noisy_actions=noisy_waypoints,
            noise_timesteps=t.unsqueeze(1),
        )

        # Compute loss (MSE between predicted and actual noise)
        loss = nn.MSELoss()(noise_preds, eps)

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss detected: {loss.item()}")
            print(f"  noise_preds stats: min={noise_preds.min():.4f}, max={noise_preds.max():.4f}, mean={noise_preds.mean():.4f}")
            print(f"  eps stats: min={eps.min():.4f}, max={eps.max():.4f}, mean={eps.mean():.4f}")
            return {'loss': float('inf'), 'grad_norm': 0.0}

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Check for NaN gradients
        if not torch.isfinite(grad_norm):
            print(f"WARNING: Non-finite gradient norm detected!")
            return {'loss': loss.detach().item(), 'grad_norm': float('inf')}

        self.optimizer.step()

        return {
            'loss': loss.detach().item(),
            'grad_norm': grad_norm.item(),
        }

    def train(
        self,
        num_epochs,
        save_every=1000,
        save_dir="checkpoints/diffusion_policy",
        wandb_logging=False,
        wandb_project="VLAD",
        wandb_entity=None,
        wandb_run_name=None,
    ):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N steps
            save_dir: Directory to save checkpoints
            wandb_logging: Whether to log to wandb
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
            wandb_run_name: W&B run name (defaults to auto-generated)
        """
        if wandb_logging and wandb is not None:
            # Initialize wandb with config
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or "bench2drive_diffusion_policy",
                group="diffusion_policy",
                config={
                    **self.config,
                    'num_epochs': num_epochs,
                    'num_train_diffusion_timesteps': self.num_train_diffusion_timesteps,
                    'clip_sample_range': self.clip_sample_range,
                    'save_dir': save_dir,
                    'train_dataset_size': len(self.train_dataloader.dataset),
                    'val_dataset_size': len(self.val_dataloader.dataset) if self.val_dataloader else 0,
                    'device': str(self.device),
                }
            )

            # Watch model for gradient and parameter tracking
            wandb.watch(self.model, log='all', log_freq=100)

            # Log normalization statistics
            wandb.config.update({
                'embedding_mean': self.embedding_mean.mean().item(),
                'embedding_std': self.embedding_std.mean().item(),
                'velocity_mean': self.velocity_mean.item(),
                'velocity_std': self.velocity_std.item(),
                'waypoint_mean': self.waypoint_mean.mean().item(),
                'waypoint_std': self.waypoint_std.mean().item(),
            })

        os.makedirs(save_dir, exist_ok=True)

        global_step = 0
        losses = []

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training batches per epoch: {len(self.train_dataloader)}")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []
            epoch_grad_norms = []

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.training_step(batch)
                loss = metrics['loss']
                grad_norm = metrics['grad_norm']

                epoch_losses.append(loss)
                epoch_grad_norms.append(grad_norm)
                losses.append(loss)
                global_step += 1

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}', 'grad_norm': f'{grad_norm:.4f}'})

                # Log to wandb
                if wandb_logging and wandb is not None:
                    log_dict = {
                        "train/loss": loss,
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    }
                    wandb.log(log_dict, step=global_step)

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                    }, checkpoint_path)

                    # Save checkpoint as wandb artifact
                    if wandb_logging and wandb is not None:
                        artifact = wandb.Artifact(
                            name=f"model-step-{global_step}",
                            type="model",
                            description=f"Model checkpoint at step {global_step}",
                            metadata={
                                'step': global_step,
                                'epoch': epoch,
                                'loss': loss,
                            }
                        )
                        artifact.add_file(checkpoint_path)
                        wandb.log_artifact(artifact)

            # End of epoch
            epoch_avg_loss = np.mean(epoch_losses)
            epoch_avg_grad_norm = np.mean(epoch_grad_norms)
            print(f"Epoch {epoch+1} completed. Average loss: {epoch_avg_loss:.4f}, Average grad_norm: {epoch_avg_grad_norm:.4f}")

            # Run validation at end of each epoch
            epoch_val_loss = None
            if self.val_dataloader is not None:
                epoch_val_loss = self.evaluate()
                print(f"Epoch {epoch+1} validation loss: {epoch_val_loss:.4f}")
                self.model.train()

            # Log epoch-level metrics
            if wandb_logging and wandb is not None:
                log_dict = {
                    "epoch/avg_train_loss": epoch_avg_loss,
                    "epoch/avg_grad_norm": epoch_avg_grad_norm,
                    "epoch/number": epoch + 1,
                }
                if epoch_val_loss is not None:
                    log_dict["epoch/val_loss"] = epoch_val_loss
                wandb.log(log_dict, step=global_step)

        # Save final model
        final_path = os.path.join(save_dir, "final_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': num_epochs,
            'step': global_step,
        }, final_path)
        print(f"Training completed. Final model saved to {final_path}")

        # Save final model as wandb artifact
        if wandb_logging and wandb is not None:
            final_artifact = wandb.Artifact(
                name="model-final",
                type="model",
                description="Final trained model",
                metadata={
                    'total_steps': global_step,
                    'num_epochs': num_epochs,
                    'final_loss': losses[-1] if losses else None,
                }
            )
            final_artifact.add_file(final_path)
            wandb.log_artifact(final_artifact)

            # Create a summary table of training
            wandb.run.summary['final_train_loss'] = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            wandb.run.summary['total_steps'] = global_step
            wandb.run.summary['num_epochs'] = num_epochs

            wandb.finish()

        return losses

    def evaluate(self):
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move to device
                embeddings = batch['embeddings'].to(self.device)
                velocity = batch['speed'].to(self.device)
                waypoints = batch['waypoints'].to(self.device)

                batch_size = embeddings.shape[0]

                # Normalize
                normalized_embeddings, normalized_velocity = self.normalize_state(embeddings, velocity)
                normalized_waypoints = self.normalize_actions(waypoints)

                # Sample noise and timesteps (avoid t=0)
                eps = torch.randn_like(normalized_waypoints)
                t = torch.randint(
                    1, self.num_train_diffusion_timesteps, (batch_size,), device=self.device
                )

                # Add noise
                noisy_waypoints = self.training_scheduler.add_noise(normalized_waypoints, eps, t)

                # Predict
                noise_preds = self.model(
                    image_encodings=normalized_embeddings,
                    velocities=normalized_velocity,
                    noisy_actions=noisy_waypoints,
                    noise_timesteps=t.unsqueeze(1),
                )

                # Compute loss
                loss = nn.MSELoss()(noise_preds, eps)
                val_losses.append(loss.item())

        return np.mean(val_losses)


if __name__ == "__main__":
    # Training configuration
    data_root = "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_mini/extracted"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Diffusion Policy Training on Bench2Drive")
    print("=" * 70)
    print(f"Device: {device}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = Bench2DriveDataset(
        data_root=data_root,
        split="train",
        num_waypoints=10,
        img_augmentation=True,
    )

    val_dataset = Bench2DriveDataset(
        data_root=data_root,
        split="val",
        num_waypoints=10,
        img_augmentation=False,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = train_dataset.get_dataloader(
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Create model
    print("\nCreating model...")
    model = PolicyDiffusionTransformer(
        num_transformer_layers=4,
        image_encoding_dim=250 * 2560,  # 640,000
        act_dim=2,  # (x, y) per waypoint
        hidden_size=512,  # Smaller for faster training
        n_transformer_heads=8,
        device=device,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer with lower learning rate for stability
    lr = 3e-5
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Configuration dictionary for wandb
    train_config = {
        'model_name': 'PolicyDiffusionTransformer',
        'num_transformer_layers': 4,
        'image_encoding_dim': 250 * 2560,
        'act_dim': 2,
        'hidden_size': 512,
        'n_transformer_heads': 8,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'batch_size': 16,
        'num_workers': 4,
        'num_waypoints': 10,
        'img_augmentation': True,
        'data_root': data_root,
    }

    # Create trainer
    print("\nInitializing trainer...")
    trainer = TrainDiffusionPolicy(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        num_train_diffusion_timesteps=30,
        config=train_config,
    )

    # Train
    print("\nStarting training...")
    losses = trainer.train(
        num_epochs=10,
        save_every=500,
        save_dir="checkpoints/diffusion_policy",
        wandb_logging=True,  # Enable wandb logging
        wandb_project="VLAD",
        wandb_run_name="diffusion_policy_bench2drive",
    )

    print("\nTraining completed!")

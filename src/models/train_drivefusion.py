import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
print(str(Path(__file__).parent))
from drivefusion import DriveFusion
from dataloaders.bench2drive_history_dataset import (
    Bench2DriveHistoryDataset,
    bench2drive_collate_fn,
)
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import pickle
from collections import defaultdict
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import os


DEBUG = False
DEBUG_PATH = "src/models/debug_outputs"

def calculateStatictics(dataset: Bench2DriveHistoryDataset):
    all_vectors = defaultdict(lambda: [])

    for i in tqdm(dataset):
        for k in i.keys():
            all_vectors[k].append(i[k])

    all_stats = {}
    for k in all_vectors.keys():
        all_stats[k] = torch.std_mean(
            torch.stack(all_vectors[k]), axis=(0, 1) if k != "waypoints" else 0
        )
    return all_stats


def findStatistics(dataset: Bench2DriveHistoryDataset):
    if Path("stats.pkl").exists():
        cached_stats = pickle.load(open("stats.pkl", "rb"))
    else:
        cached_stats = {}
    dataset_clips = frozenset(dataset.clips)
    if dataset_clips in cached_stats.keys():
        return cached_stats[dataset_clips]
    else:
        all_stats = calculateStatictics(dataset)
        cached_stats[frozenset(dataset_clips)] = all_stats
        pickle.dump(cached_stats, open("stats.pkl", "wb"))
        return all_stats


class DriveFusionTrainer:
    def __init__(
        self,
        model: DriveFusion,
        device: "str",
        dataset: Bench2DriveHistoryDataset,
        batch_size: int,
        denoising_steps: int,
        val_dataset: Bench2DriveHistoryDataset,
    ) -> None:
        self.model = model
        self.device = device
        self.denoising_steps = denoising_steps
        self.model.to(self.device)
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=bench2drive_collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=bench2drive_collate_fn,
        )
        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.denoising_steps,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.denoising_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",  # variance is different for inference, see paper https://arxiv.org/pdf/2301.10677
        )
        self.inference_scheduler.alphas_cumprod = (
            self.inference_scheduler.alphas_cumprod.to(self.device)
        )
        self.stats = findStatistics(dataset)
        self.batch_size = batch_size
        self.denoising_steps = denoising_steps
        self.epoch = 1
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), 0.0001)

    def get_inference_timesteps(self):
        """
        gets the timesteps to use for inference
        """
        self.inference_scheduler.set_timesteps(self.denoising_steps, device=self.device)
        return self.inference_scheduler.timesteps

    def _train_step(self, batch):
        batch_size = batch["waypoints"].shape[0]
        normalized_batch = {}
        for k, _ in batch.items():
            if k == "history_ts":
                normalized_batch[k] = batch[k].to(self.device)
            else:
                normalized_batch[k] = ((batch[k] - self.stats[k][1]) / self.stats[k][0]).to(self.device)

        t = torch.randint(1, self.denoising_steps, (batch_size,), device=self.device)
        eps = torch.randn_like(normalized_batch["waypoints"])

        noises = self.train_noise_scheduler.add_noise(normalized_batch["waypoints"], eps, t)

        noise_pred, attn_weights = self.model(
            normalized_batch["history_image_embeddings"],
            normalized_batch["history_state"],
            noises,
            normalized_batch["history_ts"],
            t,
        )
        loss = self.loss_fn(noise_pred, eps)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), attn_weights

    def sample(self, batch):
        batch_size = batch["history_image_embeddings"].shape[0]
        normalized_batch = {}
        for k, v in batch.items():
            if k == "history_ts":
                normalized_batch[k] = batch[k].to(self.device)
            else:
                normalized_batch[k] = ((batch[k] - self.stats[k][1]) / self.stats[k][0]).to(self.device)

        xt = torch.randn(
            (batch_size, 10, 2),
            device=self.device,
            dtype=torch.float32,
        )
        timesteps = self.get_inference_timesteps()
        with torch.no_grad():
            for t in timesteps:
                noise_levels = torch.full(
                    (batch_size,), t, dtype=torch.long, device=self.device
                )
                noise_prediction = self.model(
                    normalized_batch["history_image_embeddings"],
                    normalized_batch["history_state"],
                    xt,
                    normalized_batch["history_ts"],
                    noise_levels,
                )
                step_output = self.inference_scheduler.step(noise_prediction, t, xt)
                xt = step_output.prev_sample
        return (xt * self.stats['waypoints'][0].cuda()) + self.stats['waypoints'][1].cuda()

    def evaluate_batch(self, batch, epoch):
        predicted_trajectory = self.sample(batch)
        actual_trajectory = batch['waypoints']
        os.makedirs(f"{DEBUG_PATH}/epoch_{epoch}", exist_ok=True)
        if DEBUG:
            # history images
            for i in range(batch['history_image_paths'].__len__()):
                plt.figure()
                for j in range(batch['history_image_paths'][i].__len__()):
                    img = plt.imread(batch['history_image_paths'][i][j])
                    # history of 8 frames
                    plt.subplot(2, 4, j + 1)
                    plt.imshow(img)
                    plt.axis('off')
                plt.suptitle("History Images")
                # Save image
                plt.savefig(f"{DEBUG_PATH}/epoch_{epoch}/history_images_{i}.png")
                plt.close()

            # visualize trajectories using matplotlib
            for i in range(predicted_trajectory.shape[0]):
                plt.figure()
                plt.plot(
                    actual_trajectory[i, :, 0].cpu(),
                    actual_trajectory[i, :, 1].cpu(),
                    label="Actual Trajectory",
                    marker='o'
                )
                plt.plot(
                    predicted_trajectory[i, :, 0].cpu(),
                    predicted_trajectory[i, :, 1].cpu(),
                    label="Predicted Trajectory",
                    marker='x'
                )
                plt.title("Trajectory Comparison")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.legend()
                plt.grid()

                # Save image
                plt.savefig(f"{DEBUG_PATH}/epoch_{epoch}/trajectory_comparison_{i}.png")
                plt.close()
            
        return self.loss_fn(predicted_trajectory, actual_trajectory.cuda()).item()

    def train(self, epochs):
        for i in range(epochs):
            loss_cumsum = torch.tensor(0, dtype=torch.float32)
            steps = 0
            self.model.train()
            for batch in tqdm(self.train_dataloader):
                loss, attn_weights = self._train_step(batch)
                loss_cumsum += loss
                steps += 1
            print(f"Epoch: {self.epoch}, Loss: {loss_cumsum / steps}")
            
            val_loss_cumsum = torch.tensor(0, dtype=torch.float32)
            val_steps = 0
            self.model.eval()
            for batch in tqdm(self.val_dataloader):
                loss = self.evaluate_batch(batch, i)
                val_loss_cumsum += loss
                val_steps += 1
            print(f"Epoch: {self.epoch}, Val Loss: {val_loss_cumsum / val_steps}")
            self.epoch += 1

if __name__ == "__main__":
    model = DriveFusion(6, 256, 78, 2048, 1024, 2, 512, 8, 4, 2048, 0.3, device="cuda")
    all_clips = [
        "AccidentTwoWays_Town12_Route1102_Weather10",
        "AccidentTwoWays_Town12_Route1103_Weather11",
        "AccidentTwoWays_Town12_Route1104_Weather12",
        "AccidentTwoWays_Town12_Route1105_Weather13",
        "AccidentTwoWays_Town12_Route1106_Weather14",
        "AccidentTwoWays_Town12_Route1107_Weather15",
        "AccidentTwoWays_Town12_Route1109_Weather9",
        "AccidentTwoWays_Town12_Route1110_Weather18",
        "AccidentTwoWays_Town12_Route1111_Weather19",
        "AccidentTwoWays_Town12_Route1112_Weather20",
        "AccidentTwoWays_Town12_Route1113_Weather21",
        "AccidentTwoWays_Town12_Route1114_Weather22",
        "AccidentTwoWays_Town12_Route1115_Weather23",
        "AccidentTwoWays_Town12_Route1116_Weather23",
        "AccidentTwoWays_Town12_Route1117_Weather25",
        "AccidentTwoWays_Town12_Route1119_Weather1",
        "AccidentTwoWays_Town12_Route1120_Weather2",
        "AccidentTwoWays_Town12_Route1121_Weather3",
        "AccidentTwoWays_Town12_Route1124_Weather18",
        "AccidentTwoWays_Town12_Route1126_Weather8",
        "AccidentTwoWays_Town12_Route1127_Weather9",
        "AccidentTwoWays_Town12_Route1444_Weather0",
        "AccidentTwoWays_Town12_Route1445_Weather1",
        "AccidentTwoWays_Town12_Route1446_Weather2",
        "AccidentTwoWays_Town12_Route1448_Weather5",
        "AccidentTwoWays_Town12_Route1453_Weather12",
        "AccidentTwoWays_Town12_Route1454_Weather13",
        "AccidentTwoWays_Town12_Route1455_Weather14",
        "AccidentTwoWays_Town12_Route1456_Weather15",
        "AccidentTwoWays_Town12_Route1458_Weather9",
        "AccidentTwoWays_Town12_Route1459_Weather18",
        "AccidentTwoWays_Town12_Route1461_Weather20",
        "AccidentTwoWays_Town12_Route1463_Weather22",
        "AccidentTwoWays_Town12_Route1468_Weather2",
        "AccidentTwoWays_Town12_Route1469_Weather3",
    ]
    train_dataset = Bench2DriveHistoryDataset(
        "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_base/extracted",
        "/ocean/projects/cis250252p/shared/afadia/VLAD/data/bench2drive_base/extracted",
        "_Qwen3-VL-2B-Instruct_features",
        8,
        all_clips[:16],
    )
    val_dataset = Bench2DriveHistoryDataset(
        "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_base/extracted",
        "/ocean/projects/cis250252p/shared/afadia/VLAD/data/bench2drive_base/extracted",
        "_Qwen3-VL-2B-Instruct_features",
        8,
        all_clips[-4:],
    )
    trainer = DriveFusionTrainer(model, "cuda", train_dataset, 32, 30, val_dataset)
    trainer.train(32)
    findStatistics(train_dataset)
    exit(0)
    for item in dataloader:
        scheduler = DDPMScheduler(
            30,
        )
        ops = model(
            item["history_image_embeddings"],
            item["history_state"],
            item["waypoints"],
            item["history_ts"],
        )
        print(ops.shape)

import torch
import torch.nn as nn
import numpy as np
import random

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class StudentDataset(Dataset):
    def __init__(self, responses, item_difficulty, mask):
        self.responses = responses
        self.difficulty = item_difficulty
        self.mask = mask

    def __getitem__(self, idx):
        x = self.responses[idx]  # [I]
        item_feat = self.difficulty  # [I]
        xw = torch.stack([x, item_feat], dim=1)  # [I, 2]
        return x, xw, self.mask[idx]

    def __len__(self):
        return len(self.responses)


class KnowledgeVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, q_matrix):
        super().__init__()
        self.register_buffer("q_matrix", q_matrix.clone().detach().float())  # [I, K]

        self.mlp_len1, self.mlp_len2, self.mlp_len3 = 32, 64, 128   # changeable
        self.item_feat_dim = input_dim

        self.per_item_encoder = nn.Sequential(
            nn.Linear(self.item_feat_dim, self.mlp_len1),
            nn.ReLU(),
            nn.Linear(self.mlp_len1, self.mlp_len2),
            nn.ReLU()
        )

        self.mean_z = nn.Sequential(
            nn.Linear(self.mlp_len2, self.mlp_len3),
            nn.ReLU(),
            nn.Linear(self.mlp_len3, latent_dim),
        )
        self.log_var_z = nn.Sequential(
            nn.Linear(self.mlp_len2, self.mlp_len3),
            nn.ReLU(),
            nn.Linear(self.mlp_len3, latent_dim),
        )

        self.mlp_transform = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encoder(self, xw):  # xw: [B, I, 2]
        h = self.per_item_encoder(xw)  # [B, I, 64]
        h_mean = h.mean(dim=1)  # [B, 64]
        return self.mean_z(h_mean), self.log_var_z(h_mean)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        z_transformed = self.mlp_transform(z)  # [B, K]
        return torch.sigmoid(torch.matmul(z_transformed, self.q_matrix.T))  # [B, I]

    def forward(self, xw):
        mean_z, log_var_z = self.encoder(xw)
        z = self.reparameterize(mean_z, log_var_z)  # [B, K]
        recon = self.decoder(z)
        return recon, mean_z, log_var_z, z


def loss_function(recon_x, x, mu, log_var, mask=None, beta=0.1, tau=0.1):
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='none')  # [B, I]
    if mask is not None:
        bce = bce * mask
        recon_loss = bce.sum() / mask.sum()
    else:
        recon_loss = bce.mean()

    per_dim_kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # [B, K]
    clipped_kl = torch.clamp(per_dim_kl, min=tau)
    kld = clipped_kl.sum(dim=1).mean()

    mse = ((recon_x - x) ** 2) * mask
    rmse = torch.sqrt((mse.sum() / mask.sum()).clamp(min=1e-8))

    return recon_loss + beta * kld, rmse


class ivae_train:
    def __init__(self, exer_n, knowledge_n, device, batch_size=128, epochs=100, lr=0.0001, weight_decay=0,
                 beta=0.1, beta_max=1, tau=0.1, early_stop_delta=1e-4, early_stop_patience=10, warmup_epochs=10):
        self.exer_n = exer_n
        self.n_knowledge = knowledge_n
        self.user_con_dim = knowledge_n
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.beta_max = beta_max
        self.tau = tau
        self.warmup_epochs = warmup_epochs
        self.device = device

        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta

    def train_eval(self, train_matrix, train_mask, val_matrix, val_mask, item_diff, q_matrix):
        train_data = torch.tensor(train_matrix, dtype=torch.float32).to(self.device)
        train_mask = torch.tensor(train_mask, dtype=torch.float32).to(self.device)
        val_data = torch.tensor(val_matrix, dtype=torch.float32).to(self.device)
        val_mask = torch.tensor(val_mask, dtype=torch.float32).to(self.device)
        item_diff = torch.tensor(item_diff, dtype=torch.float32).to(self.device)
        q_matrix = torch.tensor(q_matrix, dtype=torch.float32)

        train_dataset = StudentDataset(responses=train_data, item_difficulty=item_diff, mask=train_mask)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        model = KnowledgeVAE(input_dim=2, latent_dim=self.user_con_dim, q_matrix=q_matrix).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_rmse = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            beta = min(self.beta_max, self.beta_max * epoch / self.warmup_epochs)
            for x, xw, mask in train_dataloader:
                x = x.to(self.device)
                xw = xw.to(self.device)
                mask = mask.to(self.device)

                recon, mu, log_var, z = model(xw)
                loss, _ = loss_function(recon, x, mu, log_var, mask, beta, tau=self.tau)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                item_diff_exp = item_diff.unsqueeze(0).expand(val_data.size(0), -1)
                val_xw = torch.cat([val_data.unsqueeze(-1), item_diff_exp.unsqueeze(-1)], dim=2)

                val_x = val_data
                val_xw = val_xw.to(self.device)
                val_mask = val_mask.to(self.device)

                recon, mu, log_var, z = model(val_xw)
                val_loss, val_rmse = loss_function(recon, val_x, mu, log_var, val_mask, beta, tau=self.tau)

                print(f"Epoch {epoch}: Val rmse = {val_rmse.item():.4f}")

                if val_rmse.item() < best_rmse - self.early_stop_delta:
                    best_rmse = val_rmse.item()
                    torch.save(model.state_dict(), "best_knowledge_vae.pt")
                    print(f"Model saved at Epoch {epoch} with RMSE {best_rmse:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement. Early Stop Counter: {patience_counter}/{self.early_stop_patience}")

                if patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best RMSE: {best_rmse:.4f}")
                    break

    def save_vae_params(self, full_matrix, full_mask, item_diff, q_matrix,
                        batch_size=256, save_path=None):
        full_matrix = torch.tensor(full_matrix, dtype=torch.float32)
        full_mask = torch.tensor(full_mask, dtype=torch.float32)
        item_diff = torch.tensor(item_diff, dtype=torch.float32)

        dataset = StudentDataset(full_matrix, item_diff, full_mask)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        state_dict = torch.load("best_knowledge_vae.pt", map_location=self.device)
        model = KnowledgeVAE(input_dim=2, latent_dim=self.user_con_dim,
                             q_matrix=q_matrix).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        all_z = []
        all_mu = []
        all_logvar = []

        with torch.no_grad():
            for x, xw, mask in loader:
                xw = xw.to(self.device)
                _, mu, log_var, z = model(xw)
                all_z.append(z.cpu())
                all_mu.append(mu.cpu())
                all_logvar.append(log_var.cpu())

        mu_all = torch.cat(all_mu, dim=0)  # [N, K]
        logvar_all = torch.cat(all_logvar, dim=0)  # [N, K]

        torch.save(mu_all.detach(), "mean.pt")
        torch.save(logvar_all.detach().exp().sqrt(), "std.pt")


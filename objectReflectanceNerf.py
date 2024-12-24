import torch
import torch.nn as nn


class ObjectReflectanceNeRF(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        # Poistion, light, and view direction encoder
        self.encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),  # position (3) + view_dir (3) + light_dir (3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Color modifier network
        self.color_network = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, positions, view_dirs, light_dirs, base_colors):
        features = self.encoder(torch.cat([positions, view_dirs, light_dirs], dim=1))
        combined = torch.cat([features, base_colors], dim=1)

        modified_colour = self.color_network(combined)

        return modified_colour

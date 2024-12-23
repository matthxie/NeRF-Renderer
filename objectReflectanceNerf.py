import torch
import torch.nn as nn


class ObjectReflectanceNeRF(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        # Surface geometry encoder
        self.geometry_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),  # position (3) + normal (3) + surface_id (3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Light and view direction encoder
        self.direction_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # view_dir (3) + light_dir (3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Color modifier network
        self.color_network = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + 3, hidden_dim
            ),  # geometry + directions + base_color
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(
        self, positions, normals, surface_ids, view_dirs, light_dirs, base_colors
    ):
        geometry_features = self.geometry_encoder(
            torch.cat([positions, normals, surface_ids], dim=-1)
        )

        direction_features = self.direction_encoder(
            torch.cat([view_dirs, light_dirs], dim=-1)
        )

        combined = torch.cat(
            [geometry_features, direction_features, base_colors], dim=-1
        )
        modified_colors = self.color_network(combined)

        return modified_colors

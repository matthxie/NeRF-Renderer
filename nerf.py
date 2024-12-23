import torch
import torch.nn as nn


class ReflectanceNeRF(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            # Input features: position (3), view direction (3), light direction (3), base color (3)
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output: reflectance modification factors (3) for RGB
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, positions, view_dirs, light_dirs, base_colors):
        # Normalize input vectors
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        light_dirs = light_dirs / torch.norm(light_dirs, dim=-1, keepdim=True)

        x = torch.cat([positions, view_dirs, light_dirs, base_colors], dim=-1)

        # Get reflectance factors
        reflectance = self.network(x)

        # Modify base colors with learned reflectance
        modified_colors = base_colors * reflectance

        return modified_colors


def train_reflectance_nerf(model, train_data, num_epochs=400, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_data:
            positions, view_dirs, light_dirs, base_colors, target_colors = batch

            predicted_colors = model(positions, view_dirs, light_dirs, base_colors)
            loss = loss_fn(predicted_colors, target_colors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_data):.4f}"
            )


def render_with_nerf(model, position, view_dir, light_dir, base_color):
    """
    Render a single point using the trained NeRF model
    """
    with torch.no_grad():
        position = torch.FloatTensor(position).unsqueeze(0)
        view_dir = torch.FloatTensor(view_dir).unsqueeze(0)
        light_dir = torch.FloatTensor(light_dir).unsqueeze(0)
        base_color = torch.FloatTensor(base_color).unsqueeze(0)

        modified_color = model(position, view_dir, light_dir, base_color)
        return modified_color.squeeze(0).numpy()


def main():
    model = ReflectanceNeRF()

    # Generate synthetic training data
    num_samples = 1000
    positions = torch.rand(num_samples, 3)
    view_dirs = torch.rand(num_samples, 3)
    light_dirs = torch.rand(num_samples, 3)
    base_colors = torch.rand(num_samples, 3)
    target_colors = torch.rand(num_samples, 3)

    train_data = [(positions, view_dirs, light_dirs, base_colors, target_colors)]

    # Train the model
    train_reflectance_nerf(model, train_data)

    # Test rendering
    test_position = [0.5, 0.5, 0.5]
    test_view_dir = [0.0, 0.0, 1.0]
    test_light_dir = [1.0, 1.0, 1.0]
    test_base_color = [0.8, 0.2, 0.2]

    modified_color = render_with_nerf(
        model, test_position, test_view_dir, test_light_dir, test_base_color
    )
    print(f"Modified color: {modified_color}")


if __name__ == "__main__":
    main()

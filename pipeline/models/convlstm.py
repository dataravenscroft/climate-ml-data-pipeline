import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    Replaces fully-connected gates in standard LSTM with convolutions,
    preserving spatial structure across timesteps. This is why ConvLSTM
    outperforms standard LSTM for gridded atmospheric data: spatial
    correlations (pressure gradients, jet streams) are local and shift
    over time — exactly what conv gates capture.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # All four gates (input, forget, output, cell) computed in one conv
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, C_in, H, W)      — input at current timestep
            h: (B, C_hidden, H, W)  — previous hidden state
            c: (B, C_hidden, H, W)  — previous cell state
        Returns:
            h_new, c_new
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )


class ConvLSTMForecast(nn.Module):
    """Stacked ConvLSTM encoder → 1-step forecast head.

    Architecture:
        Input: (B, T, C_in, H, W) — T input frames
        → 2 ConvLSTM layers (encoder)
        → Final hidden state → Conv2d projection → (B, C_out, H, W)

    For ERA5: C_in = number of atmospheric variables (e.g. 4: Z500, T850, U10, V10)
              C_out = same (predicting same fields one step ahead)
              H, W = spatial grid (e.g. 32×64 for 5.625° resolution)
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 2,
        out_channels: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        cells = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(c_in, hidden_channels))
        self.cells = nn.ModuleList(cells)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            forecast: (B, C_out, H, W) — predicted next frame
        """
        B, T, C, H, W = x.shape
        device = x.device

        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        for t in range(T):
            inp = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]
                h, c = cell(inp, h, c)
                states[layer_idx] = (h, c)
                inp = h

        return self.head(states[-1][0])

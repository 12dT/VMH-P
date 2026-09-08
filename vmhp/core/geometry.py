import torch


EPS = 1e-6


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def project_tangent_norm(z: torch.Tensor, max_norm: float = 3.0, eps: float = EPS) -> torch.Tensor:
    if max_norm is None or max_norm <= 0:
        return z
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return z * scale


def expmap0(z: torch.Tensor, c: float = 0.25, max_norm: float = 3.0) -> torch.Tensor:
    z = project_tangent_norm(z, max_norm=max_norm)
    c_tensor = torch.as_tensor(c, device=z.device, dtype=z.dtype).clamp_min(EPS)
    sqrt_c = torch.sqrt(c_tensor)
    norm = z.norm(dim=-1, keepdim=True).clamp_min(EPS)
    scaled_norm = sqrt_c * norm
    time = torch.cosh(scaled_norm) / sqrt_c
    coef = torch.sinh(scaled_norm) / (sqrt_c * norm)
    space = coef * z
    return torch.cat([time, space], dim=-1)


def logmap0(x: torch.Tensor, c: float = 0.25) -> torch.Tensor:
    c_tensor = torch.as_tensor(c, device=x.device, dtype=x.dtype).clamp_min(EPS)
    sqrt_c = torch.sqrt(c_tensor)
    time = (sqrt_c * x[..., :1]).clamp_min(1.0 + EPS)
    dist0 = torch.acosh(time) / sqrt_c
    space = x[..., 1:]
    space_norm = space.norm(dim=-1, keepdim=True).clamp_min(EPS)
    direction = space / space_norm
    return direction * dist0


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float = 0.25) -> torch.Tensor:
    c_tensor = torch.as_tensor(c, device=x.device, dtype=x.dtype).clamp_min(EPS)
    inner = lorentz_inner(x, y).squeeze(-1)
    z = (-c_tensor * inner).clamp_min(1.0 + EPS)
    return torch.acosh(z) / torch.sqrt(c_tensor)


def pairwise_lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float = 0.25) -> torch.Tensor:
    c_tensor = torch.as_tensor(c, device=x.device, dtype=x.dtype).clamp_min(EPS)
    inner = -x[:, :1] * y[:, :1].transpose(0, 1) + x[:, 1:] @ y[:, 1:].transpose(0, 1)
    z = (-c_tensor * inner).clamp_min(1.0 + EPS)
    return torch.acosh(z) / torch.sqrt(c_tensor)


class LorentzGeometry:
    def __init__(self, curvature: float = 0.25, max_norm: float = 3.0):
        self.curvature = float(curvature)
        self.max_norm = float(max_norm)

    def project_tangent(self, z: torch.Tensor) -> torch.Tensor:
        return project_tangent_norm(z, self.max_norm)

    def exp_origin(self, z: torch.Tensor) -> torch.Tensor:
        return expmap0(z, self.curvature, self.max_norm)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return lorentz_distance(x, y, self.curvature)

    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return pairwise_lorentz_distance(x, y, self.curvature)

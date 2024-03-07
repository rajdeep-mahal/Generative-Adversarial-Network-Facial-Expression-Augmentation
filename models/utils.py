import torch


def kp2gaussian(kp, spatial_size, kp_variance):

    mean = kp['value']

    coordinate_grid = create_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)
    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def create_coordinate_grid(spatial_size, type):
    """
    Creates a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    mesh_grid = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return mesh_grid


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class Transform:
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = create_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = create_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return torch.nn.functional.grid_sample(frame, grid, padding_mode="reflection", align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = torch.autograd.grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = torch.autograd.grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian

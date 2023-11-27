"""Create weight functions for charge and electric field."""  # FIXME add classes to comment.


class ChargeWeightFunc:
    def __init__(self, distances, spatial_grid_delta):
        self.distances = distances
        self.spatial_grid_delta = spatial_grid_delta

    def zeroth_order(self):
        return (
            1
            / self.spatial_grid_delta
            * (self.distances < self.spatial_grid_delta / 2).astype(int)
        )

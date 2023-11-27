"""Create weight functions for charge and electric field."""  # FIXME add classes to comment.


# class ChargeWeightFunc:
#     def __init__(self, evalutation_point, grid_point, spatial_grid_delta):
#         self.evalutation_point = evalutation_point
#         self.grid_point = grid_point
#         self.spatial_grid_delta = spatial_grid_delta

#     def zeroth_order(self):
#         if (
#             np.abs(self.grid_point - self.evalutation_point)
#             < self.spatial_grid_delta / 2
#         ):
#             return 1 / self.spatial_grid_delta
#         return 0


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

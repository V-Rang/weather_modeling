import torch.nn as nn
import numpy as np
import torch

class BilinearInterpolation(nn.Module):
    def __init__(self, output_size):
        super(BilinearInterpolation, self).__init__()
        self.output_size = output_size
        
    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][1]
        return (None, num_channels, height, width)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def forward(self, tensors):
        X, transformation  = tensors
        output = self._transform(X, transformation, self.output_size)
        return output
    
    def _interpolate(self, image, sampled_grids, output_size):
        batch_size = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]
        num_channels = image.shape[1]
        x = sampled_grids[:, 0:1, :].flatten().float()
        y = sampled_grids[:,1:2,:].flatten().float()

        x = (.5 * (x + 1.0) * width).float()
        y = (.5 * (y + 1.0) * height).float()

        x0 = x.int()
        x1 = x0 + 1
        y0 = y.int()        
        y1 = y0 + 1

        max_x = int(image.shape[3] - 1)
        max_y = int(image.shape[2] - 1)

        x0 = torch.clip(x0, 0, max_x)
        x1 = torch.clip(x1, 0, max_x)
        y0 = torch.clip(y0, 0, max_y)
        y1 = torch.clip(y1, 0, max_y)
        
        pixels_batch = torch.arange(0, batch_size) * (height * width)
        pixels_batch = pixels_batch.reshape(-1,1)
        
        flat_output_size = output_size[0] * output_size[1]

        base = pixels_batch.unsqueeze(1)
        base = base.repeat(1, flat_output_size, 1)
        base = base.flatten()
        base_y0 = y0 * width

        base_y0 = base + base_y0
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1
                
        flat_image = image.reshape(shape=(-1, num_channels))
        flat_image = flat_image.float()

        pixel_values_a = torch.take(flat_image, indices_a).reshape(-1,1)
        pixel_values_b = torch.take(flat_image, indices_b).reshape(-1,1)
        pixel_values_c = torch.take(flat_image, indices_c).reshape(-1,1)
        pixel_values_d = torch.take(flat_image, indices_d).reshape(-1,1)
        
        x0 = x0.float()
        x1 = x1.float()
        y0 = y0.float()
        y1 = y1.float()

        area_a = ((x1 - x) * (y1 - y)).unsqueeze_(1)
        area_b = ((x1 - x) * (y - y0)).unsqueeze_(1)
        area_c = ((x - x0) * (y1 - y)).unsqueeze_(1)
        area_d = ((x - x0) * (y - y0)).unsqueeze_(1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    
    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = torch.linspace(-1., 1., width)
        y_linspace = torch.linspace(-1., 1., height)
        x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = y_coordinates.flatten()        
        ones = torch.ones((x_coordinates.shape))
        grid = torch.cat( (x_coordinates, y_coordinates, ones), dim=0)
        grid = grid.flatten()
        grids = torch.tile(grid, (1,batch_size)).flatten()
        return grids.reshape( (batch_size, 3, height * width)  )
    
    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = X.shape[0], X.shape[1]
        transformations = affine_transformation.reshape( shape=(batch_size, 2, 3))
        # regular_grids = torch.tensor(self._make_regular_grids(batch_size, *output_size)).float()
        regular_grids = self._make_regular_grids(batch_size, *output_size).clone().detach().float()
        sampled_grids = torch.bmm(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        # print(interpolated_image.shape)
        new_shape = (batch_size, num_channels, output_size[0], output_size[1])
        interpolated_image = interpolated_image.reshape(new_shape)
        return interpolated_image

        
# Native MLX port of mlx_vlm/models/mistral3/mistral3.py
# Reference e9af376626b04ba0cdad23b1476fd7a4a583d721
# See ../UPSTREAM_LICENSE.
import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def unfold(
    input: mx.array,
    kernel_size: int | tuple[int, int] | list[int],
    dilation: int | tuple[int, int] | list[int] = 1,
    padding: int | tuple[int, int] | list[int] = 0,
    stride: int | tuple[int, int] | list[int] = 1,
) -> mx.array:
    """
    Extract sliding local blocks from a batched input tensor (MLX implementation).

    This is equivalent to PyTorch's nn.functional.unfold or im2col operation.

    Args:
        input: Input tensor of shape (B, C, H, W)
        kernel_size: Size of the sliding blocks
        dilation: Controls the spacing between kernel elements
        padding: Controls the amount of implicit padding
        stride: Controls the stride between blocks

    Returns:
        Unfolded tensor of shape (B, C*kernel_height*kernel_width, L)
        where L is the number of blocks
    """
    # Convert to pairs
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    stride = _pair(stride)

    # Input shape
    batch_size, channels, height, width = input.shape

    # Add padding if needed
    if padding[0] > 0 or padding[1] > 0:
        padding_shape = (
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1]),
        )
        input = mx.pad(input, padding_shape)

    # Calculate output dimensions
    height_out = (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    width_out = (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

    # Initialize output arrays
    blocks = []

    # Extract blocks
    for i in range(0, height + 2 * padding[0] - kernel_size[0] * dilation[0] + 1, stride[0]):
        for j in range(0, width + 2 * padding[1] - kernel_size[1] * dilation[1] + 1, stride[1]):
            # Extract the block for all channels
            block = []
            for di in range(kernel_size[0]):
                for dj in range(kernel_size[1]):
                    h_idx = i + di * dilation[0]
                    w_idx = j + dj * dilation[1]
                    # Get the block for all channels and add to our list
                    block.append(input[:, :, h_idx, w_idx])

            # Stack the channel-blocks
            block = mx.stack(block, axis=1)  # Shape: (B, k*k, C)
            block = mx.transpose(block, [0, 2, 1])  # Shape: (B, C, k*k)
            blocks.append(block)

    # Stack all blocks together
    result = mx.stack(blocks, axis=-1)  # Shape: (B, C, k*k, L)

    # Reshape to match PyTorch's unfold output format: (B, C*k*k, L)
    result = mx.reshape(
        result,
        (
            batch_size,
            channels * kernel_size[0] * kernel_size[1],
            height_out * width_out,
        ),
    )

    return result


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def __call__(self, image_features: mx.array, image_sizes: mx.array) -> mx.array:

        image_sizes = [
            (
                int(image_size[0].item() if hasattr(image_size[0], "item") else image_size[0])
                // self.patch_size,
                int(image_size[1].item() if hasattr(image_size[1], "item") else image_size[1])
                // self.patch_size,
            )
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)
        if image_features.ndim != 2:
            raise ValueError(
                f"Expected image_features to be 2D [tokens, dim], got shape {image_features.shape}"
            )

        # Split the image features into chunks based on tokens_per_image
        split_indices = []
        current_index = 0
        for tokens in tokens_per_image:
            split_indices.append(current_index + tokens)
            current_index += tokens

        # Perform the split
        chunks = mx.split(image_features, split_indices[:-1], axis=0)

        permuted_tensor = []
        for image_index, image_tokens in enumerate(chunks):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.reshape(h, w, d).transpose(2, 0, 1)[None, ...]
            grid = unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            grid = grid.reshape(d * self.spatial_merge_size**2, -1).T
            permuted_tensor.append(grid)

        image_features = mx.concatenate(permuted_tensor, axis=0)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm = nn.RMSNorm(
            config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.patch_merger = Mistral3PatchMerger(config)

        num_feature_layers = (
            1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def __call__(self, x: mx.array, image_sizes: mx.array) -> mx.array:
        x = self.norm(x)

        x = self.patch_merger(x, image_sizes)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x

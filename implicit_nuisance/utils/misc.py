


from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

def dictmap(func, d):
    """Applies a function to each value in a dictionary."""
    return {k: func(v) for k, v in d.items() if not v is None}

def dictmap_keys(func, d):
    """Applies a function to each key in a dictionary."""
    return {func(k): v for k, v in d.items()}

def omegaconf_select(cfg, key, default=None):
    """Wrapper for OmegaConf.select to allow None to be returned instead of 'None'."""
    value = OmegaConf.select(cfg, key, default=default)
    if value == "None":
        return None
    return value


def get_padded_view(tensor, start_indices, view_shape):
  """
  Extracts a view of a tensor, padding with zeros if the view
  goes out of bounds.

  Args:
    tensor (torch.Tensor): The input tensor.
    start_indices (tuple or list): The starting indices (inclusive) for
                                    the view in each dimension.
    view_shape (tuple or list): The desired shape of the output view.

  Returns:
    torch.Tensor: The extracted view, padded with zeros where necessary.
  """
  if len(start_indices) != tensor.ndim or len(view_shape) != tensor.ndim:
    raise ValueError("start_indices and view_shape must have the same "
                     "length as the tensor's number of dimensions.")

  # Calculate the end indices (exclusive) for the desired view
  end_indices = [s + v for s, v in zip(start_indices, view_shape)]

  # Calculate padding amounts needed for each side of each dimension
  # F.pad requires padding in reverse dimension order: (pad_left, pad_right, pad_top, pad_bottom, ...)
  padding_needed = []
  for i in range(tensor.ndim - 1, -1, -1):
    start = start_indices[i]
    end = end_indices[i]
    dim_size = tensor.shape[i]

    pad_before = max(0, -start)
    pad_after = max(0, end - dim_size)
    padding_needed.extend([pad_before, pad_after])

  # Pad the tensor
  padded_tensor = F.pad(tensor, pad=padding_needed, mode='constant', value=0)

  # Calculate the slice indices within the padded tensor
  # The start index in the padded tensor is the original start index + pad_before
  padded_start_indices = [s + max(0, -s) for s in start_indices]
  padded_end_indices = [ps + v for ps, v in zip(padded_start_indices, view_shape)]

  # Create slice objects
  slices = tuple(slice(start, end) for start, end in zip(padded_start_indices, padded_end_indices))

  # Extract the view from the padded tensor
  result_view = padded_tensor[slices]

  return result_view


import torch
import torch.nn as nn
from tqdm import tqdm

from implicit_nuisance.utils.misc import dictmap

def get_states_and_decodings(
    device: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    """Collects images and features from the model.
    """

    data = []
    labels = []
    states = []
    decodings = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            _to_device = lambda x: x.to(device, non_blocking=True)
            _cpu = lambda x: x.cpu()
            x = _to_device(x)
            y = dictmap(_to_device, y)

            data.append(x.cpu())
            labels.append(dictmap(_cpu, y))
            out = model(x, y["actions"]) if "actions" in y else model(x)
            state = out["latents"]
            decoding = out["target_estimates"]
            states.append(dictmap(_cpu, state))
            decodings.append(dictmap(_cpu, decoding))
    model.train()

    return data, labels, states, decodings

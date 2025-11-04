import torch

from models.radio_unet_tx.unet import TxUNet
from models.restormer_port.wrapper import RestormerRadio


@torch.no_grad()
def test_txunet_forward_shape():
    model = TxUNet(in_ch=3, out_ch=1, base_ch=16, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1))
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert tuple(y.shape) == (1, 1, 64, 64)


@torch.no_grad()
def test_restormer_forward_shape():
    model = RestormerRadio(in_ch=3, out_ch=1, base_ch=16, heads=(1, 1, 1, 1), refinement_blocks=1)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert tuple(y.shape) == (1, 1, 64, 64)





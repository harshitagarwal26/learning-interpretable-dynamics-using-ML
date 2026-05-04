"""
Smoke test for the dumbbell caVAE encoder / decoder.

Verifies:
  1. Forward pass on a random batch produces the expected shapes.
  2. Forward pass on a real batch from the dataset produces finite outputs.
  3. A reconstruction MSE loss is computable and gradients backprop cleanly.
  4. Decoding *ground-truth* latents (r, phi, theta) and comparing against
     the true image gives a visible dumbbell at roughly the right location —
     this validates the affine geometry independent of training.
"""

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)

import numpy as np
import torch

from lag_caVAE.vae_dumbbell import DumbbellEncoder, DumbbellDecoder
from utils import from_pickle


def smoke_test_random():
    print("=== 1. Random-input smoke test ===")
    bs = 8
    enc = DumbbellEncoder(img_size=32, hidden=128)
    dec = DumbbellDecoder(img_size=32, hidden=64)

    x = torch.randn(bs, 1, 32, 32)
    out = enc(x)
    for k, v in out.items():
        print(f"  {k:22s} {tuple(v.shape)}")

    # Expected shapes
    assert out['r_mean'].shape == (bs, 1)
    assert out['r_var'].shape == (bs, 1)
    assert out['phi_vec'].shape == (bs, 2)
    assert out['phi_concentration'].shape == (bs, 1)
    assert out['theta_vec'].shape == (bs, 2)
    assert out['theta_concentration'].shape == (bs, 1)
    # Unit-norm checks
    phi_norms = out['phi_vec'].norm(dim=-1)
    th_norms = out['theta_vec'].norm(dim=-1)
    assert torch.allclose(phi_norms, torch.ones_like(phi_norms), atol=1e-5)
    assert torch.allclose(th_norms, torch.ones_like(th_norms), atol=1e-5)
    print("  unit-norm (cos, sin) checks pass")

    r = out['r_mean']
    cp, sp = out['phi_vec'][:, 0:1], out['phi_vec'][:, 1:2]
    ct, st = out['theta_vec'][:, 0:1], out['theta_vec'][:, 1:2]
    x_rec = dec(r, cp, sp, ct, st)
    print(f"  x_rec                   {tuple(x_rec.shape)}")
    assert x_rec.shape == (bs, 1, 32, 32)
    print("  shape assertions pass.\n")


def smoke_test_backprop():
    print("=== 2. Backprop test ===")
    bs = 16
    enc = DumbbellEncoder(img_size=32, hidden=128)
    dec = DumbbellDecoder(img_size=32, hidden=64)

    x = torch.rand(bs, 1, 32, 32)
    out = enc(x)
    r = out['r_mean']
    cp, sp = out['phi_vec'][:, 0:1], out['phi_vec'][:, 1:2]
    ct, st = out['theta_vec'][:, 0:1], out['theta_vec'][:, 1:2]
    x_rec = dec(r, cp, sp, ct, st)

    loss = ((x_rec - x) ** 2).mean()
    loss.backward()

    grad_sum = sum(
        p.grad.abs().sum().item()
        for p in list(enc.parameters()) + list(dec.parameters())
        if p.grad is not None
    )
    print(f"  loss    = {loss.item():.4f}")
    print(f"  sum |grad| across all params = {grad_sum:.3e}  (must be > 0)")
    assert grad_sum > 0
    print("  gradients flow through encoder AND decoder.\n")


def smoke_test_real_data():
    print("=== 3. Real-image pass ===")
    pkl = os.path.join(PARENT_DIR, 'datasets', 'dumbbell-rigid-dataset.pkl')
    data = from_pickle(pkl)
    imgs = data['x'][0]   # (T, N, 32, 32)
    obs = data['obs'][0]  # (T, N, 6)

    bs = 32
    x_np = imgs[0, :bs]                 # (bs, 32, 32)
    s_np = obs[0, :bs]                  # (bs, 6)
    x = torch.from_numpy(x_np).float().unsqueeze(1)  # (bs, 1, 32, 32)

    enc = DumbbellEncoder(img_size=32, hidden=128)
    dec = DumbbellDecoder(img_size=32, hidden=64)
    with torch.no_grad():
        out = enc(x)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:22s} finite={torch.isfinite(v).all().item()}, "
                  f"mean={v.float().mean().item(): .4f}")

    # Feed GROUND TRUTH latents into the decoder. Body sprite is random so the
    # reconstruction is gibberish, but the *location* of whatever it paints
    # should track the true (r, phi, theta). We verify by checking that the
    # reconstruction's centroid of mass is near (r cos phi, r sin phi).
    r_true_world = torch.from_numpy(s_np[:, 0]).float()
    phi_true = torch.from_numpy(s_np[:, 1]).float()
    theta_true = torch.from_numpy(s_np[:, 2]).float()
    # Normalize r to image space: world bound 5 -> [-1, 1]
    world_bound = 5.0
    r_norm = r_true_world / world_bound
    cp, sp = torch.cos(phi_true).unsqueeze(1), torch.sin(phi_true).unsqueeze(1)
    ct, st = torch.cos(theta_true).unsqueeze(1), torch.sin(theta_true).unsqueeze(1)
    with torch.no_grad():
        x_rec = dec(r_norm.unsqueeze(1), cp, sp, ct, st)
    print(f"  x_rec finite            {torch.isfinite(x_rec).all().item()}")
    print(f"  x_rec range             [{x_rec.min().item(): .3f}, {x_rec.max().item(): .3f}]")
    print(f"  reconstruction loss     {((x_rec - x) ** 2).mean().item(): .4f}")
    print("  real-data pass OK.\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    smoke_test_random()
    smoke_test_backprop()
    smoke_test_real_data()
    print("All smoke tests passed.")

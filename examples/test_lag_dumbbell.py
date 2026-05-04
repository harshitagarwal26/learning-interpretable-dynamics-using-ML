"""
Smoke test for Lag_Net_Dumbbell.
Verifies forward pass produces an 8+u_dim derivative, shapes are right,
and torchdiffeq can integrate it for a few steps.
"""

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)

import torch
from torchdiffeq import odeint

from lag_caVAE.lag_dumbbell import Lag_Net_Dumbbell
from lag_caVAE.nn_models import MLP, PSD, MatrixNet


def build():
    V_net = MLP(3, 100, 1)
    M_net = PSD(3, 300, 3)
    g_net = MatrixNet(3, 100, 3, shape=(3, 1))
    return Lag_Net_Dumbbell(g_net=g_net, M_net=M_net, V_net=V_net, u_dim=1)


def test_forward():
    print("=== 1. Forward pass ===")
    net = build()
    bs = 4
    r        = torch.rand(bs, 1) * 0.5 + 0.5       # 0.5..1.0 in normalized coords
    cp, sp   = torch.cos(torch.rand(bs, 1)), torch.sin(torch.rand(bs, 1))
    ct, st   = torch.cos(torch.rand(bs, 1)), torch.sin(torch.rand(bs, 1))
    r_dot    = torch.randn(bs, 1) * 0.05
    phi_dot  = torch.randn(bs, 1) * 0.5
    th_dot   = torch.randn(bs, 1) * 0.3
    u        = torch.zeros(bs, 1)
    x = torch.cat([r, cp, sp, ct, st, r_dot, phi_dot, th_dot, u], dim=1)
    x.requires_grad_(True)
    t = torch.tensor(0.0)
    dx = net(t, x)
    print(f"  x.shape  {tuple(x.shape)}")
    print(f"  dx.shape {tuple(dx.shape)}")
    assert dx.shape == x.shape, f"expected {x.shape}, got {dx.shape}"
    assert torch.isfinite(dx).all(), "non-finite outputs"
    print("  forward OK, all finite.\n")


def test_odeint():
    print("=== 2. torchdiffeq rollout ===")
    net = build()
    bs = 4
    x0 = torch.cat([
        torch.rand(bs, 1) * 0.3 + 0.5,
        torch.cos(torch.rand(bs, 1)), torch.sin(torch.rand(bs, 1)),
        torch.cos(torch.rand(bs, 1)), torch.sin(torch.rand(bs, 1)),
        torch.zeros(bs, 3),
        torch.zeros(bs, 1),
    ], dim=1)
    x0.requires_grad_(True)
    t_eval = torch.linspace(0.0, 0.2, 5)
    traj = odeint(net, x0, t_eval, method='euler')
    print(f"  traj.shape {tuple(traj.shape)}   (T, bs, 9)")
    assert traj.shape == (5, bs, 9)
    assert torch.isfinite(traj).all()
    print("  rollout OK.\n")


def test_backprop():
    print("=== 3. Backprop through rollout ===")
    net = build()
    bs = 4
    x0 = torch.cat([
        torch.rand(bs, 1) * 0.3 + 0.5,
        torch.cos(torch.rand(bs, 1)), torch.sin(torch.rand(bs, 1)),
        torch.cos(torch.rand(bs, 1)), torch.sin(torch.rand(bs, 1)),
        torch.zeros(bs, 3),
        torch.zeros(bs, 1),
    ], dim=1)
    x0.requires_grad_(True)
    t_eval = torch.linspace(0.0, 0.1, 3)
    traj = odeint(net, x0, t_eval, method='euler')
    loss = traj.pow(2).mean()
    loss.backward()
    grad_sum = sum(
        p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None
    )
    print(f"  loss = {loss.item(): .4f}")
    print(f"  sum|grad| = {grad_sum: .3e}  (must be > 0)")
    assert grad_sum > 0
    print("  gradients flow.\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward()
    test_odeint()
    test_backprop()
    print("All Lag_Net_Dumbbell smoke tests passed.")

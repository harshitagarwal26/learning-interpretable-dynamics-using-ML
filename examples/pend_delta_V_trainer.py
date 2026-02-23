# Standard library imports
from argparse import ArgumentParser
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Third party imports
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint

# local application imports
from lag_caVAE.lag import Lag_Net
from lag_caVAE.nn_models import MLP_Encoder, MLP, MLP_Decoder, PSD
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import arrange_data, from_pickle, my_collate, ImageDataset, HomoImageDataset

seed_everything(0)


class DeltaVModel(pl.LightningModule):
    """
    Perturbation identification model using additive ΔV_net.
    
    Loads a pre-trained unperturbed model, freezes ALL existing parameters
    (encoder, decoder, V_net, M_net, g_net), and creates a small ΔV_net(q)
    that learns only the perturbation potential.
    
    V_total(q) = V_net_frozen(cos q, sin q) + ΔV_net(q)
    """

    def __init__(self, hparams, data_path=None, pretrained_ckpt=None):
        super(DeltaVModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.data_path = data_path
        self.T_pred = self.hparams.T_pred
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        # --- Build architecture (same as original Model) ---
        self.recog_q_net = MLP_Encoder(32*32, 300, 3, nonlinearity='elu')
        self.obs_net = MLP_Encoder(1, 100, 32*32, nonlinearity='elu')

        V_net = MLP(2, 50, 1)
        g_net = MLP(2, 50, 1)
        M_net = PSD(2, 50, 1)

        # Create the small ΔV_net: takes raw q (q_dim=1) → scalar potential
        delta_V_net = MLP(
            self.hparams.q_dim,
            self.hparams.delta_v_hidden,
            1,
            nonlinearity='tanh'
        )

        self.ode = Lag_Net(
            q_dim=self.hparams.q_dim, u_dim=1,
            g_net=g_net, M_net=M_net, V_net=V_net,
            delta_V_net=delta_V_net
        )

        self.train_dataset = None
        self.non_ctrl_ind = 1

        # --- Load pre-trained weights and freeze ---
        if pretrained_ckpt is not None:
            self._load_and_regularize(pretrained_ckpt)

    def _load_and_regularize(self, ckpt_path):
        """Load pre-trained weights. Store reference copies for regularization."""
        from examples.pend_lag_cavae_trainer import Model as OriginalModel
        
        print(f"Loading pre-trained model from: {ckpt_path}")
        pretrained = OriginalModel.load_from_checkpoint(ckpt_path, map_location='cpu')

        # Copy encoder weights
        self.recog_q_net.load_state_dict(pretrained.recog_q_net.state_dict())
        self.obs_net.load_state_dict(pretrained.obs_net.state_dict())

        # Copy dynamics weights (V_net, M_net, g_net)
        self.ode.V_net.load_state_dict(pretrained.ode.V_net.state_dict())
        self.ode.M_net.load_state_dict(pretrained.ode.M_net.state_dict())
        self.ode.g_net.load_state_dict(pretrained.ode.g_net.state_dict())

        # --- Store pretrained weights as reference for regularization ---
        self._pretrained_params = {}
        for name, param in self.named_parameters():
            if 'delta_V_net' not in name:  # don't regularize delta_V_net
                self._pretrained_params[name] = param.data.clone()

        # Everything is trainable — regularization keeps existing params close to pretrained
        # No freezing!

        total = sum(p.numel() for p in self.parameters())
        delta_v_params = sum(p.numel() for p in self.ode.delta_V_net.parameters())
        print(f"  Total parameters:     {total}")
        print(f"  ΔV_net parameters:    {delta_v_params}")
        print(f"  Regularized params:   {total - delta_v_params}")
        print(f"  ALL parameters are trainable (existing params regularized toward pretrained)")

    def train_dataloader(self):
        if self.hparams.homo_u:
            if self.train_dataset is None:
                self.train_dataset = HomoImageDataset(self.data_path, self.hparams.T_pred)
            if self.current_epoch < 1000:
                if self.current_epoch % 2 == 0:
                    u_idx = 0
                else:
                    u_idx = self.non_ctrl_ind
                    self.non_ctrl_ind += 1
                    if self.non_ctrl_ind == 9:
                        self.non_ctrl_ind = 1
            else:
                u_idx = self.current_epoch % 9
            self.train_dataset.u_idx = u_idx
            self.t_eval = torch.from_numpy(self.train_dataset.t_eval)
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                              shuffle=True, collate_fn=my_collate)
        else:
            train_dataset = ImageDataset(self.data_path, self.hparams.T_pred)
            self.t_eval = torch.from_numpy(train_dataset.t_eval)
            return DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                              shuffle=True, collate_fn=my_collate)

    def angle_vel_est(self, q0_m_n, q1_m_n, delta_t):
        delta_cos = q1_m_n[:,0:1] - q0_m_n[:,0:1]
        delta_sin = q1_m_n[:,1:2] - q0_m_n[:,1:2]
        q_dot0 = - delta_cos * q0_m_n[:,1:2] / delta_t + delta_sin * q0_m_n[:,0:1] / delta_t
        return q_dot0

    def encode(self, batch_image):
        q_m_logv = self.recog_q_net(batch_image)
        q_m, q_logv = q_m_logv.split([2, 1], dim=1)
        q_m_n = q_m / q_m.norm(dim=-1, keepdim=True)
        q_v = F.softplus(q_logv) + 1
        return q_m, q_v, q_m_n

    def get_theta_inv(self, cos, sin, x, y, bs=None):
        bs = self.bs if bs is None else bs
        theta = torch.zeros([bs, 2, 3], dtype=self.dtype, device=self.device)
        theta[:, 0, 0] += cos ; theta[:, 0, 1] += -sin ; theta[:, 0, 2] += - x * cos + y * sin
        theta[:, 1, 0] += sin ; theta[:, 1, 1] += cos ;  theta[:, 1, 2] += - x * sin - y * cos
        return theta

    def forward(self, X, u):
        [_, self.bs, d, d] = X.shape
        T = len(self.t_eval)

        # ---- Encode ALL frames to get observed latent trajectory ----
        q_obs_list = []
        for t_idx in range(T):
            _, _, q_m_n = self.encode(X[t_idx].reshape(self.bs, d*d))
            q_obs_list.append(q_m_n)  # each is (bs, 2) = (cos_q, sin_q)
        self.q_obs = torch.stack(q_obs_list, dim=0)  # (T, bs, 2)

        # ---- Initial conditions from first two frames ----
        self.q0_m, self.q0_v, self.q0_m_n = self.encode(X[0].reshape(self.bs, d*d))
        self.q1_m, self.q1_v, self.q1_m_n = self.encode(X[1].reshape(self.bs, d*d))

        # Use deterministic or stochastic initial conditions
        use_deterministic = getattr(self.hparams, 'deterministic_encoding', False)
        if use_deterministic:
            # Use encoder mean directly — no sampling noise
            self.q0 = self.q0_m_n
            # Still need Q_q and P_q for KL (set KL weight to 0 effectively)
            self.Q_q = VonMisesFisher(self.q0_m_n, self.q0_v)
            self.P_q = HypersphericalUniform(1, device=self.device)
        else:
            # Original VAE sampling
            self.Q_q = VonMisesFisher(self.q0_m_n, self.q0_v)
            self.P_q = HypersphericalUniform(1, device=self.device)
            self.q0 = self.Q_q.rsample()
            while torch.isnan(self.q0).any():
                self.q0 = self.Q_q.rsample()

        # estimate velocity
        self.q_dot0 = self.angle_vel_est(self.q0_m_n, self.q1_m_n,
                                          self.t_eval[1]-self.t_eval[0])

        # ---- ODE integration ----
        z0_u = torch.cat((self.q0, self.q_dot0, u), dim=1)
        zT_u = odeint(self.ode, z0_u, self.t_eval, method=self.hparams.solver)
        self.qT_pred, self.q_dotT_pred, _ = zT_u.split([2, 1, 1], dim=-1)
        # qT_pred: (T, bs, 2) = predicted (cos_q, sin_q) trajectory

        return None

    def _ewc_loss(self):
        """Elastic weight consolidation loss: penalize deviation from pretrained weights."""
        loss = 0.0
        for name, param in self.named_parameters():
            if name in self._pretrained_params:
                loss = loss + ((param - self._pretrained_params[name]) ** 2).sum()
        return loss

    def training_step(self, train_batch, batch_idx):
        X, u = train_batch
        self.forward(X, u)

        # ---- Latent trajectory loss ----
        latent_loss = self.loss_fn(self.qT_pred, self.q_obs)
        latent_loss = latent_loss.sum([0, 2]).mean()

        # KL and norm penalty
        kl_q = torch.distributions.kl.kl_divergence(self.Q_q, self.P_q).mean()
        norm_penalty = (self.q0_m.norm(dim=-1).mean() - 1) ** 2

        # EWC regularization: keep existing params close to pretrained
        ewc_weight = getattr(self.hparams, 'ewc_lambda', 100.0)
        ewc_loss = self._ewc_loss()

        lambda_ = self.current_epoch/8000 if self.hparams.annealing else 1/100
        loss = latent_loss + kl_q + lambda_ * norm_penalty + ewc_weight * ewc_loss

        logs = {
            'latent_loss': latent_loss,
            'kl_q_loss': kl_q,
            'ewc_loss': ewc_loss,
            'train_loss': loss,
            'monitor': latent_loss + kl_q
        }
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # Differential learning rates: low for existing params, normal for ΔV_net
        delta_v_params = list(self.ode.delta_V_net.parameters())
        other_params = [p for n, p in self.named_parameters() if 'delta_V_net' not in n]
        
        delta_v_lr = self.hparams.learning_rate
        other_lr = self.hparams.learning_rate * 0.01  # 100x lower for existing params
        
        optimizer = torch.optim.Adam([
            {'params': delta_v_params, 'lr': delta_v_lr},
            {'params': other_params, 'lr': other_lr}
        ])
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=512, type=int)
        parser.add_argument('--q_dim', default=1, type=int)
        parser.add_argument('--delta_v_hidden', default=32, type=int,
                            help='Hidden dim for ΔV_net MLP')
        parser.add_argument('--ewc_lambda', default=100.0, type=float,
                            help='EWC regularization strength (higher = more regularized)')
        return parser


def main(args):
    # Resolve data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(
            PARENT_DIR, 'datasets',
            'pendulum-gym-image-dataset-train-reverse-angle-perturbed.pkl'
        )

    model = DeltaVModel(
        hparams=args,
        data_path=data_path,
        pretrained_ckpt=args.pretrained_ckpt
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='monitor',
        filepath=os.path.join(PARENT_DIR, 'logs', args.name, args.name + f'-T_p={args.T_pred}'),
        save_top_k=1,
        save_last=True
    )

    trainer = Trainer.from_argparse_args(
        args,
        deterministic=False,
        default_root_dir=os.path.join(PARENT_DIR, 'logs', args.name),
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--name', default='pend-delta-v', type=str)
    parser.add_argument('--T_pred', default=4, type=int)
    parser.add_argument('--solver', default='euler', type=str)
    parser.add_argument('--homo_u', dest='homo_u', action='store_true')
    parser.add_argument('--annealing', dest='annealing', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                        help='Path to unperturbed model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to perturbed training data (default: perturbed pendulum)')
    parser.add_argument('--unfreeze_encoder', action='store_true',
                        help='Unfreeze encoder during training')
    parser.add_argument('--deterministic_encoding', action='store_true',
                        help='Use deterministic encoding (no VAE sampling)')
    parser.set_defaults(homo_u=False, annealing=False, unfreeze_encoder=False, deterministic_encoding=False)
    parser = Trainer.add_argparse_args(parser)
    parser = DeltaVModel.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)

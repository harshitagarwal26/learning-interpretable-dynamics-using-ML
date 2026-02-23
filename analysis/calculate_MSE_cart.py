#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
current_directory = os.getcwd()
print(current_directory)
# local application imports
from lag_caVAE.lag import Lag_Net
from lag_caVAE.nn_models import MLP_Encoder, MLP, MLP_Decoder, PSD
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import arrange_data, from_pickle, my_collate, ImageDataset, HomoImageDataset
from examples.cart_lag_cavae_trainer import Model as Model_lag_cavae
from ablations.ablation_cart_MLPdyna_cavae_trainer import Model as Model_MLPdyna_cavae
from ablations.ablation_cart_lag_vae_trainer import Model as Model_lag_vae
from ablations.ablation_cart_lag_MLPEnc_caDec_trainer import Model as Model_lag_MLPEnc_caDec
from ablations.ablation_cart_lag_caEnc_MLPDec_trainer import Model as Model_lag_caEnc_MLPDec
from ablations.ablation_cart_lag_caAE_trainer import Model as Model_lag_caAE
from ablations.HGN import Model as Model_HGN

seed_everything(0)
# get_ipython().run_line_magic('matplotlib', 'inline')
DPI = 600


# In[3]:


checkpoint_path = os.path.join(PARENT_DIR, 
                               'results', 
                               'cart', 
                               'last.ckpt')
model_lag_cavae = Model_lag_cavae.load_from_checkpoint(checkpoint_path)

# checkpoint_path = os.path.join(PARENT_DIR, 
#                                'checkpoints', 
#                                'ablation-cart-MLPdyna-cavae-T_p=4-epoch=807.ckpt')
# model_MLPdyna_cavae = Model_MLPdyna_cavae.load_from_checkpoint(checkpoint_path)

# checkpoint_path = os.path.join(PARENT_DIR, 
#                                'checkpoints', 
#                                'ablation-cart-lag-vae-T_p=4-epoch=987.ckpt')
# model_lag_vae = Model_lag_vae.load_from_checkpoint(checkpoint_path)

# checkpoint_path = os.path.join(PARENT_DIR, 
#                                'checkpoints', 
#                                'ablation-cart-lag-MLPEnc-caDec-T_p=4-epoch=524.ckpt')
# model_lag_MLPEnc_caDec = Model_lag_MLPEnc_caDec.load_from_checkpoint(checkpoint_path)

# # this checkpoint is trained with learning rate 1e-4
# checkpoint_path = os.path.join(PARENT_DIR, 
#                                'checkpoints', 
#                                'ablation-cart-lag-caEnc-MLPDec-T_p=4-epoch=954.ckpt')
# model_lag_caEnc_MLPDec = Model_lag_caEnc_MLPDec.load_from_checkpoint(checkpoint_path)

# # this checkpoint is trained with learning rate 1e-4
# checkpoint_path = os.path.join(PARENT_DIR, 
#                                'checkpoints', 
#                                'ablation-cart-lag-caAE-T_p=4-epoch=909.ckpt')
# model_lag_caAE = Model_lag_caAE.load_from_checkpoint(checkpoint_path)

# checkpoint_path = os.path.join(PARENT_DIR, 
#                                'checkpoints', 
#                                'baseline-cart-HGN-T_p=4-epoch=1777.ckpt')
# model_HGN = Model_HGN.load_from_checkpoint(checkpoint_path)


# In[ ]:


# Load train data, prepare for plotting prediction
# WARNING: this might requires ~18G memory at peak
data_path=os.path.join(PARENT_DIR, 'datasets', 'cartpole-gym-image-dataset-rgb-u9-train.pkl')
train_dataset = HomoImageDataset(data_path, T_pred=4)
# prepare model
model_lag_cavae.t_eval = torch.from_numpy(train_dataset.t_eval)
model_lag_cavae.hparams.annealing = False
# model_MLPdyna_cavae.t_eval = torch.from_numpy(train_dataset.t_eval)
# model_lag_vae.t_eval = torch.from_numpy(train_dataset.t_eval)
# model_lag_MLPEnc_caDec.t_eval = torch.from_numpy(train_dataset.t_eval)
# model_lag_caEnc_MLPDec.t_eval = torch.from_numpy(train_dataset.t_eval)
# model_lag_caAE.t_eval = torch.from_numpy(train_dataset.t_eval)
# model_HGN.t_eval = torch.from_numpy(train_dataset.t_eval)
# model_HGN.step = 3 ; model_HGN.alpha = 1


# In[ ]:


lag_cavae_train_loss = []
MLPdyna_cavae_train_loss = []
lag_MLPEnc_caDec_train_loss = []
lag_caEnc_MLPDec_train_loss = []
lag_vae_train_loss = []
lag_caAE_train_loss = []

for i in range(len(train_dataset.x)):
    train_dataset.u_idx = i
    dataLoader = DataLoader(train_dataset, batch_size=512, shuffle=False, collate_fn=my_collate)
    for batch in dataLoader:
        lag_cavae_train_loss.append(model_lag_cavae.training_step(batch, 0)['log']['recon_loss'].item())
        # MLPdyna_cavae_train_loss.append(model_MLPdyna_cavae.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_vae_train_loss.append(model_lag_vae.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_MLPEnc_caDec_train_loss.append(model_lag_MLPEnc_caDec.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_caEnc_MLPDec_train_loss.append(model_lag_caEnc_MLPDec.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_caAE_train_loss.append(model_lag_caAE.training_step(batch, 0)['log']['recon_loss'].item())


# In[ ]:


# HGN_train_loss = []
# train_dataset.u_idx = 0
# dataLoader = DataLoader(train_dataset, batch_size=256, shuffle=False, collate_fn=my_collate)
# for batch in dataLoader:
#     HGN_train_loss.append(model_HGN.training_step(batch, 0)['log']['recon_loss'].item())


# In[ ]:


del dataLoader
del train_dataset


# In[ ]:


# Load train data, prepare for plotting prediction
# WARNING: this might requires ~18G memory at peak
# data_path=os.path.join(PARENT_DIR, 'datasets', 'cartpole-gym-image-dataset-rgb-u9-test.pkl')
# test_dataset = HomoImageDataset(data_path, T_pred=4)


# In[ ]:


lag_cavae_test_loss = []
MLPdyna_cavae_test_loss = []
lag_MLPEnc_caDec_test_loss = []
lag_caEnc_MLPDec_test_loss = []
lag_vae_test_loss = []
lag_caAE_test_loss = []

# for i in range(len(test_dataset.x)):
#     test_dataset.u_idx = i
#     dataLoader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=my_collate)
#     for batch in dataLoader:
#         lag_cavae_test_loss.append(model_lag_cavae.training_step(batch, 0)['log']['recon_loss'].item())
        # MLPdyna_cavae_test_loss.append(model_MLPdyna_cavae.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_vae_test_loss.append(model_lag_vae.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_MLPEnc_caDec_test_loss.append(model_lag_MLPEnc_caDec.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_caEnc_MLPDec_test_loss.append(model_lag_caEnc_MLPDec.training_step(batch, 0)['log']['recon_loss'].item())
        # lag_caAE_test_loss.append(model_lag_caAE.training_step(batch, 0)['log']['recon_loss'].item())


# In[ ]:


# HGN_test_loss = []
# test_dataset.u_idx = 0
# dataLoader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=my_collate)
# for batch in dataLoader:
#     HGN_test_loss.append(model_HGN.training_step(batch, 0)['log']['recon_loss'].item())


# In[ ]:


scale = 64*64*5
print(f'lag_cavae: train: {np.mean(lag_cavae_train_loss)/scale}')
    #   , test: {np.mean(lag_cavae_test_loss)/scale}')
# print(f'MLPdyna_cavae: train: {np.mean(MLPdyna_cavae_train_loss)/scale}, test: {np.mean(MLPdyna_cavae_test_loss)/scale}')
# print(f'lag_vae: train: {np.mean(lag_vae_train_loss)/scale}, test: {np.mean(lag_vae_test_loss)/scale}')
# print(f'lag_MLPEnc_caDec: train: {np.mean(lag_MLPEnc_caDec_train_loss)/scale}, test: {np.mean(lag_MLPEnc_caDec_test_loss)/scale}')
# print(f'lag_caEnc_MLPDec: train: {np.mean(lag_caEnc_MLPDec_train_loss)/scale}, test: {np.mean(lag_caEnc_MLPDec_test_loss)/scale}')
# print(f'lag_caAE: train: {np.mean(lag_caAE_train_loss)/scale}, test: {np.mean(lag_caAE_test_loss)/scale}')
# print(f'HGN: train: {np.mean(HGN_train_loss)/scale}, test: {np.mean(HGN_test_loss)/scale}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





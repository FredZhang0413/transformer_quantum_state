# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:38:55 2022

@author: Yuanhang Zhang
"""


from model import TransformerModel
from model_utils import sample, compute_observable
from Hamiltonian import Ising, XYZ
from optimizer import Optimizer
from evaluation import compute_E_sample, compute_magnetization

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plt_config

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def summarize_ensemble(values):
    values = np.asarray(values)
    return {
        'q10': np.quantile(values, 0.1, axis=1),
        'median': np.quantile(values, 0.5, axis=1),
        'q90': np.quantile(values, 0.9, axis=1),
        'mean': values.mean(axis=1),
        'std': values.std(axis=1),
    }


def save_test_visualizations(folder, save_str, h, E_samples, m_samples, dEs, E_dmrgs):
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    E_stats = summarize_ensemble(E_samples)
    m_stats = summarize_ensemble(m_samples)
    dE_stats = summarize_ensemble(dEs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(h, E_dmrgs / n, color=color[2], ls='--', label='DMRG')
    axes[0, 0].plot(h, E_stats['median'], color=color[0], label='TQS median')
    axes[0, 0].fill_between(h, E_stats['q10'], E_stats['q90'], color=color[0], alpha=0.2, label='10-90%')
    axes[0, 0].set_xlabel('$h$')
    axes[0, 0].set_ylabel('Energy Per Site')
    axes[0, 0].set_title('Energy vs Field')
    axes[0, 0].legend()

    axes[0, 1].plot(h, dE_stats['median'], color=color[1], label='Relative Error')
    axes[0, 1].fill_between(h, dE_stats['q10'], dE_stats['q90'], color=color[1], alpha=0.2)
    axes[0, 1].set_xlabel('$h$')
    axes[0, 1].set_ylabel('Relative Energy Error')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Energy Error vs Field')
    axes[0, 1].legend()

    axes[1, 0].plot(h, m_stats['median'], color=color[3], label=r'$|m_z|$ median')
    axes[1, 0].fill_between(h, m_stats['q10'], m_stats['q90'], color=color[3], alpha=0.2)
    axes[1, 0].set_xlabel('$h$')
    axes[1, 0].set_ylabel(r'$|\langle \sigma^z \rangle|$')
    axes[1, 0].set_title('Magnetization vs Field')
    axes[1, 0].legend()

    axes[1, 1].hist(dEs.reshape(-1), bins=30, color=color[1], alpha=0.8)
    axes[1, 1].set_xlabel('Relative Energy Error')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Error Distribution Over All Test Runs')

    plt.tight_layout()
    plt.savefig(f'{folder}test_summary_{save_str}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for ensemble_id in range(E_samples.shape[1]):
        ax.plot(h, E_samples[:, ensemble_id], color=color[0], alpha=0.15)
    ax.plot(h, E_dmrgs / n, color=color[2], ls='--', lw=2, label='DMRG')
    ax.plot(h, E_stats['median'], color=color[1], lw=2, label='TQS median')
    ax.set_xlabel('$h$')
    ax.set_ylabel('Energy Per Site')
    ax.set_title('Per-Ensemble Energy Traces')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{folder}test_energy_ensembles_{save_str}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    summary_table = np.column_stack([
        h,
        E_stats['median'],
        E_dmrgs / n,
        dE_stats['median'],
        m_stats['median'],
        E_stats['std'],
        dE_stats['std'],
        m_stats['std'],
    ])
    np.savetxt(
        f'{folder}test_summary_{save_str}.csv',
        summary_table,
        delimiter=',',
        header='h,E_tqs_median,E_dmrg,E_relerr_median,mz_median,E_std,dE_std,mz_std',
        comments='',
    )

try:
    os.mkdir('results/')
except FileExistsError:
    pass

system_sizes = torch.tensor([[40]], dtype=torch.int64, device='cpu')
H = Ising(system_sizes[0], periodic=False)
# H = XYZ(system_size[0], periodic=False)

n = int(H.n)
param_dim = H.param_dim
embedding_size = 32
n_head = 8
n_hid = embedding_size
n_layers = 8
dropout = 0
minibatch = 10000
batch = 10000
max_unique = 1000
ensemble_size = 10
name = type(H).__name__

folder = 'results/'
save_str = f'{name}_{embedding_size}_{n_head}_{n_layers}'
checkpoint_path = f'{folder}ckpt_100000_{save_str}_0.ckpt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
use_custom_attention = any('self_attn.linear_Q' in key for key in checkpoint)
model = TransformerModel(system_sizes, param_dim, embedding_size, n_head, n_hid, n_layers,
                         dropout=dropout, minibatch=minibatch, custom_attention=use_custom_attention)
num_params = sum([param.numel() for param in model.parameters()])
print('Number of parameters: ', num_params)
model.load_state_dict(checkpoint)

optim = Optimizer(model, [H])

n_data_point = 101
param = torch.tensor([1], dtype=torch.get_default_dtype())

E_samples = np.zeros((n_data_point, ensemble_size))
E_exacts = np.zeros((n_data_point, ensemble_size))
m_samples = np.zeros((n_data_point, ensemble_size))
E_dmrgs = np.load(f'{folder}E_dmrg_40.npy')

dEs = np.zeros((n_data_point, ensemble_size))

h = np.arange(n_data_point) / (n_data_point-1) * 2   # [0, 2]
with torch.no_grad():
    for ensemble_id in range(ensemble_size):
        for i in range(n_data_point):
            param[0] = h[i]
            model.set_param(system_sizes[0], param)

            start = time.time()
            print_str = f'{i} {h[i]:.2f} {ensemble_id} '
            samples, sample_weight = sample(model, batch, max_unique, symmetry=H.symmetry)

            t1 = time.time()

            E = H.Eloc(samples, sample_weight, model)
            E_sample = (E * sample_weight).sum()
            E_sample = E_sample.real.detach().cpu().numpy() / n

            print_str += f'{E_sample:.6f}\t'
            E_dmrg = E_dmrgs[i] / n
            print_str += f'{E_dmrg:.6f}\t'

            dE = np.abs((E_sample - E_dmrg) / E_dmrg)
            print_str += f'{dE:.6f}\t'

            t2 = time.time()

            # (n_op, batch)
            samples_pm = 2 * samples - 1
            mz = (samples_pm.mean(dim=0).abs() * sample_weight).sum().detach().cpu().numpy()

            print_str += f'{mz:.6f}\t'
            E_samples[i, ensemble_id] = E_sample
            m_samples[i, ensemble_id] = mz
            dEs[i, ensemble_id] = dE
            t3 = time.time()
            print_str += f'{t1-start:.4f} {t2-t1:.4f} {t3-t2:.4f}'
            print(print_str)

with open(f'results/E_sample_{save_str}.npy', 'wb') as f:
    np.save(f, E_samples)
with open(f'results/m_sample_{save_str}.npy', 'wb') as f:
    np.save(f, m_samples)
with open(f'results/dE_{save_str}.npy', 'wb') as f:
    np.save(f, dEs)

save_test_visualizations(folder, save_str, h, E_samples, m_samples, dEs, E_dmrgs)
print(f'Saved test summary plot to {folder}test_summary_{save_str}.png')
print(f'Saved test ensemble plot to {folder}test_energy_ensembles_{save_str}.png')
print(f'Saved test summary table to {folder}test_summary_{save_str}.csv')


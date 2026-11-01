"""Final C3F: country-specific linear bases and one shared coefficient MoE."""
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import COUNTRIES, oil_factors

BASE_EPOCHS = 80
CORRECTION_STEPS = 400


def fit_base(data):
    phi = data['phi']
    y = torch.as_tensor((data['y']-data['y_mean'])/data['y_std'], dtype=torch.float32)
    model = nn.Linear(60, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    generator = torch.Generator().manual_seed(42+2001)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.01)
    for epoch in range(BASE_EPOCHS):
        optimizer.param_groups[0]['lr'] = .003*(.05+.95*.5*(1+math.cos(math.pi*epoch/BASE_EPOCHS)))
        for ids in torch.randperm(len(y), generator=generator).split(32):
            optimizer.zero_grad()
            prediction = model(phi[ids]).squeeze(-1)
            penalty = model.weight[:, :48].square().sum()
            penalty += .01*(model.weight[:, 48:].square().sum()+model.bias.square().sum())
            loss = (prediction-y[ids]).square().mean()+penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    return model.eval()


def base_predict(model, data):
    with torch.no_grad():
        return (model(data['eval_phi']).squeeze(-1).double().numpy()
                *data['y_std']+data['y_mean'])


class SharedCorrector(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.cap = .5
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+29073)
            self.experts = nn.ModuleList([nn.Linear(60, 4) for _ in range(4)])
            self.gate = nn.Sequential(nn.Linear(60, 8), nn.Tanh(), nn.Linear(8, 4))
            for expert in self.experts:
                nn.init.zeros_(expert.weight)
                nn.init.zeros_(expert.bias)
        self.beta = nn.Parameter(torch.zeros(5))

    def forward(self, phi, oil):
        route = self.gate(phi).softmax(-1)
        experts = torch.stack([expert(phi) for expert in self.experts], 1)
        coefficients = .05*torch.tanh((route[:, :, None]*experts).sum(1))
        raw = oil@self.beta[:4]+self.beta[4]+(oil*coefficients).sum(-1)
        return self.cap*torch.tanh(raw/self.cap)


def fit_corrector(data, history, month, seed=42):
    """Fit on historical base forecast errors using the frozen paper settings."""
    target = pd.Timestamp(month)
    cutoff_id = target.year*12+target.month-2
    xs, oils, ys, groups, ages, scales = [], [], [], [], [], []
    for i, country in enumerate(COUNTRIES):
        d = data[country]
        dates = d['dates']
        month_ids = dates.year.to_numpy()*12+dates.month.to_numpy()-1
        prior = history.loc[history.country.eq(country)].set_index('month').prediction
        forecasts = prior.reindex(dates).to_numpy(float)
        mask = np.isfinite(forecasts) & (month_ids >= cutoff_id-59)
        if not mask.any() or (month_ids[mask] > cutoff_id).any():
            raise ValueError(f'{country}: historical forecast errors are incomplete')
        oil = oil_factors(d['phi'])
        scale = oil.std(0, unbiased=False).clamp_min(.1)
        scales.append(scale)
        xs.append(d['phi'][mask])
        oils.append((oil/scale)[mask])
        ys.append(torch.as_tensor(d['y']-np.where(np.isfinite(forecasts), forecasts, 0.),
                                  dtype=torch.float32)[mask])
        groups.append(torch.full((int(mask.sum()),), i, dtype=torch.long))
        ages.append(torch.as_tensor(cutoff_id-month_ids[mask], dtype=torch.float32))
    x, oil, y = torch.cat(xs), torch.cat(oils), torch.cat(ys)
    group, weight = torch.cat(groups), torch.exp2(-torch.cat(ages)/24)
    weight_sums = weight.new_zeros(7).index_add(0, group, weight)

    def country_mean(values):
        return values.new_zeros(7).index_add(0, group, values*weight)/weight_sums

    model = SharedCorrector(seed)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if n != 'beta'],
         'lr': .003, 'initial_lr': .003, 'weight_decay': .001},
        {'params': [model.beta], 'lr': .01, 'initial_lr': .01, 'weight_decay': 0.},
    ])

    def objective():
        out = model(x, oil)
        risk = (country_mean((out-y).square())+1e-8).sqrt().mean()
        energy = country_mean(out.square()).mean()
        penalty = (model.beta.square()*model.beta.new_tensor([1, 1, 1, 1, 10])).sum()
        return risk+10*energy+penalty

    for step in range(CORRECTION_STEPS):
        factor = .05+.95*.5*(1+math.cos(math.pi*step/CORRECTION_STEPS))
        for opt_group in optimizer.param_groups:
            opt_group['lr'] = opt_group['initial_lr']*factor
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite correction loss')
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = float(objective())
    return model, torch.stack(scales), dict(objective=final_loss, error_samples=len(y), steps=400)

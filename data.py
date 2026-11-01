"""Monthly inputs for the final C3F model."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COUNTRIES = ('USA', 'GBR', 'DEU', 'FRA', 'ITA', 'CAN', 'JPN')
SCHEMA = ('cpi', 'core_cpi', 'indpro', 'unrate', 'rate_10y',
          'neer', 'share_price', 'oil_wti')


def load_panels(folder):
    panels = {}
    for country in COUNTRIES:
        frame = pd.read_csv(Path(folder)/f'{country}.csv', parse_dates=['DATE']).set_index('DATE')
        if tuple(frame.columns) != SCHEMA:
            raise ValueError(f'{country}: expected columns {SCHEMA}')
        if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq='MS')):
            raise ValueError(f'{country}: dates must form a monthly calendar')
        panels[country] = frame
    return panels


def representation(x):
    a = x[:, :, :8]
    statistics = torch.stack((a[:, -1], a[:, -3:].mean(1), a[:, -12:].mean(1),
                              a[:, -12], a[:, -24], a[:, -1]-a[:, -13]), -1)
    return torch.cat((statistics.flatten(1), x[:, -1, 8:]-1/12), -1)


def make_data(frame, month, normalization=None):
    """Fit normalization on months before the forecast; keep observed targets."""
    target = pd.Timestamp(month)
    if target.day != 1 or target not in frame.index:
        raise ValueError('Forecast month must be a month start within the supplied panel')
    cutoff = target-pd.DateOffset(months=1)
    filled = frame.ffill()
    prefix = filled.loc[:cutoff]
    if prefix.empty or not np.isfinite(prefix.to_numpy()).all():
        raise ValueError('Historical inputs have unfillable initial gaps or nonfinite values')
    if normalization is None:
        mean, std = prefix.mean(), prefix.std(ddof=0).replace(0, 1)
    else:
        mean = pd.Series(normalization['mean'], index=SCHEMA)
        std = pd.Series(normalization['std'], index=SCHEMA)
    scaled = (filled-mean)/std
    dates = frame.index[30:]
    dates = dates[(dates <= cutoff) & frame.loc[dates, 'cpi'].notna()]
    if len(dates) == 0:
        raise ValueError('No observed training targets after the 30-month lookback')

    def features(targets):
        blocks = []
        for date in targets:
            history = scaled.loc[scaled.index < date].tail(30).to_numpy(np.float32)
            if history.shape != (30, 8) or not np.isfinite(history).all():
                raise ValueError(f'Incomplete past inputs for {date:%Y-%m}')
            seasonal = np.eye(12, dtype=np.float32)[date.month-1]
            blocks.append(np.concatenate((history, np.broadcast_to(seasonal, (30, 12))), axis=1))
        return representation(torch.from_numpy(np.asarray(blocks, dtype=np.float32)))

    return dict(phi=features(dates), dates=dates,
                y=frame.loc[dates, 'cpi'].to_numpy(float),
                eval_phi=features(pd.DatetimeIndex([target])),
                y_mean=float(mean.cpi), y_std=float(std.cpi),
                mean=mean.to_numpy(float).tolist(), std=std.to_numpy(float).tolist())


def oil_factors(phi):
    d1, d2 = phi[:, 42]-phi[:, 43], phi[:, 43]-phi[:, 44]
    return 2*torch.tanh(torch.stack((d1, d2, d1.abs(), d2.abs()), -1)/2)

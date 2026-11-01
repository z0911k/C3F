"""Train the final C3F model and predict one month for the seven G7 countries."""
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import COUNTRIES, load_panels, make_data, oil_factors
from model import SharedCorrector, base_predict, fit_base, fit_corrector

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger('c3f')


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    panels = load_panels(args.data_dir)
    target = pd.Timestamp(args.month)
    if target.day != 1:
        raise ValueError('--month must identify a month start')
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved['month'] != target.strftime('%Y-%m'):
            raise ValueError('Checkpoint month must match --month')
        current = {c: make_data(panels[c], target, saved['normalization'][c]) for c in COUNTRIES}
        bases = {c: nn.Linear(60, 1) for c in COUNTRIES}
        for c in COUNTRIES:
            bases[c].load_state_dict(saved['bases'][c])
            bases[c].eval()
        corrector = SharedCorrector(saved['seed'])
        corrector.load_state_dict(saved['corrector'])
        corrector.eval()
        scales, fit_info = saved['oil_scale'], saved['fit']
        LOGGER.info('Loaded checkpoint for %s', args.month)
    else:
        warmup = pd.date_range(target-pd.DateOffset(months=60), target-pd.DateOffset(months=1), freq='MS')
        rows = []
        LOGGER.info('Rebuilding 60 months of historical base forecasts from the supplied data')
        for i, month in enumerate(warmup, 1):
            for c in COUNTRIES:
                if month not in panels[c].index:
                    raise ValueError(f'{c}: the 60-month history is outside the supplied data')
                if pd.isna(panels[c].loc[month, 'cpi']):
                    continue
                d = make_data(panels[c], month)
                prediction = float(base_predict(fit_base(d), d)[0])
                rows.append(dict(country=c, month=month, prediction=prediction))
            if i == 1 or i % 10 == 0:
                LOGGER.info('Historical base forecasts: %d/60 months', i)
        history = pd.DataFrame(rows)
        history.to_csv(args.out_dir/'base_forecasts.csv', index=False)
        current = {c: make_data(panels[c], target) for c in COUNTRIES}
        LOGGER.info('Fitting the seven final country bases (80 epochs each)')
        bases = {c: fit_base(current[c]) for c in COUNTRIES}
        LOGGER.info('Fitting the final shared corrector (400 updates, cap=0.5)')
        corrector, scales, fit_info = fit_corrector(current, history, target, args.seed)
        saved = dict(month=target.strftime('%Y-%m'), seed=args.seed,
                     bases={c: bases[c].state_dict() for c in COUNTRIES},
                     corrector=corrector.state_dict(), oil_scale=scales, fit=fit_info,
                     normalization={c: {k: current[c][k] for k in ('mean', 'std')} for c in COUNTRIES})
        torch.save(saved, args.out_dir/'model.pt')
    rows = []
    for i, c in enumerate(COUNTRIES):
        d = current[c]
        base = float(base_predict(bases[c], d)[0])
        with torch.no_grad():
            change = float(corrector(d['eval_phi'], oil_factors(d['eval_phi'])/scales[i])[0])
        prediction = base+change
        if not np.isfinite(prediction):
            raise RuntimeError(f'{c}: nonfinite forecast')
        rows.append(dict(country=c, month=target.strftime('%Y-%m'),
                         prediction=prediction, base=base, correction=change,
                         observed_cpi=panels[c].loc[target, 'cpi']))
    output = pd.DataFrame(rows)
    output.to_csv(args.out_dir/'predictions.csv', index=False)
    report = dict(month=target.strftime('%Y-%m'), seed=saved['seed'], countries=list(COUNTRIES),
                  base_parameters_each=61, corrector_parameters=sum(p.numel() for p in corrector.parameters()),
                  base_epochs=80, correction_cap_pp=.5, fit=fit_info,
                  elapsed_seconds=time.perf_counter()-started,
                  environment=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__),
                  data_sha256={c: hashlib.sha256((args.data_dir/f'{c}.csv').read_bytes()).hexdigest() for c in COUNTRIES})
    (args.out_dir/'run.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    LOGGER.info('\n%s', output.to_string(index=False))
    LOGGER.info('Finished in %.1fs; outputs: %s', report['elapsed_seconds'], args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month', default='2026-06', help='Forecast month, YYYY-MM')
    parser.add_argument('--seed', type=int, default=42, help='Shared corrector seed; the country base seed is fixed at 42')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--checkpoint', type=Path, help='Load a checkpoint instead of training')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(args.out_dir/'run.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    try:
        run(args)
    except Exception:
        LOGGER.exception('Run failed')
        raise

# C3F

## Run

Python 3.10–3.13, with CPU execution. From this directory:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run.py
```

The default run fits the model for **June 2026**, using shared-corrector seed **42**. It rebuilds the preceding 60 months of country base forecasts, trains the seven final country bases and the shared corrector, and predicts CPI changes for all seven countries. Allow several minutes for training.

Outputs are saved in `outputs/`:

- `predictions.csv`: the seven forecasts and their base/correction components, in percentage points.
- `model.pt`: trained weights, country normalization and energy-factor scales.
- `base_forecasts.csv`: the historical base forecasts used to train the corrector.
- `run.json` and `run.log`: settings, data hashes and the execution record.

To reload the saved model and reproduce its forecasts without training:

```bash
python run.py --checkpoint outputs/model.pt --out-dir outputs_reload
```

Another archived forecast month can be selected with `--month YYYY-MM`; the input panel must include the preceding 60 forecast months and their 30-month lookbacks. The shared-corrector seed can be set with `--seed` (the paper uses 42, 7 and 13). Country base training always uses seed 42.

## Model

Each country has a 61-parameter linear base, trained for 80 epochs on a 60-dimensional representation of the previous 30 monthly observations and target-month seasonality. One shared 1,505-parameter mixture of four experts produces conditional energy-factor coefficients. It is trained for 400 updates on historical base forecast errors, using a 60-month window and a 24-month weighting half-life. The correction bound is **0.5 percentage points**. These are the final paper settings.

## Data

`data/{USA,GBR,DEU,FRA,ITA,CAN,JPN}.csv` contains the frozen, transformed monthly inputs used by the paper, starting in January 2000. `data/sources.csv` records the original providers, series identifiers and selection filters. The inputs are already transformed: CPI, core CPI, industrial production, effective exchange rates, share prices and WTI oil use monthly percentage changes; unemployment and 10-year interest rates use monthly differences. Do not transform them again.

Normalization uses observations before each forecast month. Input gaps are forward filled from past observations; missing CPI targets are excluded from training and historical error labels. Observed CPI values in the output are included for checking forecasts and are not used as forecast inputs.

The data snapshot uses the revised values collected for the paper in October 2026. Source providers retain their rights to the underlying series; their access and reuse terms apply.

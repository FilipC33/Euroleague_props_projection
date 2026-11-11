# 2026 EuroLeague projection model

The projection code is isolated in `projection_model.py` and is used by the
Projection tab in `euroleague_props.py`.

## Data scope

- This branch uses the 2026 season only.
- Box scores are cached at `data_cache/boxscores_2026.parquet`.
- Shot data is read from `data_cache/shots_2026.parquet`.
- No 2025 rows are intentionally loaded or mixed into these features.
- After updating to this branch, click **Fetch / update box scores** once to
  rebuild an older box-score cache with the additional shooting and rebounding
  columns.

## Projection inputs

- Player box-score rates per 40 minutes, shrunk toward the league rate using a
  120-minute prior.
- Expected player minutes from recent and season-to-date rotation averages.
- Manual unavailable-player selections; their minutes are redistributed over
  available players, with a 40-minute regulation cap per player and 200 team
  minutes in total.
- Team possessions estimated as
  `FGA - OREB + TO + 0.44 * FTA`, normalized to a 40-minute pace.
- Expected pace blends the recent pace of both teams.
- Player shot-zone mix and points per attempt, shrunk toward league zone rates
  using a 30-attempt prior.
- Shot zones are taken directly from the existing `zone()` function in
  `euroleague_props.py`; the projection module consumes its `Zone` column and
  does not classify coordinates or create another zone dataset.
- Opponent shot-zone attempt volume and field-goal accuracy, shrunk over games
  and capped to keep small samples from dominating.
- Opponent defensive rating and position-based points, rebounds and assists
  allowed. Position adjustments are deliberately capped and sample-shrunk.

## Validation

The Projection tab includes a chronological walk-forward benchmark. For each
target game, the rate/minutes/pace baseline is rebuilt using earlier games only.
The first benchmark does not yet include the live shot-zone and position
adjustments; those layers should be added to the benchmark and calibrated before
treating the output as a finished betting model.

## Important limitations

- Unavailable players are selected manually; the app does not yet ingest live
  injury reports.
- Positions are initially estimated from box-score production. Review and edit
  them in the Positions tab.
- The model currently produces point estimates, not calibrated outcome
  probabilities. Use the backtest results before relying on prop edges.

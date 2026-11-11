"""Season-2026 feature foundations for the EuroLeague projection model.

This module is separate from the Streamlit research app and only accepts the
2026 player-game box-score table.
"""
from __future__ import annotations

import pandas as pd

SEASON = 2026
REGULATION_MINUTES = 40.0
SHRINK_PRIOR_MINUTES = 120.0

RATE_STATS = [
    "PTS", "REB", "AST", "3PM", "STL", "BLK", "TO",
    "FGA", "3PA", "FTA", "FTM", "OREB", "DREB",
]
TEAM_STATS = ["PTS", "FGA", "OREB", "TO", "FTA"]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required box-score columns: {missing}")


def player_rate_table(
    box: pd.DataFrame,
    prior_minutes: float = SHRINK_PRIOR_MINUTES,
) -> pd.DataFrame:
    """Return player totals and shrunk per-40 rates.

    The league prior is weighted by minutes, not by averaging player averages.
    DNP rows do not contribute. The default prior is equivalent to 120 minutes
    of league-average production.
    """
    if prior_minutes < 0:
        raise ValueError("prior_minutes must be non-negative")
    _require_columns(box, ["Player_ID", "Player", "Team", "MIN"] + RATE_STATS)

    played = box.loc[pd.to_numeric(box["MIN"], errors="coerce").fillna(0) > 0].copy()
    if played.empty:
        return pd.DataFrame(columns=["Player_ID", "Player", "Team", "GP", "MIN"])

    for stat in RATE_STATS:
        played[stat] = pd.to_numeric(played[stat], errors="coerce").fillna(0)
    played["MIN"] = pd.to_numeric(played["MIN"], errors="coerce").fillna(0)

    aggregations = {
        "Player": ("Player", "last"),
        "Team": ("Team", "last"),
        "GP": ("Gamecode", "nunique") if "Gamecode" in played.columns else ("MIN", "count"),
        "MIN": ("MIN", "sum"),
    }
    aggregations.update({stat: (stat, "sum") for stat in RATE_STATS})
    totals = played.groupby("Player_ID", as_index=False).agg(**aggregations)

    league_minutes = played["MIN"].sum()
    if league_minutes <= 0:
        return totals
    league_rates = {
        stat: played[stat].sum() * REGULATION_MINUTES / league_minutes
        for stat in RATE_STATS
    }

    for stat in RATE_STATS:
        totals[f"{stat}_per40"] = totals[stat] * REGULATION_MINUTES / totals["MIN"]
        prior_stat_total = league_rates[stat] * prior_minutes / REGULATION_MINUTES
        totals[f"{stat}_per40_shrunk"] = (
            (totals[stat] + prior_stat_total) * REGULATION_MINUTES
            / (totals["MIN"] + prior_minutes)
        )

    totals["league_MIN_prior"] = prior_minutes
    return totals.sort_values("MIN", ascending=False).reset_index(drop=True)


def team_game_table(box: pd.DataFrame) -> pd.DataFrame:
    """Build team-game possessions, per-40 pace and defensive rating.

    Possessions are estimated as FGA - OREB + TO + 0.44 * FTA. Game length is
    inferred from total player-minutes / five, allowing pace to be normalized
    for overtime. DRtg is opponent points per 100 opponent possessions.
    """
    _require_columns(box, ["Gamecode", "Team", "MIN"] + TEAM_STATS)

    data = box.copy()
    for column in TEAM_STATS + ["MIN"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    group_cols = ["Gamecode", "Team"]
    if "Home" in data.columns:
        group_cols.append("Home")
    teams = data.groupby(group_cols, as_index=False).agg(
        **{stat: (stat, "sum") for stat in TEAM_STATS},
        TEAM_MIN=("MIN", "sum"),
    )
    teams["Poss"] = (
        teams["FGA"] - teams["OREB"] + teams["TO"] + 0.44 * teams["FTA"]
    ).clip(lower=0)
    teams["Game_minutes"] = (teams["TEAM_MIN"] / 5.0).clip(lower=REGULATION_MINUTES)
    teams["Pace_per40"] = teams["Poss"] * REGULATION_MINUTES / teams["Game_minutes"]

    opponents = teams[["Gamecode", "Team", "PTS", "Poss"]].rename(
        columns={"Team": "Opp", "PTS": "Opp_PTS", "Poss": "Opp_Poss"}
    )
    teams = teams.merge(opponents, on="Gamecode", how="inner")
    teams = teams.loc[teams["Team"] != teams["Opp"]].copy()
    teams["DRtg"] = (
        teams["Opp_PTS"] * 100 / teams["Opp_Poss"].where(teams["Opp_Poss"] > 0)
    )
    return teams.sort_values(["Gamecode", "Team"]).reset_index(drop=True)


def league_context(team_games: pd.DataFrame) -> dict[str, float]:
    """Return league-average 2026 pace and defensive-rating baselines."""
    _require_columns(team_games, ["Pace_per40", "DRtg"])
    return {
        "league_pace_per40": float(team_games["Pace_per40"].mean()),
        "league_drtg": float(team_games["DRtg"].mean()),
    }

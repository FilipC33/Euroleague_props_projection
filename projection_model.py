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


ZONES = ["Rim", "Paint", "Midrange", "Corner 3", "Above-the-break 3"]


def add_shot_zones(shots: pd.DataFrame) -> pd.DataFrame:
    """Apply the same court-zone boundaries used by the existing app."""
    _require_columns(shots, ["COORD_X", "COORD_Y", "Result", "POINTS"])
    data = shots.copy()
    data["COORD_X"] = pd.to_numeric(data["COORD_X"], errors="coerce")
    data["COORD_Y"] = pd.to_numeric(data["COORD_Y"], errors="coerce")
    data = data.dropna(subset=["COORD_X", "COORD_Y"])
    data = data[(data["COORD_X"] != -1) & (data["COORD_Y"] != -1)].copy()

    x, y = data["COORD_X"], data["COORD_Y"]
    distance = (x ** 2 + y ** 2) ** 0.5
    rim = distance <= 100
    paint = (x.abs() <= 245) & (y <= 430) & ~rim
    corner = (x.abs() >= 660) & (y <= 142)
    above_break = (distance >= 675) & ~corner

    data["Zone"] = "Midrange"
    data.loc[rim, "Zone"] = "Rim"
    data.loc[paint, "Zone"] = "Paint"
    data.loc[corner, "Zone"] = "Corner 3"
    data.loc[above_break, "Zone"] = "Above-the-break 3"
    data["FGM"] = data["Result"].eq("Made").astype(int)
    data["POINTS"] = pd.to_numeric(data["POINTS"], errors="coerce").fillna(0)
    return data


def player_zone_table(shots: pd.DataFrame) -> pd.DataFrame:
    """Player shooting profile by zone: volume share, accuracy and points/shot."""
    _require_columns(shots, ["ID_PLAYER", "COORD_X", "COORD_Y", "Result", "POINTS"])
    data = add_shot_zones(shots)
    if data.empty:
        return pd.DataFrame(columns=[
            "ID_PLAYER", "Zone", "FGA", "FGM", "PTS", "FGA_share", "FG_pct", "PPP"
        ])

    grouped = data.groupby(["ID_PLAYER", "Zone"], as_index=False).agg(
        FGA=("Result", "size"), FGM=("FGM", "sum"), PTS=("POINTS", "sum")
    )
    player_attempts = grouped.groupby("ID_PLAYER")["FGA"].transform("sum")
    grouped["FGA_share"] = grouped["FGA"] / player_attempts
    grouped["FG_pct"] = grouped["FGM"] / grouped["FGA"].where(grouped["FGA"] > 0)
    grouped["PPP"] = grouped["PTS"] / grouped["FGA"].where(grouped["FGA"] > 0)
    return grouped.sort_values(["ID_PLAYER", "Zone"]).reset_index(drop=True)


def opponent_zone_defense(shots: pd.DataFrame) -> pd.DataFrame:
    """Zone shots conceded per team-game, compared with the 2026 league mean.

    Each defensive team-game is retained for every zone, including zones where
    it allowed zero attempts. League FG% is weighted by attempts.
    """
    _require_columns(shots, ["Gamecode", "TEAM", "COORD_X", "COORD_Y", "Result", "POINTS"])
    data = add_shot_zones(shots)
    if data.empty:
        return pd.DataFrame()

    game_teams = data[["Gamecode", "TEAM"]].drop_duplicates()
    opponent_rows = []
    for gamecode, group in game_teams.groupby("Gamecode"):
        teams = group["TEAM"].dropna().unique().tolist()
        if len(teams) == 2:
            opponent_rows.extend([
                {"Gamecode": gamecode, "DefTeam": teams[0], "OffTeam": teams[1]},
                {"Gamecode": gamecode, "DefTeam": teams[1], "OffTeam": teams[0]},
            ])
    matchups = pd.DataFrame(opponent_rows)
    if matchups.empty:
        return pd.DataFrame()

    allowed = data.merge(
        matchups, left_on=["Gamecode", "TEAM"], right_on=["Gamecode", "OffTeam"],
        how="inner",
    )
    observed = allowed.groupby(["Gamecode", "DefTeam", "Zone"], as_index=False).agg(
        FGA=("Result", "size"), FGM=("FGM", "sum"), PTS=("POINTS", "sum")
    )

    zone_frame = pd.DataFrame({"Zone": ZONES})
    complete = matchups.assign(_key=1).merge(zone_frame.assign(_key=1), on="_key").drop(columns="_key")
    complete = complete.merge(observed, on=["Gamecode", "DefTeam", "Zone"], how="left")
    for stat in ["FGA", "FGM", "PTS"]:
        complete[stat] = complete[stat].fillna(0)

    team_zone = complete.groupby(["DefTeam", "Zone"], as_index=False).agg(
        Games=("Gamecode", "nunique"), FGA=("FGA", "sum"),
        FGM=("FGM", "sum"), PTS=("PTS", "sum"),
    )
    team_zone["FGA_allowed_per_game"] = team_zone["FGA"] / team_zone["Games"]
    team_zone["FG_pct_allowed"] = team_zone["FGM"] / team_zone["FGA"].where(team_zone["FGA"] > 0)
    team_zone["PTS_allowed_per_game"] = team_zone["PTS"] / team_zone["Games"]

    league = complete.groupby("Zone", as_index=False).agg(
        Team_games=("Gamecode", lambda s: complete.loc[s.index, ["Gamecode", "DefTeam"]].drop_duplicates().shape[0]),
        League_FGA=("FGA", "sum"), League_FGM=("FGM", "sum"), League_PTS=("PTS", "sum"),
    )
    league["League_FGA_per_game"] = league["League_FGA"] / league["Team_games"]
    league["League_FG_pct"] = league["League_FGM"] / league["League_FGA"].where(league["League_FGA"] > 0)
    league["League_PTS_per_game"] = league["League_PTS"] / league["Team_games"]

    result = team_zone.merge(league, on="Zone", how="left")
    result["FGA_vs_league"] = result["FGA_allowed_per_game"] - result["League_FGA_per_game"]
    result["FG_pct_vs_league"] = result["FG_pct_allowed"] - result["League_FG_pct"]
    result["PTS_vs_league"] = result["PTS_allowed_per_game"] - result["League_PTS_per_game"]
    return result.sort_values(["DefTeam", "Zone"]).reset_index(drop=True)

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


def _prepare_zoned_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Prepare shots already classified by the app's existing zone() function.

    This module deliberately does not classify coordinates or create a second
    zone dataset. Pass the existing app output with its Zone column.
    """
    _require_columns(shots, ["Zone", "Result", "POINTS"])
    data = shots.copy()
    invalid = sorted(set(data["Zone"].dropna().unique()) - set(ZONES))
    if invalid:
        raise ValueError(f"Unknown shot zones in existing shot data: {invalid}")
    data = data.dropna(subset=["Zone"]).copy()
    data["FGM"] = data["Result"].eq("Made").astype(int)
    data["POINTS"] = pd.to_numeric(data["POINTS"], errors="coerce").fillna(0)
    return data


def player_zone_table(shots: pd.DataFrame) -> pd.DataFrame:
    """Player shooting profile by zone: volume share, accuracy and points/shot."""
    _require_columns(shots, ["ID_PLAYER", "Zone", "Result", "POINTS"])
    data = _prepare_zoned_shots(shots)
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
    _require_columns(shots, ["Gamecode", "TEAM", "Zone", "Result", "POINTS"])
    data = _prepare_zoned_shots(shots)
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



def position_defense_table(
    box: pd.DataFrame,
    position_map: dict[str, str],
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate opponent production allowed per 40 position-minutes.

    Position labels should come from the app's saved position map. Unknown
    players default to Wing, matching the current app's conservative fallback.
    Per-40 rates make teams comparable even when their position groups receive
    different amounts of court time.
    """
    stats = stats or ["PTS", "REB", "AST", "3PM"]
    _require_columns(box, ["Gamecode", "Team", "Player_ID", "MIN"] + stats)
    data = box.copy()
    data["MIN"] = pd.to_numeric(data["MIN"], errors="coerce").fillna(0)
    data = data.loc[data["MIN"] > 0].copy()
    if data.empty:
        return pd.DataFrame()

    if "Opp" not in data.columns:
        game_teams = data[["Gamecode", "Team"]].drop_duplicates()
        opponent_map = {}
        for gamecode, group in game_teams.groupby("Gamecode"):
            teams = group["Team"].dropna().unique().tolist()
            if len(teams) == 2:
                opponent_map[(gamecode, teams[0])] = teams[1]
                opponent_map[(gamecode, teams[1])] = teams[0]
        data["Opp"] = [
            opponent_map.get((gamecode, team))
            for gamecode, team in zip(data["Gamecode"], data["Team"])
        ]

    data = data.dropna(subset=["Opp"])
    data["Position"] = data["Player_ID"].astype(str).map(position_map).fillna("Wing")
    data.loc[~data["Position"].isin(["Guard", "Wing", "Big"]), "Position"] = "Wing"
    for stat in stats:
        data[stat] = pd.to_numeric(data[stat], errors="coerce").fillna(0)

    allowed = data.groupby(["Opp", "Position"], as_index=False).agg(
        Games=("Gamecode", "nunique"),
        Position_MIN=("MIN", "sum"),
        **{stat: (stat, "sum") for stat in stats},
    )
    league = data.groupby("Position").agg(
        League_MIN=("MIN", "sum"),
        **{f"{stat}_league_total": (stat, "sum") for stat in stats},
    ).reset_index()

    result = allowed.merge(league, on="Position", how="left")
    for stat in stats:
        result[f"{stat}_allowed_per40"] = (
            result[stat] * REGULATION_MINUTES / result["Position_MIN"]
        )
        result[f"{stat}_league_per40"] = (
            result[f"{stat}_league_total"] * REGULATION_MINUTES / result["League_MIN"]
        )
        result[f"{stat}_vs_league"] = (
            result[f"{stat}_allowed_per40"] - result[f"{stat}_league_per40"]
        )
    return result.sort_values(["Opp", "Position"]).reset_index(drop=True)



def _cap_and_scale_minutes(
    weights: pd.Series,
    total_minutes: float = 200.0,
    player_cap: float = 40.0,
) -> pd.Series:
    """Scale rotation weights to a team-minute total without exceeding 40."""
    if weights.empty:
        return weights.astype(float)
    if total_minutes < 0 or player_cap <= 0:
        raise ValueError("Minute totals must be non-negative and player_cap positive")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    if weights.sum() <= 0:
        weights[:] = 1.0

    allocation = pd.Series(0.0, index=weights.index)
    remaining = min(float(total_minutes), player_cap * len(weights))
    active = weights.index.tolist()
    while active and remaining > 1e-9:
        active_weights = weights.loc[active]
        if active_weights.sum() <= 0:
            active_weights = pd.Series(1.0, index=active)
        proposed = active_weights / active_weights.sum() * remaining
        capped = proposed[proposed >= player_cap - 1e-9]
        if capped.empty:
            allocation.loc[active] += proposed
            break
        for player_id in capped.index:
            allocation.loc[player_id] += player_cap
            remaining -= player_cap
            active.remove(player_id)
    return allocation


def expected_minutes_table(
    box: pd.DataFrame,
    team: str,
    unavailable_player_ids: set[str] | None = None,
    recent_games: int = 5,
    recent_weight: float = 0.65,
    team_minutes: float = 200.0,
) -> pd.DataFrame:
    """Estimate the next game's player minutes and redistribute absences.

    Recent and season-to-date team-game averages form each player's baseline.
    Players listed as unavailable receive zero; their rotation share is
    redistributed proportionally among the remaining roster, with a 40-minute
    regulation cap per player. Pass player IDs as strings.
    """
    _require_columns(box, ["Gamecode", "Team", "Player_ID", "Player", "MIN"])
    if recent_games < 1:
        raise ValueError("recent_games must be at least 1")
    if not 0 <= recent_weight <= 1:
        raise ValueError("recent_weight must be between 0 and 1")

    data = box.loc[box["Team"] == team].copy()
    if data.empty:
        return pd.DataFrame(columns=[
            "Player_ID", "Player", "Recent_MIN", "Season_MIN",
            "Baseline_MIN", "Projected_MIN", "Unavailable",
        ])
    data["Player_ID"] = data["Player_ID"].astype(str)
    data["MIN"] = pd.to_numeric(data["MIN"], errors="coerce").fillna(0).clip(lower=0)
    games = sorted(data["Gamecode"].dropna().unique())[-recent_games:]
    roster = (
        data.sort_values("Gamecode").groupby("Player_ID", as_index=False)
        .agg(Player=("Player", "last"))
    )
    pivot = data.pivot_table(
        index="Gamecode", columns="Player_ID", values="MIN", aggfunc="sum", fill_value=0
    ).reindex(index=games, columns=roster["Player_ID"], fill_value=0)
    first_game = data.groupby("Player_ID")["Gamecode"].min()
    recent_avg = pd.Series({
        player_id: pivot.loc[
            pivot.index >= first_game[player_id], player_id
        ].mean()
        for player_id in roster["Player_ID"]
    })
    season_pivot = data.pivot_table(
        index="Gamecode", columns="Player_ID", values="MIN",
        aggfunc="sum", fill_value=0,
    ).reindex(columns=roster["Player_ID"], fill_value=0)
    season_avg = pd.Series({
        player_id: season_pivot.loc[
            season_pivot.index >= first_game[player_id], player_id
        ].mean()
        for player_id in roster["Player_ID"]
    })
    result = roster.set_index("Player_ID")
    result["Recent_MIN"] = recent_avg.reindex(result.index).fillna(0)
    result["Season_MIN"] = season_avg.reindex(result.index).fillna(0)
    result["Baseline_MIN"] = (
        recent_weight * result["Recent_MIN"]
        + (1 - recent_weight) * result["Season_MIN"]
    )

    unavailable = {str(player_id) for player_id in (unavailable_player_ids or set())}
    result["Unavailable"] = result.index.isin(unavailable)
    available = result.loc[~result["Unavailable"], "Baseline_MIN"]
    allocation = _cap_and_scale_minutes(available, team_minutes, 40.0)
    result["Projected_MIN"] = 0.0
    result.loc[allocation.index, "Projected_MIN"] = allocation
    result.index.name = "Player_ID"
    return result.reset_index().sort_values("Projected_MIN", ascending=False).reset_index(drop=True)



def game_environment(
    team_games: pd.DataFrame,
    team: str,
    opponent: str,
    recent_games: int = 5,
    max_adjustment: float = 0.05,
) -> dict[str, float]:
    """Estimate a conservative pace and scoring environment from prior games.

    Call this with historical rows only (games before the game being projected)
    when backtesting. Both teams' recent pace is blended; opponent DRtg adjusts
    scoring modestly and is not applied automatically to non-scoring stats.
    """
    _require_columns(team_games, ["Gamecode", "Team", "Pace_per40", "DRtg"])
    if recent_games < 1 or not 0 <= max_adjustment < 1:
        raise ValueError("Invalid recent_games or max_adjustment")

    league_pace = float(team_games["Pace_per40"].mean())
    league_drtg = float(team_games["DRtg"].mean())
    if league_pace <= 0 or league_drtg <= 0:
        return {
            "expected_pace": league_pace,
            "pace_factor": 1.0,
            "opponent_drtg": league_drtg,
            "scoring_factor": 1.0,
        }

    def recent_mean(team_name: str, column: str, fallback: float) -> float:
        subset = team_games.loc[team_games["Team"] == team_name].sort_values("Gamecode")
        values = subset[column].dropna().tail(recent_games)
        return float(values.mean()) if not values.empty else fallback

    team_pace = recent_mean(team, "Pace_per40", league_pace)
    opponent_pace = recent_mean(opponent, "Pace_per40", league_pace)
    opponent_drtg = recent_mean(opponent, "DRtg", league_drtg)
    expected_pace = (team_pace + opponent_pace) / 2.0
    pace_factor = expected_pace / league_pace
    scoring_factor = opponent_drtg / league_drtg

    return {
        "expected_pace": expected_pace,
        "pace_factor": float(max(1 - max_adjustment, min(1 + max_adjustment, pace_factor))),
        "opponent_drtg": opponent_drtg,
        "scoring_factor": float(max(1 - max_adjustment, min(1 + max_adjustment, scoring_factor))),
    }


def project_player_box_score(
    rates: pd.DataFrame,
    minutes: pd.DataFrame,
    pace_factor: float = 1.0,
    scoring_factor: float = 1.0,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Convert shrunk per-40 rates and projected minutes into box-score means.

    Pace scales counting-stat volume. The optional opponent scoring factor is
    applied to points only; zone-based shot efficiency and prop-specific
    adjustments can be layered on later without double-counting defense.
    """
    stats = stats or ["PTS", "REB", "AST", "3PM", "STL", "BLK", "TO"]
    _require_columns(rates, ["Player_ID"] + [f"{stat}_per40_shrunk" for stat in stats])
    _require_columns(minutes, ["Player_ID", "Projected_MIN"])
    result = minutes.merge(rates, on="Player_ID", how="left")
    result["Projected_MIN"] = pd.to_numeric(
        result["Projected_MIN"], errors="coerce"
    ).fillna(0)
    for stat in stats:
        rate = pd.to_numeric(result[f"{stat}_per40_shrunk"], errors="coerce").fillna(0)
        projection = rate * result["Projected_MIN"] / REGULATION_MINUTES * pace_factor
        if stat == "PTS":
            projection = projection * scoring_factor
        result[f"Proj_{stat}"] = projection
    return result



def walk_forward_backtest(
    box: pd.DataFrame,
    min_train_games: int = 3,
    recent_games: int = 5,
    stats: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest projections chronologically without using future game data.

    For each target game, every feature is rebuilt from games with a lower
    Gamecode. This is slower than a random split, but avoids look-ahead leakage.
    Returns (player-game predictions, summary metrics).
    """
    stats = stats or ["PTS", "REB", "AST", "3PM"]
    _require_columns(
        box,
        ["Gamecode", "Team", "Player_ID", "Player", "MIN"] + stats
        + ["FGA", "OREB", "TO", "FTA"],
    )
    data = box.copy()
    data["Player_ID"] = data["Player_ID"].astype(str)
    gamecodes = sorted(data["Gamecode"].dropna().unique())
    records = []

    for gamecode in gamecodes:
        history = data.loc[data["Gamecode"] < gamecode].copy()
        target = data.loc[data["Gamecode"] == gamecode].copy()
        teams = target["Team"].dropna().unique().tolist()
        if len(teams) != 2:
            continue
        if any(
            history.loc[history["Team"] == team, "Gamecode"].nunique() < min_train_games
            for team in teams
        ):
            continue

        rates = player_rate_table(history)
        historical_team_games = team_game_table(history)
        if rates.empty or historical_team_games.empty:
            continue

        for team in teams:
            opponent = teams[1] if teams[0] == team else teams[0]
            minutes = expected_minutes_table(
                history, team, recent_games=recent_games
            )
            if minutes.empty:
                continue
            environment = game_environment(
                historical_team_games, team, opponent, recent_games=recent_games
            )
            projected = project_player_box_score(
                rates,
                minutes,
                pace_factor=environment["pace_factor"],
                scoring_factor=environment["scoring_factor"],
                stats=stats,
            )
            actual = target.loc[target["Team"] == team, ["Player_ID", "MIN"] + stats].copy()
            actual["MIN"] = pd.to_numeric(actual["MIN"], errors="coerce").fillna(0)
            for stat in stats:
                actual[stat] = pd.to_numeric(actual[stat], errors="coerce").fillna(0)
            actual = actual.rename(
                columns={"MIN": "Actual_MIN", **{stat: f"Actual_{stat}" for stat in stats}}
            )
            joined = projected.merge(actual, on="Player_ID", how="inner")
            if joined.empty:
                continue
            joined["Gamecode"] = gamecode
            joined["Team"] = team
            joined["Opponent"] = opponent
            joined["Expected_pace"] = environment["expected_pace"]
            joined["Pace_factor"] = environment["pace_factor"]
            joined["Scoring_factor"] = environment["scoring_factor"]
            joined["Error_MIN"] = joined["Projected_MIN"] - joined["Actual_MIN"]
            for stat in stats:
                joined[f"Error_{stat}"] = joined[f"Proj_{stat}"] - joined[f"Actual_{stat}"]
            records.append(joined)

    if not records:
        empty = pd.DataFrame()
        return empty, empty
    predictions = pd.concat(records, ignore_index=True)
    metric_rows = []
    for stat in ["MIN"] + stats:
        errors = predictions[f"Error_{stat}"]
        metric_rows.append({
            "Stat": stat,
            "N": int(errors.notna().sum()),
            "MAE": float(errors.abs().mean()),
            "RMSE": float((errors.pow(2).mean()) ** 0.5),
            "Bias": float(errors.mean()),
        })
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics



def zone_points_projection(
    shots: pd.DataFrame,
    player_id: str,
    opponent_code: str | None,
    projected_fga: float,
    projected_ftm: float,
    prior_attempts: float = 30.0,
    defense_prior_games: float = 5.0,
) -> dict:
    """Project field-goal points from player zone mix and opponent zone defense.

    Player zone shares and points-per-attempt are shrunk toward league rates.
    Opponent zone volume and accuracy effects are also shrunk and capped, so a
    small sample cannot create an extreme adjustment. Free throws are added
    separately from the box-score FTM projection.
    """
    if projected_fga < 0 or projected_ftm < 0 or prior_attempts < 0:
        raise ValueError("Attempts, free throws and prior must be non-negative")
    _require_columns(
        shots, ["ID_PLAYER", "TEAM", "Gamecode", "Zone", "Result", "POINTS"]
    )
    data = _prepare_zoned_shots(shots)
    if data.empty:
        return {
            "Projected_FG_PTS": 0.0,
            "Projected_FT_PTS": float(projected_ftm),
            "Projected_PTS": float(projected_ftm),
            "zones": pd.DataFrame(),
        }

    league = data.groupby("Zone").agg(
        FGA=("Result", "size"), FGM=("FGM", "sum"), PTS=("POINTS", "sum")
    ).reindex(ZONES).fillna(0)
    league_total = float(league["FGA"].sum())
    if league_total <= 0:
        return {
            "Projected_FG_PTS": 0.0,
            "Projected_FT_PTS": float(projected_ftm),
            "Projected_PTS": float(projected_ftm),
            "zones": pd.DataFrame(),
        }
    league["Share"] = league["FGA"] / league_total
    league["PPP"] = league["PTS"] / league["FGA"].where(league["FGA"] > 0)
    league["Points_per_make"] = (
        league["PTS"] / league["FGM"].where(league["FGM"] > 0)
    )

    player = data.loc[data["ID_PLAYER"].astype(str) == str(player_id)]
    player_zone = player.groupby("Zone").agg(
        FGA=("Result", "size"), FGM=("FGM", "sum"), PTS=("POINTS", "sum")
    ).reindex(ZONES).fillna(0)
    player_attempts = float(player_zone["FGA"].sum())

    defense = opponent_zone_defense(data)
    if not defense.empty and opponent_code is not None:
        defense = defense.loc[defense["DefTeam"].astype(str) == str(opponent_code)].set_index("Zone")
    else:
        defense = pd.DataFrame()

    rows = []
    for zone_name in ZONES:
        lg = league.loc[zone_name]
        pl = player_zone.loc[zone_name]
        league_share = float(lg["Share"])
        league_ppp = float(lg["PPP"]) if pd.notna(lg["PPP"]) else 0.0
        zone_fga = float(pl["FGA"])
        zone_pts = float(pl["PTS"])
        shrunk_share = (
            (zone_fga + prior_attempts * league_share)
            / (player_attempts + prior_attempts)
            if player_attempts + prior_attempts > 0 else league_share
        )
        shrunk_ppp = (
            (zone_pts + prior_attempts * league_ppp) / (zone_fga + prior_attempts)
            if zone_fga + prior_attempts > 0 else league_ppp
        )
        points_per_make = (
            float(lg["Points_per_make"])
            if pd.notna(lg["Points_per_make"]) else 2.0
        )

        volume_factor, efficiency_delta = 1.0, 0.0
        if not defense.empty and zone_name in defense.index:
            d = defense.loc[zone_name]
            games = float(d["Games"])
            reliability = games / (games + defense_prior_games) if games + defense_prior_games > 0 else 0
            league_fga = float(d["League_FGA_per_game"])
            if league_fga > 0:
                volume_ratio = float(d["FGA_allowed_per_game"]) / league_fga
                volume_factor = max(0.95, min(1.05, 1 + reliability * (volume_ratio - 1)))
            if pd.notna(d["FG_pct_vs_league"]):
                efficiency_delta = max(
                    -0.05, min(0.05, reliability * float(d["FG_pct_vs_league"]))
                )

        adjusted_ppp = max(0.0, shrunk_ppp + efficiency_delta * points_per_make)
        zone_fga_projection = projected_fga * shrunk_share * volume_factor
        rows.append({
            "Zone": zone_name,
            "FGA_share": shrunk_share,
            "PPP_shrunk": shrunk_ppp,
            "Opponent_volume_factor": volume_factor,
            "Opponent_efficiency_delta": efficiency_delta,
            "FG_pct_shrunk": (
                (float(pl["FGM"]) + prior_attempts * (
                    float(lg["FGM"]) / float(lg["FGA"]) if float(lg["FGA"]) > 0 else 0.0
                )) / (zone_fga + prior_attempts)
                if zone_fga + prior_attempts > 0 else 0.0
            ),
            "Projected_FGA": zone_fga_projection,
            "Projected_FGM": zone_fga_projection * max(
                0.0,
                min(1.0, (
                    (float(pl["FGM"]) + prior_attempts * (
                        float(lg["FGM"]) / float(lg["FGA"]) if float(lg["FGA"]) > 0 else 0.0
                    )) / (zone_fga + prior_attempts)
                    if zone_fga + prior_attempts > 0 else 0.0
                ) + efficiency_delta),
            ),
            "Projected_PTS": zone_fga_projection * adjusted_ppp,
        })

    zone_result = pd.DataFrame(rows)
    fg_points = float(zone_result["Projected_PTS"].sum())
    fg_makes = float(zone_result["Projected_FGM"].sum())
    three_makes = float(zone_result.loc[
        zone_result["Zone"].isin(["Corner 3", "Above-the-break 3"]), "Projected_FGM"
    ].sum())
    ft_points = float(projected_ftm)
    return {
        "Projected_FG_PTS": fg_points,
        "Projected_FGM": fg_makes,
        "Projected_3PM": three_makes,
        "Projected_FT_PTS": ft_points,
        "Projected_PTS": fg_points + ft_points,
        "zones": zone_result,
    }



def position_adjustment_factors(
    position_defense: pd.DataFrame,
    opponent: str,
    position: str,
    stats: list[str] | None = None,
    max_adjustment: float = 0.05,
    prior_games: float = 5.0,
) -> dict[str, float]:
    """Return small, sample-shrunk multipliers for a player's position matchup."""
    stats = stats or ["PTS", "REB", "AST"]
    if position_defense.empty:
        return {stat: 1.0 for stat in stats}
    row = position_defense.loc[
        (position_defense["Opp"].astype(str) == str(opponent))
        & (position_defense["Position"] == position)
    ]
    if row.empty:
        return {stat: 1.0 for stat in stats}

    sample = row.iloc[0]
    games = float(sample["Games"])
    reliability = games / (games + prior_games) if games + prior_games > 0 else 0.0
    factors = {}
    for stat in stats:
        allowed = sample.get(f"{stat}_allowed_per40")
        league = sample.get(f"{stat}_league_per40")
        if pd.isna(allowed) or pd.isna(league) or float(league) <= 0:
            factors[stat] = 1.0
            continue
        raw_ratio = float(allowed) / float(league)
        shrunk_ratio = 1.0 + reliability * (raw_ratio - 1.0)
        factors[stat] = float(max(1 - max_adjustment, min(1 + max_adjustment, shrunk_ratio)))
    return factors

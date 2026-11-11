"""Euroleague prop research tool (v3).

Run:  streamlit run euroleague_props.py
Needs: data_cache/shots_2026.parquet (your existing file).
Box scores are downloaded once via the sidebar button and cached to
data_cache/boxscores_2026.parquet (only new games are fetched afterwards).

New in v2:
  * Next opponent is picked automatically from the schedule
  * Defence vs league average by zone (Opponent defence tab)
  * Odds / edge box (Game log tab)

New in v3:
  * Defence vs position: points / rebounds / assists conceded to Guards, Wings and Bigs
  * Positions tab to fix a player's position (saved in data_cache/positions.csv)
"""
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.schedule import Schedule
from matplotlib.patches import Arc, Circle, Rectangle

SEASON = 2026
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)
BOX_PATH = os.path.join(DATA_DIR, f"boxscores_{SEASON}.parquet")
SHOT_PATH = os.path.join(DATA_DIR, f"shots_{SEASON}.parquet")
BG, GREEN, RED = "#0e1117", "#1de28b", "#e8452c"

# If a team code in the schedule differs from the code in your shots file,
# add it here as  "schedule code": "shots code"  (the app warns you if needed).
CODE_ALIASES = {}

# API column -> short name. If the API names differ, clean_box() tells you.
RENAME = {"Points": "PTS", "TotalRebounds": "REB", "Assistances": "AST",
          "FieldGoalsMade3": "3PM", "Steals": "STL", "BlocksFavour": "BLK",
          "Turnovers": "TO", "Minutes": "MIN"}
BASE = ["PTS", "REB", "AST", "3PM", "STL", "BLK", "TO"]
COMBOS = {"PRA": ["PTS", "REB", "AST"], "PR": ["PTS", "REB"],
          "PA": ["PTS", "AST"], "RA": ["REB", "AST"]}
STAT_OPTIONS = ["PTS", "REB", "AST", "3PM", "STL", "BLK", "PRA", "PR", "PA", "RA"]
ZONES = ["Rim", "Paint", "Midrange", "Corner 3", "Above-the-break 3"]

# Positions: the box score has none, so we guess Guard / Wing / Big from
# (rebounds - assists) per 36 minutes. Lower the numbers to get more Guards,
# raise BIG_MIN to get fewer Bigs. You can always fix players in the Positions tab.
POS_PATH = os.path.join(DATA_DIR, "positions.csv")
POS_OPTIONS = ["Guard", "Wing", "Big"]
GUARD_MAX = 0.5   # (REB - AST) per 36 at or below this -> Guard
BIG_MIN = 4.5     # at or above this -> Big


# ---------------------------------------------------------------- box scores
def to_minutes(v):
    if isinstance(v, (int, float)):
        return 0.0 if pd.isna(v) else float(v)
    s = str(v).strip()
    if ":" in s:
        try:
            m, sec = s.split(":")
            return int(m) + int(sec) / 60
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0  # "DNP" etc.


def fetch_box(gamecode, retries=5):
    api, delay = BoxScoreData("E"), 2.0
    for _ in range(retries):
        try:
            return api.get_players_boxscore_stats(SEASON, gamecode)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            return None
        except Exception:
            return None
    return None


def clean_box(raw):
    df = raw[~raw["Player"].isin(["Team", "Total"])].copy()
    missing = [c for c in list(RENAME) + ["Player_ID", "Team", "Home"] if c not in df.columns]
    if missing:
        raise ValueError(f"Box score columns missing {missing}. Actual columns: {list(df.columns)}")
    df = df.rename(columns=RENAME)
    df["MIN"] = df["MIN"].apply(to_minutes)
    for c in BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["Player_ID"] = df["Player_ID"].astype(str).str.strip()
    return df[["Gamecode", "Home", "Team", "Player_ID", "Player", "MIN"] + BASE]


def update_box(latest):
    old = pd.read_parquet(BOX_PATH) if os.path.exists(BOX_PATH) else pd.DataFrame()
    have = set(old["Gamecode"]) if len(old) else set()
    todo = [g for g in range(1, latest + 1) if g not in have]
    if not todo:
        st.sidebar.success("Already up to date.")
        return
    bar, new, failed = st.sidebar.progress(0.0), [], []
    for i, g in enumerate(todo):
        raw = fetch_box(g)
        if raw is not None and len(raw):
            try:
                new.append(clean_box(raw))
            except ValueError as e:
                st.sidebar.error(str(e))
                break
        else:
            failed.append(g)
        bar.progress((i + 1) / len(todo))
        time.sleep(1.0)
    if new:
        pd.concat(([old] if len(old) else []) + new, ignore_index=True).to_parquet(BOX_PATH)
    if failed:
        st.sidebar.warning(f"No data for games: {failed}")


def enrich(box, shots):
    box = box.copy()
    opp = {}
    for gc, teams in box.groupby("Gamecode")["Team"].unique().items():
        if len(teams) == 2:
            opp[(gc, teams[0])], opp[(gc, teams[1])] = teams[1], teams[0]
    box["Opp"] = [opp.get((g, t)) for g, t in zip(box["Gamecode"], box["Team"])]
    for k, cols in COMBOS.items():
        box[k] = box[cols].sum(axis=1)
    # team name -> shot-data team code, learned from player IDs present in both
    m = box.merge(shots[["Gamecode", "ID_PLAYER", "TEAM"]].drop_duplicates(),
                  left_on=["Gamecode", "Player_ID"], right_on=["Gamecode", "ID_PLAYER"])
    code = m.groupby("Team")["TEAM"].agg(lambda s: s.mode().iat[0]).to_dict()
    box["Code"], box["OppCode"] = box["Team"].map(code), box["Opp"].map(code)
    dates = shots.groupby("Gamecode")["UTC"].min().astype(str).str[:8]
    box["Date"] = box["Gamecode"].map(lambda g: f"{dates[g][6:8]}/{dates[g][4:6]}" if g in dates else "")
    box["Label"] = (box["Date"] + "\n" + box["Home"].map({1: "vs ", 0: "@ "})
                    + box["OppCode"].fillna(box["Opp"].astype(str).str[:3]))
    return box


def team_defense(box):
    tot = box.groupby(["Gamecode", "Team", "Opp"])[BASE].sum().reset_index()
    allowed = tot.groupby("Opp")[["PTS", "REB", "AST", "3PM"]].mean().round(1)
    allowed.columns = [f"{c} allowed" for c in allowed.columns]
    for c in list(allowed.columns):  # rank 1 = concedes the most (best for overs)
        allowed[c.replace("allowed", "rank")] = allowed[c].rank(ascending=False, method="min").astype(int)
    return allowed.sort_values("PTS allowed", ascending=False)


def team_code(box, team):
    s = box.loc[box["Team"] == team, "Code"].dropna()
    return s.iat[0] if len(s) else None


# ---------------------------------------------------------------- positions
def auto_positions(box):
    """Rough Guard / Wing / Big guess from rebounds vs assists per 36 minutes."""
    b = box[box["MIN"] > 0].sort_values("Gamecode")
    g = b.groupby("Player_ID").agg(Player=("Player", "last"), Team=("Team", "last"),
                                   MIN=("MIN", "sum"), REB=("REB", "sum"),
                                   AST=("AST", "sum")).reset_index()
    d = (g["REB"] - g["AST"]) * 36 / g["MIN"]
    g["Auto"] = "Wing"
    g.loc[d <= GUARD_MAX, "Auto"] = "Guard"
    g.loc[d >= BIG_MIN, "Auto"] = "Big"
    g.loc[g["MIN"] < 50, "Auto"] = "Wing"  # too few minutes to judge
    return g


def load_positions(box):
    """Returns (table, {Player_ID: position}). Your saved choices in positions.csv always win;
    new players get the automatic guess and are added to the file."""
    auto = auto_positions(box)
    old, saved = None, {}
    if os.path.exists(POS_PATH):
        try:
            old = pd.read_csv(POS_PATH, dtype=str)
            saved = {r.Player_ID: r.Position for r in old.itertuples() if r.Position in POS_OPTIONS}
        except Exception:
            old, saved = None, {}
    auto["Position"] = [saved.get(i, a) for i, a in zip(auto["Player_ID"], auto["Auto"])]
    out = (auto[["Player_ID", "Player", "Team", "Position"]]
           .sort_values(["Team", "Player"]).reset_index(drop=True))
    try:
        same = old is not None and old[list(out.columns)].reset_index(drop=True).equals(out)
    except Exception:
        same = False
    if not same:
        try:
            out.to_csv(POS_PATH, index=False)
        except OSError:
            pass  # e.g. file open in Excel; the app still works from memory
    return out, dict(zip(out["Player_ID"], out["Position"]))


def defence_vs_position(box, pos_map, defence, games):
    """Points / rebounds / assists per game that `defence` concedes to each position,
    next to the league average. Rank 1 = concedes the most (season-wide)."""
    b = box[(box["MIN"] > 0) & box["Opp"].notna()].copy()
    b["Pos"] = b["Player_ID"].map(pos_map).fillna("Wing")
    cols, n_teams = {}, 0
    for s in ("PTS", "REB", "AST"):
        t = (b.pivot_table(index=["Gamecode", "Opp"], columns="Pos", values=s,
                           aggfunc="sum", fill_value=0)
               .reindex(columns=POS_OPTIONS, fill_value=0))
        team_avg = t.groupby(level="Opp").mean()
        if defence not in team_avg.index:
            return None
        mask = ((t.index.get_level_values("Opp") == defence)
                & t.index.get_level_values("Gamecode").isin(list(games)))
        mine = t[mask]
        if mine.empty:
            return None
        n_teams = len(team_avg)
        cols[f"{s}/g"] = mine.mean()
        cols[f"{s} lg"] = t.mean()
        cols[f"{s} rank"] = team_avg.rank(ascending=False, method="min").loc[defence]
    out = pd.DataFrame(cols)
    out.index.name = "Position"
    return {"table": out.reset_index(), "n_teams": n_teams}


def dvp_display(dvp):
    t, n = dvp["table"], dvp["n_teams"]
    d = pd.DataFrame({"Position": t["Position"]})
    for s in ("PTS", "REB", "AST"):
        d[f"{s}/g"] = t[f"{s}/g"].round(1)
        d[f"{s} vs lg"] = (t[f"{s}/g"] - t[f"{s} lg"]).round(1)
        d[f"{s} rank"] = t[f"{s} rank"].astype(int).astype(str) + f"/{n}"
    return d


# ---------------------------------------------------------------- schedule
@st.cache_data(ttl=3600)
def load_schedule():
    """Season schedule. Returns None if the API call fails."""
    try:
        s = Schedule("E").get_schedule(SEASON).copy()
    except Exception:
        return None
    s["played"] = s["played"].astype(str).str.lower().eq("true")
    s["when"] = pd.to_datetime(s["date"].astype(str) + " " + s["startime"].astype(str),
                               format="%b %d, %Y %H:%M", errors="coerce")
    for c in ("homecode", "awaycode"):
        s[c] = s[c].map(lambda x: CODE_ALIASES.get(x, x))
    return s


def next_game(sched, code):
    """Next unplayed game for a team code -> dict, or None."""
    if sched is None or code is None:
        return None
    g = sched[(~sched["played"]) & ((sched["homecode"] == code) | (sched["awaycode"] == code))]
    g = g.sort_values("when")
    if g.empty:
        return None
    r = g.iloc[0]
    home = r["homecode"] == code
    return {"opp_code": r["awaycode"] if home else r["homecode"],
            "home": bool(home), "when": r["when"]}


# ---------------------------------------------------------------- shots
def zone(df):
    df = df.copy()
    x, y = df["COORD_X"], df["COORD_Y"]
    d = (x**2 + y**2) ** 0.5
    rim = d <= 100
    paint = (x.abs() <= 245) & (y <= 430) & ~rim
    corner = (x.abs() >= 660) & (y <= 142)
    atb = (d >= 675) & ~corner
    df["Zone"] = "Midrange"
    for name, mask in [("Rim", rim), ("Paint", paint), ("Corner 3", corner), ("Above-the-break 3", atb)]:
        df.loc[mask, "Zone"] = name
    return df


def profile(shots):
    s = zone(shots)
    g = s.groupby("Zone").agg(FGA=("Result", "size"), FGM=("Result", lambda r: (r == "Made").sum()),
                              PTS=("POINTS", "sum")).reindex(ZONES).fillna(0)
    g["FG%"] = (g["FGM"] / g["FGA"] * 100).round(1)
    g["% of points"] = (g["PTS"] / g["PTS"].sum() * 100).round(1) if g["PTS"].sum() else 0
    return g.reset_index().fillna(0)


def _zone_totals(df):
    return df.groupby("Zone").agg(FGA=("Result", "size"),
                                  FGM=("Result", lambda r: (r == "Made").sum()),
                                  PTS=("POINTS", "sum")).reindex(ZONES).fillna(0)


def zone_vs_league(real, opp_code, opp_games):
    """Shots the opponent concedes per game, by zone, next to the league average.
    'PTS/g vs league' above 0 = the opponent gives up MORE than average there."""
    z = zone(real)
    league_games = z["Gamecode"].nunique()
    conceded = z[z["Gamecode"].isin(opp_games) & (z["TEAM"] != opp_code)]
    n = conceded["Gamecode"].nunique()
    if n == 0 or league_games == 0:
        return None
    o, lg = _zone_totals(conceded), _zone_totals(z)
    nan = float("nan")
    out = pd.DataFrame(index=ZONES)
    out["Opp FGA/g"] = o["FGA"] / n
    out["League FGA/g"] = lg["FGA"] / (2 * league_games)
    out["Opp FG%"] = o["FGM"] / o["FGA"].replace(0, nan) * 100
    out["League FG%"] = lg["FGM"] / lg["FGA"].replace(0, nan) * 100
    out["Opp PTS/g"] = o["PTS"] / n
    out["League PTS/g"] = lg["PTS"] / (2 * league_games)
    out["PTS/g vs league"] = out["Opp PTS/g"] - out["League PTS/g"]
    out = out.round(1)
    out.index.name = "Zone"
    return out.reset_index()


def draw_court(ax):
    c, lw = "white", 1.2
    ax.set_facecolor(BG)
    ax.add_patch(Rectangle((-750, -150), 1500, 1550, fill=False, ec=c, lw=lw))
    ax.add_patch(Circle((0, 0), 22.5, fill=False, ec=c, lw=lw))
    ax.plot([-90, 90], [-15, -15], c=c, lw=lw)
    ax.add_patch(Rectangle((-245, -150), 490, 580, fill=False, ec=c, lw=lw))
    ax.add_patch(Arc((0, 430), 360, 360, theta1=0, theta2=180, ec=c, lw=lw))
    ax.add_patch(Arc((0, 0), 250, 250, theta1=0, theta2=180, ec=c, lw=lw))
    ax.plot([-660, -660], [-150, 141.5], c=c, lw=lw)
    ax.plot([660, 660], [-150, 141.5], c=c, lw=lw)
    ax.add_patch(Arc((0, 0), 1350, 1350, theta1=12.08, theta2=167.92, ec=c, lw=lw))
    ax.set_xlim(-800, 800)
    ax.set_ylim(-200, 900)
    ax.set_aspect("equal")
    ax.axis("off")


def shot_chart(shots, title):
    fig, ax = plt.subplots(figsize=(6, 5.6))
    fig.patch.set_facecolor(BG)
    draw_court(ax)
    for res, col in [("Made", GREEN), ("Missed", RED)]:
        s = shots[shots["Result"] == res]
        ax.scatter(s["COORD_X"], s["COORD_Y"], c=col, s=30, alpha=0.7, edgecolors="none", label=res)
    ax.set_title(title, color="white", fontsize=12)
    leg = ax.legend(loc="upper right", facecolor=BG, edgecolor="white")
    for t in leg.get_texts():
        t.set_color("white")
    return fig


def bar_chart(df, stat, line, title):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    vals = df[stat].values
    ax.bar(range(len(df)), vals, color=[GREEN if v > line else RED for v in vals], width=0.7)
    ax.axhline(line, color="white", ls="--", lw=1)
    top = max(vals.max(), line, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.015, f"{v:g}", ha="center", color="white", fontsize=10, weight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Label"], color="white", fontsize=8)
    ax.set_ylim(0, top * 1.12)
    ax.tick_params(axis="y", colors="white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{title}  (line {line:g})", color="white", loc="left")
    return fig


# ---------------------------------------------------------------- app
st.set_page_config(page_title="Euroleague Props", layout="wide")
st.title("Euroleague Prop Research")

if not os.path.exists(SHOT_PATH):
    st.error(f"Put your shots file at {SHOT_PATH}")
    st.stop()
shots = pd.read_parquet(SHOT_PATH)

st.sidebar.header("Data")
latest = st.sidebar.number_input("Latest game code", 1, 400, int(shots["Gamecode"].max()))
if st.sidebar.button("Fetch / update box scores"):
    update_box(int(latest))
if not os.path.exists(BOX_PATH):
    st.info("Click 'Fetch / update box scores' in the sidebar (first run takes a few minutes).")
    st.stop()

box = enrich(pd.read_parquet(BOX_PATH), shots)
teams = sorted(box["Team"].unique())
sched = load_schedule()
pos_df, pos_map = load_positions(box)

# warn once if schedule codes don't line up with shot-data codes
if sched is not None:
    unmatched = set(box["Code"].dropna()) - (set(sched["homecode"]) | set(sched["awaycode"]))
    if unmatched:
        st.sidebar.warning(f"Schedule codes don't match for: {sorted(unmatched)}. "
                           "Add them to CODE_ALIASES at the top of the file.")
elif sched is None:
    st.sidebar.warning("Couldn't load the schedule; pick the opponent by hand.")

st.sidebar.header("Player")
team = st.sidebar.selectbox("Team", teams)
roster = box[box["Team"] == team].groupby(["Player_ID", "Player"])["MIN"].sum().reset_index()
roster = roster[roster["MIN"] > 0].sort_values("MIN", ascending=False)
pname = st.sidebar.selectbox("Player", roster["Player"].tolist())
pid = roster.loc[roster["Player"] == pname, "Player_ID"].iat[0]
st.sidebar.caption(f"Position: {pos_map.get(pid, '?')} (change it in the Positions tab)")
stat = st.sidebar.selectbox("Stat", STAT_OPTIONS)

mine = box[(box["Player_ID"] == pid) & (box["MIN"] > 0)].sort_values("Gamecode")
default_line = round(mine[stat].mean() * 2) / 2 if len(mine) else 10.0
line = st.sidebar.number_input("Line", 0.0, 100.0, float(default_line), 0.5, key=f"line_{pid}_{stat}")

st.sidebar.header("Filters")
n_opt = st.sidebar.radio("Games", ["L5", "L10", "L20", "All"], index=1, horizontal=True)
venue = st.sidebar.radio("Venue", ["All", "Home", "Away"], horizontal=True)

# default opponent = this team's next scheduled game
nxt = next_game(sched, team_code(box, team))
code2name = (box.dropna(subset=["Code"]).drop_duplicates("Code").set_index("Code")["Team"].to_dict())
opp_options = [t for t in teams if t != team]
default_opp = code2name.get(nxt["opp_code"]) if nxt else None
opp_idx = opp_options.index(default_opp) if default_opp in opp_options else 0
opp_view = st.sidebar.selectbox("Opponent", opp_options, index=opp_idx, key=f"opp_{team}")
only_h2h = st.sidebar.checkbox("Only games vs this opponent")

tm_mode = st.sidebar.radio("Teammate", ["Any", "With", "Without"], horizontal=True)
tm_name = None
if tm_mode != "Any":
    mates = roster[roster["Player_ID"] != pid]["Player"].tolist()
    tm_name = st.sidebar.selectbox("Teammate", mates)

# apply filters (hit rates are computed on the filtered pool; L5/L10/L20 slice it)
pool = mine.copy()
if venue != "All":
    pool = pool[pool["Home"] == (1 if venue == "Home" else 0)]
if only_h2h:
    pool = pool[pool["Opp"] == opp_view]
if tm_name:
    tid = roster.loc[roster["Player"] == tm_name, "Player_ID"].iat[0]
    tm_played = set(box[(box["Player_ID"] == tid) & (box["MIN"] > 0)]["Gamecode"])
    pool = pool[pool["Gamecode"].isin(tm_played) if tm_mode == "With" else ~pool["Gamecode"].isin(tm_played)]
if pool.empty:
    st.warning("No games match these filters.")
    st.stop()
sel = pool if n_opt == "All" else pool.tail(int(n_opt[1:]))

if nxt:
    st.info(f"**{team}** next game: {'vs' if nxt['home'] else '@'} "
            f"**{code2name.get(nxt['opp_code'], nxt['opp_code'])}** · {nxt['when']:%a %d %b, %H:%M}")

opp_games = box[box["Team"] == opp_view]["Gamecode"].unique()
opp_sel = sorted(opp_games)[-int(n_opt[1:]):] if n_opt != "All" else sorted(opp_games)
opp_code = team_code(box, opp_view)
dvp = defence_vs_position(box, pos_map, opp_view, opp_sel)

tab1, tab2, tab3, tab4 = st.tabs(["Game log", "Shot maps", "Opponent defence", "Positions"])

with tab1:
    st.subheader(f"{pname} — {stat} over {line:g}")
    cols = st.columns(4)
    for col, (lab, k) in zip(cols, [("L5", 5), ("L10", 10), ("L20", 20), ("All", len(pool))]):
        part = pool.tail(k)
        col.metric(lab, f"{(part[stat] > line).mean() * 100:.0f}%", f"{(part[stat] > line).sum()} of {len(part)}",
                   delta_color="off")
    st.pyplot(bar_chart(sel, stat, line, f"{pname} — {stat}"))
    st.caption(f"{len(sel)} games shown · {len(pool)} match filters · {len(mine)} played this season. "
               "Small samples (esp. with/without) are noisy.")
    avg = sel[["MIN", "PTS", "REB", "AST", "3PM"]].mean().round(1)
    st.write("**Averages (shown games):** " + " · ".join(f"{k} {v}" for k, v in avg.items()))

    my_pos = pos_map.get(pid)
    parts = [stat] if stat in ("PTS", "REB", "AST") else COMBOS.get(stat)
    if dvp is not None and parts and my_pos in POS_OPTIONS:
        row = dvp["table"].set_index("Position").loc[my_pos]
        got = sum(row[f"{c}/g"] for c in parts)
        lg = sum(row[f"{c} lg"] for c in parts)
        st.write(f"**{opp_view} vs {my_pos}s:** they concede {got:.1f} {stat} per game to that position "
                 f"(league average {lg:.1f}, {got - lg:+.1f}).")

    # ---- odds / edge box
    st.markdown("#### Odds / edge")
    hits, n_sel = int((sel[stat] > line).sum()), len(sel)
    hit_rate = hits / n_sel
    c1, c2, c3 = st.columns(3)
    over_odds = c1.number_input("Over odds (decimal)", 1.01, 20.0, 1.90, 0.01, key=f"oo_{pid}_{stat}")
    under_odds = c2.number_input("Under odds (decimal)", 1.01, 20.0, 1.90, 0.01, key=f"uo_{pid}_{stat}")
    est_key = f"est_{pid}_{stat}_{line}_{n_opt}_{venue}_{only_h2h}_{tm_mode}_{tm_name}_{opp_view}"
    est = c3.slider("Your chance of the over (%)", 0, 100, int(round(hit_rate * 100)), key=est_key) / 100
    p_over, p_under = 1 / over_odds, 1 / under_odds
    fair_over = p_over / (p_over + p_under)  # bookmaker's margin removed
    fair_under = 1 - fair_over
    e1, e2 = st.columns(2)
    e1.metric(f"OVER {line:g} @ {over_odds:.2f}",
              f"{(est - fair_over) * 100:+.1f}% edge",
              f"EV {(est * over_odds - 1) * 100:+.1f}% per unit", delta_color="off")
    e2.metric(f"UNDER {line:g} @ {under_odds:.2f}",
              f"{((1 - est) - fair_under) * 100:+.1f}% edge",
              f"EV {((1 - est) * under_odds - 1) * 100:+.1f}% per unit", delta_color="off")
    st.caption(f"The slider starts at the hit rate of the shown games ({hits} of {n_sel}). "
               f"Bookmaker's no-margin chance of the over: {fair_over * 100:.1f}%. "
               "Move the slider to adjust for the opponent, minutes, injuries. "
               "A landing exactly on the line counts as an under here.")

    st.dataframe(sel[["Date", "Label", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TO"]]
                 .assign(Label=lambda d: d["Label"].str.replace("\n", " "), MIN=lambda d: d["MIN"].round(1))
                 .iloc[::-1], hide_index=True)

real = shots[shots["COORD_X"] != -1]  # drops free throws (no coordinates)

with tab2:
    ps = real[(real["ID_PLAYER"] == pid) & (real["Gamecode"].isin(sel["Gamecode"]))]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(shot_chart(ps, f"{pname} — {len(sel)} selected games"))
    with c2:
        st.subheader("Where the points come from")
        if ps.empty:
            st.info("No shots for this selection.")
        else:
            st.dataframe(profile(ps)[["Zone", "FGM", "FGA", "FG%", "PTS", "% of points"]], hide_index=True)
    st.caption("Free throws are not included in shot maps; shot-data FT misses don't exist in this file.")

with tab3:
    st.subheader(f"{opp_view} defence — {len(opp_sel)} games")
    c1, c2 = st.columns([3, 2])
    conceded = real[(real["Gamecode"].isin(opp_sel)) & (real["TEAM"] != opp_code)] if opp_code else real.iloc[0:0]
    with c1:
        st.pyplot(shot_chart(conceded, f"Shots conceded by {opp_view}"))
    with c2:
        st.write("**Where they concede points**")
        if conceded.empty:
            st.info("No shot data for this opponent.")
        else:
            st.dataframe(profile(conceded)[["Zone", "FGM", "FGA", "FG%", "PTS", "% of points"]], hide_index=True)

    st.write("**Zone defence vs league average (per game)**")
    zl = zone_vs_league(real, opp_code, opp_sel) if opp_code else None
    if zl is None:
        st.info("Not enough shot data for this comparison.")
    else:
        st.dataframe(zl.style.background_gradient(subset=["PTS/g vs league"], cmap="RdYlGn", vmin=-3, vmax=3),
                     hide_index=True)
        st.caption("PTS/g vs league above 0 (green) = this team gives up more points than the average "
                   "team in that zone. The opponent uses the games selected in the sidebar; "
                   "the league average uses the whole season.")

    st.write("**Defence vs position (conceded per game to each position)**")
    if dvp is None:
        st.info("Not enough data for this comparison.")
    else:
        sty = dvp_display(dvp).style
        for s_, v_ in (("PTS", 6), ("REB", 3), ("AST", 2)):
            sty = sty.background_gradient(subset=[f"{s_} vs lg"], cmap="RdYlGn", vmin=-v_, vmax=v_)
        st.dataframe(sty, hide_index=True)
        st.caption("'vs lg' above 0 (green) = gives up more than the league average to that position. "
                   "Per-game figures use the games selected in the sidebar; rank uses the whole season "
                   "(1 = concedes the most). Positions are a rough guess, so check the Positions tab.")

    st.write("**Per-game stats allowed (all teams, season; rank 1 = concedes the most)**")
    st.dataframe(team_defense(box).style.apply(
        lambda r: ["background-color:#26324a" if r.name == opp_view else "" for _ in r], axis=1))

with tab4:
    st.subheader(f"{team} — player positions")
    st.write("Positions are a rough Guard / Wing / Big guess from rebounds and assists. "
             "Fix any that look wrong, then press Save. Your choices are kept in data_cache/positions.csv.")
    mins = box.groupby("Player_ID")["MIN"].sum().round(0).rename("Min")
    view = (pos_df[pos_df["Team"] == team]
            .merge(mins, left_on="Player_ID", right_index=True, how="left")
            .sort_values("Min", ascending=False)[["Player_ID", "Player", "Min", "Position"]])
    edited = st.data_editor(
        view, hide_index=True, key=f"pos_{team}", disabled=["Player_ID", "Player", "Min"],
        column_config={"Player_ID": None,
                       "Position": st.column_config.SelectboxColumn("Position", options=POS_OPTIONS,
                                                                    required=True)})
    if st.button("Save positions"):
        new = dict(zip(edited["Player_ID"], edited["Position"]))
        pos_df["Position"] = [new.get(i, p) for i, p in zip(pos_df["Player_ID"], pos_df["Position"])]
        try:
            pos_df.to_csv(POS_PATH, index=False)
            st.rerun()
        except OSError:
            st.error("Couldn't write positions.csv (is it open in Excel?).")
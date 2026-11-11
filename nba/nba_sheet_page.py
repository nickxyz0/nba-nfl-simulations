"""Build tonight's NBA cheat sheet pages from output/cheat_sheets/sheet_<date>.json."""
import datetime as dt
import html
import json
import sys
from pathlib import Path

OUT = Path("output/cheat_sheets")
SOURCE_SHORT = {"DK": "DK", "fanduel": "FD", "pinnacle": "PIN", "kalshi": "KAL", "polymarket": "POLY"}

STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap">
<style>
:root {
  --bg: #0a0f1a; --panel: #111a2b; --panel-2: #16223a; --line: #22314f;
  --text: #eef3fb; --muted: #8d9bb8; --dim: #5d6b89;
  --hot: #3cf0a0; --warm: #a8f06a; --mild: #ffc857; --cold: #ff7a7a; --court: #ff9a3c;
  --display: "Barlow Condensed", "Arial Narrow", Impact, sans-serif;
  --body: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
  color-scheme: dark;
}
* { box-sizing: border-box }
body { background: var(--bg); color: var(--text); font: 14px/1.4 var(--body); }
.page { max-width: 980px; margin: 0 auto; padding-block: 24px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: flex; flex-direction: column; gap: 22px; }
header.top { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 12px; }
.kicker { font: 700 12px/1 var(--display); letter-spacing: .18em; text-transform: uppercase; color: var(--court); }
h1 { margin: 6px 0 0; font: 800 clamp(40px, 9vw, 76px)/.88 var(--display); letter-spacing: .01em; text-transform: uppercase; text-wrap: balance; }
h1 em { font-style: normal; color: var(--hot); }
.meta { color: var(--muted); font-size: 13px; max-width: 46ch; }
.meta b { color: var(--text); font-weight: 600 }
.sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip { font: 700 10.5px/1 var(--mono); letter-spacing: .04em; padding: 4px 6px; border-radius: 4px; background: var(--panel-2); color: var(--muted); border: 1px solid var(--line); white-space: nowrap; }
section.board { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.board-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding: 14px 16px 10px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
h2 { margin: 0; font: 800 26px/1 var(--display); text-transform: uppercase; letter-spacing: .02em; }
h2 span { color: var(--hot); }
.board-note { color: var(--dim); font-size: 12px; }
.row { display: grid; grid-template-columns: 34px minmax(0, 1.6fr) minmax(0, 1.25fr) 58px 58px 70px 74px; align-items: center; gap: 10px; padding: 10px 16px; border-top: 1px solid rgba(255,255,255,.04); }
.row.head { border-top: 0; padding-block: 8px; font: 700 10.5px/1 var(--display); letter-spacing: .12em; text-transform: uppercase; color: var(--dim); }
.row.head > :nth-child(n+4) { text-align: right }
.rank { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; background: var(--panel-2); font: 700 13px/1 var(--display); color: var(--muted); }
.row:nth-child(-n+4) .rank { background: var(--text); color: var(--bg); }
.who { min-width: 0 }
.who b { display: block; font: 700 20px/1.05 var(--display); letter-spacing: .01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.who small { color: var(--muted); font-size: 11.5px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 3px; }
.pick { font: 700 15px/1.15 var(--display); text-transform: uppercase; letter-spacing: .02em; min-width: 0; }
.pick .side-over { color: var(--hot) } .pick .side-under { color: var(--court) }
.pick .src { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 5px; }
.pick .src .chip { font-size: 9.5px; padding: 3px 4px }
.num { text-align: right; font: 500 13px/1 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.odds { text-align: right; font: 700 14px/1 var(--mono); font-variant-numeric: tabular-nums; }
.odds small { display: block; font: 500 9.5px/1.4 var(--body); color: var(--dim); letter-spacing: .04em; text-transform: uppercase; }
.score { justify-self: end; min-width: 64px; text-align: center; padding: 8px 6px; border-radius: 8px; font: 800 22px/1 var(--display); font-variant-numeric: tabular-nums; border: 1px solid; }
.t-hot { color: var(--hot); background: rgba(60,240,160,.10); border-color: rgba(60,240,160,.45) }
.t-warm { color: var(--warm); background: rgba(168,240,106,.08); border-color: rgba(168,240,106,.38) }
.t-mild { color: var(--mild); background: rgba(255,200,87,.08); border-color: rgba(255,200,87,.38) }
.t-cold { color: var(--cold); background: rgba(255,122,122,.08); border-color: rgba(255,122,122,.38) }
.spread-row { grid-template-columns: 34px minmax(0, 1.3fr) minmax(0, 1fr) 70px minmax(0, 1.1fr) 74px; }
.winner b { font: 700 18px/1 var(--display); text-transform: uppercase; }
.winner small { display: block; color: var(--muted); font-size: 11px; margin-top: 3px }
footer.legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px 22px; color: var(--muted); font-size: 12.5px; }
footer.legend b { color: var(--text); font-weight: 600 }
.combo-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.combo { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.combo-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.combo h3 { margin: 0; font: 800 22px/1 var(--display); text-transform: uppercase; letter-spacing: .02em; }
.combo ol { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 8px; }
.combo li { display: grid; grid-template-columns: 62px minmax(0,1fr) 44px; gap: 8px; align-items: center; font-size: 13px; }
.combo li .k { font: 700 10px/1 var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--dim); }
.combo li .p { text-align: right; font: 500 12px/1 var(--mono); color: var(--muted); }
.combo .fair { color: var(--dim); font-size: 12px; border-top: 1px solid var(--line); padding-top: 10px; display: flex; justify-content: space-between; }
@media (max-width: 640px) {
  .row { grid-template-columns: 28px minmax(0, 1fr) 68px; row-gap: 6px; }
  .row .num, .row .odds, .row.head { display: none; }
  .row .pick { grid-column: 2 / 3; }
  .row .score { grid-column: 3; grid-row: 1 / span 2; }
  .spread-row .winner { grid-column: 2 / 3; }
  .rank { width: 24px; height: 24px; }
}
@media (prefers-reduced-motion: no-preference) { .score { transition: transform .15s } .row:hover .score { transform: scale(1.05) } }
</style>
"""


def esc(text):
    return html.escape(str(text))


def tier(p):
    return "t-hot" if p >= .70 else "t-warm" if p >= .62 else "t-mild" if p >= .55 else "t-cold"


def fair(p):
    return f"{-round(100 * p / (1 - p)):+d}" if p >= .5 else f"+{round(100 * (1 - p) / p)}"


def pct(p):
    return f"{100 * p:.0f}%"


def sources(text):
    return "".join(f'<span class="chip">{SOURCE_SHORT.get(s, s.upper())}</span>' for s in str(text).split(",") if s)


def prop_rows(items):
    head = ('<div class="row head"><span>#</span><span>Player</span><span>Pick</span>'
            '<span>Sim</span><span>Market</span><span>Odds</span><span>Final</span></div>')
    rows = []
    for i, r in enumerate(items, 1):
        price = r.get("dk_price")
        odds = (f'{int(price):+d}<small>DraftKings</small>' if price not in (None, "None") and str(price) != "nan"
                else f'{fair(r["final"])}<small>fair</small>')
        rows.append(f"""<div class="row"><span class="rank">{i}</span>
<div class="who"><b>{esc(r['player'])}</b><small>{esc(r['team'])} · {esc(r['game'])}</small></div>
<div class="pick"><span class="side-{r['side']}">{esc(r['side'])} {r['line']:g}</span> {esc(r['label'])}<div class="src">{sources(r['sources'])}</div></div>
<span class="num">{pct(r['sim'])}</span><span class="num">{pct(r['market'])}</span>
<span class="odds">{odds}</span><span class="score {tier(r['final'])}">{pct(r['final'])}</span></div>""")
    return head + "".join(rows)


def spread_rows(items):
    head = ('<div class="row head spread-row"><span>#</span><span>Game</span><span>Spread pick</span>'
            '<span>Cover</span><span>Winner</span><span>Win</span></div>')
    rows = []
    for i, r in enumerate(items, 1):
        rows.append(f"""<div class="row spread-row"><span class="rank">{i}</span>
<div class="who"><b>{esc(r['game'])}</b><small>sim covers {pct(r['sim'])} · {r['books']} books at this line</small></div>
<div class="pick">{esc(r['pick'])}</div><span class="score {tier(r['final'])}">{pct(r['final'])}</span>
<div class="winner"><b>{esc(r['winner'])}</b><small>sim {pct(r['sim_win'])}</small></div>
<span class="score {tier(r['win'])}">{pct(r['win'])}</span></div>""")
    return head + "".join(rows)


def board(title, accent, note, body):
    return (f'<section class="board"><div class="board-head"><h2>{title} <span>{accent}</span></h2>'
            f'<span class="board-note">{note}</span></div>{body}</section>')


def main(date):
    data = json.loads((OUT / f"sheet_{date}.json").read_text())
    day = dt.date.fromisoformat(date)
    nice = day.strftime("%A, %b ") + str(day.day)
    games = len(data["all_spreads"])
    chips = "".join(f'<span class="chip">{SOURCE_SHORT.get(k, k.upper())} {v:,}</span>' for k, v in data["sources"].items())
    sheet = f"""<title>NBA Sim Board {day.strftime('%b')} {day.day}</title>
<meta name="claude:viewport-fit-sides" content="cover">
{STYLE}
<div class="page">
<header class="top"><div><div class="kicker">Ask the sims · {esc(nice)} · {games} games</div>
<h1>NBA <em>Cheat</em> Sheet</h1></div>
<div class="meta">Every game played <b>10,000 times</b>, then blended with the market: DraftKings, FanDuel, Pinnacle, Kalshi and Polymarket.
<b>Final</b> is the blended chance, weighted by how each source predicted 132 past games.<div class="sources">{chips}</div></div></header>
{board("Top 10", "Steals", "Lowest-noise prop type in the last week", prop_rows(data["steals"]))}
{board("Top 10", "Props", "Points, rebounds, assists and combos", prop_rows(data["general"]))}
{board("Top 5", "Starter Props", "Starters, points-based lines, 55%+", prop_rows(data["starters"]))}
{board("Top 5", "Spreads + Winners", "Spreads are near coin flips once the market weighs in; winners are where the edge shows", spread_rows(data["spreads"]))}
<footer class="legend">
<div><b>Sim</b> share of 10,000 simulated games that clear the line.</div>
<div><b>Market</b> no-vig consensus of the sources shown on each pick; Pinnacle counts double.</div>
<div><b>Final</b> sim and market blended with weights fit on Oct 20 to Nov 7 results. Day-to-day players are left off.</div>
<div><b>Odds</b> DraftKings price where it has the line, otherwise the fair price for the final chance.</div>
</footer></div>"""
    (OUT / f"sheet_{date}.html").write_text(sheet)

    cards = []
    for c in data["combos"]:
        leg_p = []
        items = []
        for text, kind in zip(c["legs"], c["kinds"]):
            p = next((r["final"] for lst in ("steals", "general", "starters") for r in data[lst]
                      if f"{r['player']} {r['side']} {r['line']:g} {r['label']}" == text),
                     next((s["final"] for s in data["spreads"] if s["pick"] == text), None))
            leg_p.append(p)
            items.append(f'<li><span class="k">{esc(kind)}</span><span>{esc(text)}</span><span class="p">{pct(p) if p else ""}</span></li>')
        cards.append(f"""<article class="combo"><div class="combo-top"><h3>{esc(c['name'])}</h3>
<span class="score {tier(c['p']) if c['p'] >= .55 else 't-mild'}">{pct(c['p'])}</span></div>
<ol>{''.join(items)}</ol><div class="fair"><span>Chance all legs hit</span><span>fair {fair(c['p'])}</span></div></article>""")
    combos = f"""<title>NBA Parlay Builder {day.strftime('%b')} {day.day}</title>
<meta name="claude:viewport-fit-sides" content="cover">
{STYLE}
<div class="page">
<header class="top"><div><div class="kicker">Ask the sims · {esc(nice)}</div><h1>Best <em>Combos</em></h1></div>
<div class="meta">Legs drawn from every list, including spreads. Chances use each leg's <b>final</b> number, adjusted where same-game legs
move together in the sims. Tonight's picks barely move together, so the strongest single legs win.</div></header>
<div class="combo-grid">{''.join(cards)}</div>
<footer class="legend"><div><b>Fair</b> the price that matches the chance. A book paying less than fair has the edge.</div>
<div><b>Expect less than shown on long parlays:</b> small errors in each leg multiply.</div></footer></div>"""
    (OUT / f"combos_{date}.html").write_text(combos)
    print(OUT / f"sheet_{date}.html", OUT / f"combos_{date}.html")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat())

"""Build the human distinguish test: can a reader tell two NLAs apart?

`plot_activation_logreg.py` says a linear probe on the base model's activations
identifies the authoring arm at 91-98% for every one of the six pairs. That is a
statement about the activation space, not about the text. This asks the other
question: shown explanations of the SAME document, can a human reader tell which
two came from the same arm?

The design is the sensory-science TETRAD, not the odd-one-out the question
usually suggests:

    one document -> 2 explanations from arm i, 2 from arm j, shuffled
    task: split the four into two same-author pairs

There are exactly three ways to partition four items into two unordered pairs, so
chance is 1/3 and the subject never has to know which arm is which -- no training
block, no labels, no way for a prior belief about "arm 3 is the terse one" to
enter except through the text itself. The tetrad is also the efficient member of
the family: simulated against a 1-D Thurstonian observer it needs roughly half the
reading of a 4-vs-1 odd-one-out at the same power, because grouping uses evidence
from both arms instead of only from the one that sticks out.

Why the document is held fixed: document content is the dominant source of
variance in these explanations. Four explanations of four different documents
would be grouped by subject matter, and the test would measure nothing.

Allocation is 7 trials x 6 pairs = 42, with the arm pair ROTATING from trial to
trial rather than blocked. Blocking would let a cue learned on trial 1 ("in this
block the short one is the odd arm") carry across the whole block, which both
inflates accuracy and destroys the independence the binomial test assumes.

Read the result as an OMNIBUS: 42 trials at chance 1/3 needs >=20 correct for
p<0.05, and that is 80%+ powered only against a true rate around 0.55. The
per-pair breakdown the scorer prints is 7 trials wide and is descriptive only --
it cannot support a per-pair claim, and was never meant to.

Samples are the four temp-1.0 seeds, which is the only place within-arm variation
exists: greedy decode gives one explanation per arm per document, so a same-arm
pair is not constructible from it at all. All 4 arms x 4 seeds x 128 documents
extract cleanly, so every document is eligible (the idx-94 repetition-loop failure
is greedy-only).

Writes two files, and the separation is the point:

    human_distinguish.html  -- the four texts per trial in shuffled display order,
                          and NOTHING else. No arm names, no seed indices, no
                          document ids beyond an opaque trial number.
    human_distinguish_key.json -- the answers. Do not open it until the run is
                          exported; opening it mid-run ends the experiment.

Usage:
    python human_distinguish_build.py --seed 0
    python human_distinguish_serve.py    # 42 trials, ~1 hour; autosaves to this folder
    python human_distinguish_score.py results_progress.json

Opening human_distinguish.html directly also works, but then the run lives in the
browser's storage plus whatever you export by hand -- see human_distinguish_serve.py
for why that is not enough.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

# Defaults are resolved against this file, not the shell's cwd, so the four files
# stay one self-contained folder and the build can be run from anywhere.
HERE = Path(__file__).resolve().parent

ARMS = ["nla1", "nla2", "nla3", "nla4"]
SEEDS = [f"temp1_seed{i}" for i in range(4)]
PAIRS = list(itertools.combinations(ARMS, 2))  # 6

# What each pair isolates under the OFAT design. Carried into the key so the
# scorer can group the six pairs into the three factors without restating it.
FACTOR = {
    ("nla1", "nla2"): "rollout noise floor",
    ("nla1", "nla3"): "SFT init (+ rollout)",
    ("nla1", "nla4"): "RL data order (+ rollout)",
    ("nla2", "nla3"): "SFT init",
    ("nla2", "nla4"): "RL data order",
    ("nla3", "nla4"): "SFT init vs RL data order",
}

# The three partitions of four display slots into two unordered pairs. Order is
# fixed so the button order in the HTML is stable across builds; which one is
# correct is of course shuffled by the slot assignment, not by this list.
PARTITIONS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def load_generations(gen_dir):
    """(seed, arm) -> {idx: explanation}, keeping only extracted rows."""
    out = {}
    for s in SEEDS:
        for a in ARMS:
            rows = json.loads((gen_dir / s / f"{a}.json").read_text())
            out[(s, a)] = {r["idx"]: r["explanation"].strip()
                           for r in rows if r["extracted"]}
    return out


def eligible_docs(gens):
    """Documents whose explanation extracted in all 16 (seed, arm) cells.

    A document missing even one cell could still serve some pairs, but letting the
    eligible set differ by pair would confound pair with document difficulty, and
    at 128/128 the strictness costs nothing.
    """
    sets = [set(v) for v in gens.values()]
    return sorted(set.intersection(*sets))


def build_trials(gens, docs, rng, per_pair):
    """One trial per document: pair, two seeds per arm, placed into four slots.

    Which of the three buttons is correct is BALANCED rather than left to chance --
    a free shuffle of the slots gave 19/14/9 at this n, and a subject who simply
    favours one button position would then score above or below 1/3 on layout
    alone. Everything else is random: which arm of the pair takes which half of the
    split, which seeds are drawn, and which of a group's two texts comes first.
    """
    order = [p for p in PAIRS for _ in range(per_pair)]
    rng.shuffle(order)
    picked = rng.sample(docs, len(order))
    n = len(order)
    if n % len(PARTITIONS):
        raise SystemExit(f"{n} trials does not divide by {len(PARTITIONS)} partitions; "
                         f"--per-pair must be a multiple of 3 / 6")
    targets = [q for q in range(len(PARTITIONS)) for _ in range(n // len(PARTITIONS))]
    rng.shuffle(targets)

    trials, key = [], []
    for t, (pair, idx, truth) in enumerate(zip(order, picked, targets)):
        # Two distinct seeds per arm, drawn independently, so a trial never
        # compares seed 0 against seed 0 by construction -- that would make
        # "same seed" a cue for "same arm" if the seeds shared any decode state.
        g1, g2 = PARTITIONS[truth]
        halves = [g1, g2]
        rng.shuffle(halves)                    # which arm of the pair sits where
        texts = [None] * 4
        arms = [None] * 4
        seeds = [None] * 4
        for arm, half in zip(pair, halves):
            for pos, seed in zip(half, rng.sample(SEEDS, 2)):
                texts[pos] = gens[(seed, arm)][idx]
                arms[pos] = arm
                seeds[pos] = seed

        trials.append({"t": t, "texts": texts})
        key.append({
            "t": t,
            "doc_idx": idx,
            "pair": list(pair),
            "factor": FACTOR[pair],
            "arms": arms,
            "seeds": seeds,
            "correct": truth,
        })
    return trials, key


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>NLA distinguish</title>
<style>
  :root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e2e0dc;
          --sel:#2a78d6; --selbg:#eaf2fd; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; background:var(--bg);
           border-bottom:1px solid var(--line); padding:10px 20px; z-index:2;
           display:flex; align-items:center; gap:16px; }
  header b { font-size:14px; font-weight:600; }
  #bar { flex:1; height:4px; background:var(--line); border-radius:2px; }
  #bar>div { height:100%; width:0; background:var(--sel); border-radius:2px;
             transition:width .2s; }
  main { padding:20px; max-width:1600px; margin:0 auto; }
  #cards { display:grid; gap:14px; grid-template-columns:repeat(4,1fr); }
  @media (max-width:1400px){ #cards{ grid-template-columns:repeat(2,1fr);} }
  @media (max-width:760px){ #cards{ grid-template-columns:1fr;} }
  .card { border:1px solid var(--line); border-radius:8px; padding:14px;
          background:#fff; }
  .card.on { border-color:var(--sel); background:var(--selbg); }
  .tag { font-weight:700; font-size:13px; letter-spacing:.04em; }
  .card p { margin:8px 0 0; white-space:pre-wrap; }
  #ask { margin:22px 0 8px; font-size:14px; color:var(--muted); }
  #opts { display:flex; gap:10px; flex-wrap:wrap; }
  button { font:inherit; padding:9px 16px; border:1px solid var(--line);
           border-radius:7px; background:#fff; cursor:pointer; }
  button:hover { border-color:var(--sel); }
  button.on { background:var(--sel); border-color:var(--sel); color:#fff; }
  #why { width:100%; max-width:640px; margin-top:16px; padding:8px 10px;
         border:1px solid var(--line); border-radius:7px; font:inherit; }
  #next { margin-top:16px; font-weight:600; }
  #next[disabled] { opacity:.4; cursor:default; }
  #warn { display:none; background:#fdecea; border:1px solid #f3b4ad;
          border-radius:7px; padding:10px 14px; margin-bottom:18px; font-size:14px; }
  #done { display:none; }
  textarea { width:100%; height:300px; font:12px/1.4 ui-monospace,Menlo,monospace;
             border:1px solid var(--line); border-radius:7px; padding:10px; }
  .note { color:var(--muted); font-size:13px; max-width:760px; }
  .note button { padding:4px 10px; font-size:13px; }
  #stat { margin-left:8px; }
</style>

<header>
  <b>NLA distinguish</b>
  <span id="n"></span>
  <div id="bar"><div></div></div>
</header>

<main>
  <div id="warn"><b>Progress will not be saved in this browser.</b>
    Closing the tab loses the run. Use a normal (non-private) window, or finish
    in one sitting.</div>

  <div id="run">
    <div id="cards"></div>
    <div id="ask">Which two were written by the same NLA? Pick the split.</div>
    <div id="opts"></div>
    <input id="why" placeholder="Reason (optional)">
    <br><button id="next" disabled>Next &rarr;</button>
    <p class="note"><span id="where">Answers so far live only in this
      browser's storage, which a cache clear &mdash; or a storage database the
      browser decides to rebuild after a crash &mdash; takes with it. Export
      every so often, or open this page through
      <code>human_distinguish_serve.py</code> and it saves itself.</span>
      <button id="snap">Save progress to file</button>
      <button id="resume">Resume from file</button>
      <span id="stat"></span>
      <input id="file" type="file" accept=".json,application/json" hidden>
    </p>
  </div>

  <div id="done">
    <h2>Done &mdash; 42 trials</h2>
    <p class="note">Save this as <code>results.json</code>, then run
      <code>python human_distinguish_score.py results.json</code>. The key was never in
      this page, so nothing here tells you how you did.</p>
    <p><button id="dl">Download results.json</button>
       <button id="clear">Clear saved progress</button></p>
    <textarea id="out" readonly></textarea>
  </div>
</main>

<script>
const TRIALS = __TRIALS__;
const BUILD = __BUILD__;
const LABELS = ["A","B","C","D"];
const PARTS = [[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]];
const KEY = "nla_distinguish_" + BUILD.seed;

let state = load();

function load(){
  try { const s = JSON.parse(localStorage.getItem(KEY));
        if (s && s.build === BUILD.stamp) return s;
        // A stamp mismatch is a rebuild, not a new subject, and the probe save()
        // at the bottom of this file would write over those answers before
        // anyone saw them. Park them under their own key: a run that no longer
        // matches the trials is still worth more than a deleted one.
        if (s && s.answers && s.answers.length)
          localStorage.setItem(KEY + "_orphan_" + s.build, JSON.stringify(s));
      } catch(e){}
  return {build: BUILD.stamp, i: 0, answers: []};
}
// A silent catch here would cost an hour of reading without telling anyone, so a
// failed write raises the banner instead of being swallowed.
function save(){
  let ok = true;
  try { localStorage.setItem(KEY, JSON.stringify(state)); }
  catch(e){ document.getElementById("warn").style.display = "block"; ok = false; }
  if (SYNC) push();
  return ok;
}

// Served through human_distinguish_serve.py the page keeps its run in
// results_progress.json next to itself, which no browser can clear. A failed
// write is shown, not swallowed: it means the server has gone away and the
// answers are once again only in this browser.
const SYNC = location.protocol === "http:" || location.protocol === "https:";
function push(){
  fetch("/progress", {method:"PUT", body: exportText(),
                      headers:{"Content-Type":"application/json"}})
    .then(r => { if (!r.ok) throw new Error(r.status); synced(true); })
    .catch(() => synced(false));
}
function synced(ok){
  const w = document.getElementById("warn");
  if (ok) { w.style.display = "none"; return; }
  w.innerHTML = "<b>Could not write results_progress.json.</b> Is " +
    "human_distinguish_serve.py still running? Until it is, answers are only in " +
    "this browser -- use <i>Save progress to file</i>.";
  w.style.display = "block";
}
async function pull(){
  try {
    const r = await fetch("/progress", {cache:"no-store"});
    if (!r.ok) return;
    const d = await r.json();
    if (!d.build || d.build.stamp !== BUILD.stamp || !Array.isArray(d.answers)) return;
    // The file wins whenever it is at least as far along: it is what survives.
    if (d.answers.length >= state.answers.length)
      state = {build: BUILD.stamp, i: d.answers.length, answers: d.answers};
  } catch(e){}
}

function exportText(){
  return JSON.stringify({build: BUILD, answers: state.answers}, null, 1);
}

function download(name, text){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type:"application/json"}));
  a.download = name; a.click();
}

function render(){
  if (state.i >= TRIALS.length) return finish();
  document.getElementById("stat").textContent =
    state.answers.length + " answered";
  const tr = TRIALS[state.i];
  document.getElementById("n").textContent = (state.i+1) + " / " + TRIALS.length;
  document.querySelector("#bar>div").style.width =
    (100*state.i/TRIALS.length) + "%";

  document.getElementById("cards").innerHTML = tr.texts.map((x,k) =>
    '<div class="card" id="c'+k+'"><span class="tag">'+LABELS[k]+'</span>' +
    '<p>'+esc(x)+'</p></div>').join("");

  document.getElementById("opts").innerHTML = PARTS.map((p,k) =>
    '<button data-k="'+k+'">'+LABELS[p[0][0]]+'+'+LABELS[p[0][1]]+
    ' &nbsp;/&nbsp; '+LABELS[p[1][0]]+'+'+LABELS[p[1][1]]+'</button>').join("");

  let pick = null;
  document.querySelectorAll("#opts button").forEach(b => b.onclick = () => {
    pick = +b.dataset.k;
    document.querySelectorAll("#opts button").forEach(o =>
      o.classList.toggle("on", o === b));
    // Tint the two cards the chosen split calls same-author, so a misclick on
    // the button row is visible against the text rather than only in the buttons.
    const g = PARTS[pick][0];
    [0,1,2,3].forEach(c =>
      document.getElementById("c"+c).classList.toggle("on", g.includes(c)));
    document.getElementById("next").disabled = false;
  });

  document.getElementById("why").value = "";
  document.getElementById("next").disabled = true;
  document.getElementById("next").onclick = () => {
    if (pick === null) return;
    state.answers.push({t: tr.t, pick: pick,
                        why: document.getElementById("why").value.trim()});
    state.i++; save();
    window.scrollTo(0,0); render();
  };
  window.scrollTo(0,0);
}

function finish(){
  document.getElementById("run").style.display = "none";
  document.getElementById("done").style.display = "block";
  document.querySelector("#bar>div").style.width = "100%";
  document.getElementById("n").textContent = "complete";
  if (SYNC) document.querySelector("#done .note").innerHTML =
    "Already written to <code>results_progress.json</code> in this folder: run " +
    "<code>python human_distinguish_score.py results_progress.json</code>. The key " +
    "was never in this page, so nothing here tells you how you did.";
  const blob = exportText();
  document.getElementById("out").value = blob;
  document.getElementById("dl").onclick = () => download("results.json", blob);
  document.getElementById("clear").onclick = () => {
    if (confirm("Erase saved progress for this build?")) {
      localStorage.removeItem(KEY); location.reload();
    }
  };
}

function esc(s){ return s.replace(/[&<>]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// Browser storage is one bad shutdown away from being rebuilt empty, and the
// write probe cannot see that coming, so the run can also be carried in a file.
// Resume trusts answers.length over the saved index, so the two cannot drift.
document.getElementById("snap").onclick = () =>
  download("results_partial_" + state.answers.length + ".json", exportText());
document.getElementById("resume").onclick = () =>
  document.getElementById("file").click();
document.getElementById("file").onchange = ev => {
  const f = ev.target.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    let d;
    try { d = JSON.parse(r.result); } catch(e){ return alert("Not JSON."); }
    const stamp = d && d.build && d.build.stamp;
    if (stamp !== BUILD.stamp)
      return alert("That file is from build " + (stamp || "unknown") +
                   ", this page is " + BUILD.stamp + ".");
    if (!Array.isArray(d.answers)) return alert("No answers in that file.");
    if (d.answers.length < state.answers.length &&
        !confirm("That file has " + d.answers.length + " answers; this browser " +
                 "has " + state.answers.length + ". Load it anyway?")) return;
    state = {build: BUILD.stamp, i: d.answers.length, answers: d.answers};
    save(); render();
  };
  r.readAsText(f);
};

(async () => {
  if (SYNC) {
    await pull();
    document.getElementById("where").textContent =
      "Saving to results_progress.json in this folder after every answer.";
  }
  save();   // probes writability on load, so the banner appears before trial 1
  render();
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, default=HERE / "generations",
                    help="directory holding temp1_seed*/nla*.json")
    ap.add_argument("--per-pair", type=int, default=7,
                    help="trials per arm pair; 7 x 6 pairs = 42 (default)")
    ap.add_argument("--seed", type=int, default=0,
                    help="randomisation seed; also names the localStorage slot, so "
                         "a rebuild with a new seed does not resume an old run")
    ap.add_argument("--out", type=Path, default=HERE / "human_distinguish.html")
    ap.add_argument("--key", type=Path, default=HERE / "human_distinguish_key.json")
    a = ap.parse_args()

    gens = load_generations(a.gen_dir)
    docs = eligible_docs(gens)
    n = a.per_pair * len(PAIRS)
    if len(docs) < n:
        raise SystemExit(f"need {n} documents, only {len(docs)} extract in all 16 cells")

    rng = random.Random(a.seed)
    trials, key = build_trials(gens, docs, rng, a.per_pair)

    build = {"seed": a.seed, "n": n, "per_pair": a.per_pair,
             "stamp": f"s{a.seed}n{n}v2"}
    a.out.write_text(HTML
                     .replace("__TRIALS__", json.dumps(trials))
                     .replace("__BUILD__", json.dumps(build)))
    a.key.write_text(json.dumps({"build": build, "key": key}, indent=1))

    print(f"{n} trials over {len(PAIRS)} pairs ({a.per_pair} each), "
          f"{n} distinct documents of {len(docs)} eligible")
    print(f"  {a.out}  <- open this")
    print(f"  {a.key}  <- do NOT open until you have exported results.json")


if __name__ == "__main__":
    main()

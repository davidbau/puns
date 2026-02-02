#!/usr/bin/env python3
"""
Interactive 3D layer visualization for contrastive activation data.

Generates a self-contained HTML page with embedded SVG/JS that renders
a 3D scatterplot of contrastive projections across all layers.  No
external JS dependencies — everything is vanilla JS + inline SVG.

Usage from Python / Jupyter:

    from puns_viz import make_layer_viz
    html = make_layer_viz("results/raw_activations/llama31_70b_instruct_pred_c_meta.json")

    # In Jupyter:
    from IPython.display import HTML
    HTML(html)

    # Or write to file:
    with open("results/figures/pred_c_3d_layers.html", "w") as f:
        f.write(html)

Standalone:

    python puns_viz.py          # writes results/figures/pred_c_3d_layers.html
"""

import json
import math
from pathlib import Path

import numpy as np

from analyze_activations import (
    load_activations,
    get_pair_indices,
    contrastive_projection,
    stable_contrastive_projections,
    load_detailed_predictions,
    pun_boost_per_pair,
)

BASE = Path(__file__).parent
RAW_DIR = BASE / "results" / "raw_activations"
FIGURES_DIR = BASE / "results" / "figures"


def _build_data_payload(projections_result, meta, detailed_preds=None):
    """
    Assemble the JSON data payload embedded in the HTML.

    Returns a dict ready for json.dumps().
    """
    samples = meta["samples"]
    n_prompts = len(samples)
    projs = projections_result["projections"]
    var_ratios = projections_result["var_ratios"]
    axis_ranges = projections_result["axis_ranges"]

    # Load pun words from cloze tests if available
    tests_file = BASE / "datasets" / "contextual_cloze_tests_100.json"
    pun_words_map = {}  # pair_id -> list of pun words
    straight_words_map = {}  # pair_id -> list of expected completions
    if tests_file.exists():
        with open(tests_file) as f:
            tests = json.load(f)
        for t in tests:
            pid = t["pair_id"]
            if t["type"] == "funny":
                pun_words_map[pid] = t.get("expected_completion", [])
                straight_words_map[pid] = t.get("contrast_completion", [])
            elif t["type"] == "straight" and pid not in straight_words_map:
                straight_words_map[pid] = t.get("expected_completion", [])
                pun_words_map[pid] = t.get("contrast_completion", [])

    # Detailed predictions per sample
    pred_by_index = {}
    boost_by_pair = {}
    if detailed_preds is not None:
        boost_by_pair = pun_boost_per_pair(detailed_preds)
        for r in detailed_preds["results"]:
            top1 = r["top_tokens"][0]["word"] if r["top_tokens"] else "?"
            pun_prob = sum(v["prob"] for v in r["pun_word_probs"].values())
            pred_by_index[r["index"]] = {
                "top1": top1,
                "pun_prob": round(pun_prob, 4),
            }

    # Build points array
    points = []
    for i, s in enumerate(samples):
        pid = s["pair_id"]
        boost_ratio = boost_by_pair.get(pid, 1.0)
        has_boost = boost_ratio >= 2.0 and not math.isinf(boost_ratio)
        pred_info = pred_by_index.get(s["index"], {})

        point = {
            "i": i,
            "pair_id": pid,
            "type": s["type"],
            "sentence": s.get("joke_c_sentence", ""),
            "pun_words": pun_words_map.get(pid, []),
            "straight_words": straight_words_map.get(pid, []),
            "top1": pred_info.get("top1", ""),
            "pun_prob": pred_info.get("pun_prob", None),
            "boost_ratio": round(boost_ratio, 1) if not math.isinf(boost_ratio) else 9999,
            "has_boost": has_boost,
        }
        points.append(point)

    # Pair map: list of [straight_idx, funny_idx]
    pair_map = {}
    for i, s in enumerate(samples):
        pair_map.setdefault(s["pair_id"], {})[s["type"]] = i
    pair_list = []
    for pid in sorted(pair_map.keys()):
        p = pair_map[pid]
        if "straight" in p and "funny" in p:
            pair_list.append([p["straight"], p["funny"]])

    # Pair labels
    pair_labels = {}
    for pid in sorted(pair_map.keys()):
        s = samples[pair_map[pid].get("straight", pair_map[pid].get("funny"))]
        sentence = s.get("joke_c_sentence", "")
        words = sentence.replace("The ", "").replace("the ", "").split()
        subject = words[0].rstrip("'s") if words else "?"
        pun = pun_words_map.get(pid, ["?"])[0] if pun_words_map.get(pid) else "?"
        pair_labels[pid] = f"{subject}...{pun}"

    # Layer projections (rounded for compact JSON)
    layers = {}
    vr_out = {}
    for layer_idx, proj in projs.items():
        layers[str(layer_idx)] = np.round(proj, 3).tolist()
        vr_out[str(layer_idx)] = [round(float(v), 4) for v in var_ratios[layer_idx]]

    return {
        "points": points,
        "layers": layers,
        "varRatios": vr_out,
        "pairMap": pair_list,
        "pairLabels": pair_labels,
        "axisRanges": axis_ranges,
        "layerIndices": sorted(projs.keys()),
        "model": meta.get("model", ""),
        "position": meta.get("position", ""),
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Contrastive Activation Explorer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
       background: #1a1a2e; color: #e0e0e0; display: flex; justify-content: center; padding: 16px; }
.container { width: __WIDTH__px; }
.header { margin-bottom: 8px; }
.header h1 { font-size: 18px; font-weight: 600; color: #fff; }
.header .subtitle { font-size: 13px; color: #999; margin-top: 2px; }
.var-info { font-size: 12px; color: #888; margin-top: 4px; font-family: monospace; }

.svg-wrap { position: relative; border: 1px solid #333; border-radius: 8px;
            background: #0f0f23; overflow: hidden; cursor: grab; }
.svg-wrap:active { cursor: grabbing; }
svg { display: block; }

.controls { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
.controls label { font-size: 13px; color: #aaa; white-space: nowrap; }
.slider-group { flex: 1; display: flex; align-items: center; gap: 8px; min-width: 200px; }
.slider-group input[type=range] { flex: 1; accent-color: #7B68EE; }
.layer-num { font-family: monospace; font-size: 14px; color: #fff; min-width: 60px; }
.btn { background: #333; color: #ccc; border: 1px solid #555; border-radius: 4px;
       padding: 4px 12px; font-size: 12px; cursor: pointer; }
.btn:hover { background: #444; color: #fff; }
.btn.active { background: #7B68EE; color: #fff; border-color: #7B68EE; }

.legend { display: flex; gap: 16px; margin-top: 8px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 12px; color: #bbb; }
.legend-swatch { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #555; }
.legend-star { font-size: 16px; line-height: 12px; }

.tooltip { position: absolute; pointer-events: none; background: rgba(15,15,35,0.95);
           border: 1px solid #555; border-radius: 6px; padding: 10px 12px;
           font-size: 12px; line-height: 1.5; max-width: 380px; z-index: 100;
           display: none; box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
.tooltip .tt-title { font-weight: 600; color: #fff; margin-bottom: 4px; }
.tooltip .tt-context { color: #aaa; }
.tooltip .tt-sentence { color: #ccc; font-style: italic; margin: 4px 0; }
.tooltip .tt-detail { color: #999; font-family: monospace; font-size: 11px; }
.tooltip .tt-boost { color: #f0c040; }

.checkbox-group { display: flex; align-items: center; gap: 4px; }
.checkbox-group input { accent-color: #7B68EE; }
.checkbox-group label { font-size: 12px; color: #aaa; cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Contrastive Activation Explorer</h1>
    <div class="subtitle" id="modelInfo"></div>
    <div class="var-info" id="varInfo"></div>
  </div>

  <div class="svg-wrap" id="svgWrap">
    <svg id="mainSvg" width="__WIDTH__" height="__HEIGHT__"></svg>
    <div class="tooltip" id="tooltip"></div>
  </div>

  <div class="controls">
    <label>Layer:</label>
    <div class="slider-group">
      <input type="range" id="layerSlider" min="0" max="79" value="40">
      <span class="layer-num" id="layerNum">Layer 40</span>
    </div>
    <button class="btn" id="playBtn">Play &#9654;</button>
    <div class="checkbox-group">
      <input type="checkbox" id="showLines" checked>
      <label for="showLines">Pair lines</label>
    </div>
    <div class="checkbox-group">
      <input type="checkbox" id="showLabels">
      <label for="showLabels">Labels</label>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#4A90D9"></div> Straight context</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#E85D75"></div> Funny context</div>
    <div class="legend-item"><span class="legend-star" style="color:#f0c040">&#9733;</span> 2x+ pun boost</div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

const W = __WIDTH__, H = __HEIGHT__;
const PAD = 50;
const COLOR_S = "#4A90D9", COLOR_F = "#E85D75";

// State
let azimuth = -0.5, elevation = 0.35;
let currentLayer = 0;
let playing = false;
let playTimer = null;
let selectedPair = null;
let dragging = false, dragStartX = 0, dragStartY = 0, dragAz0 = 0, dragEl0 = 0;

// DOM
const svg = document.getElementById("mainSvg");
const svgWrap = document.getElementById("svgWrap");
const tooltip = document.getElementById("tooltip");
const slider = document.getElementById("layerSlider");
const layerNum = document.getElementById("layerNum");
const playBtn = document.getElementById("playBtn");
const showLines = document.getElementById("showLines");
const showLabels = document.getElementById("showLabels");
const modelInfo = document.getElementById("modelInfo");
const varInfo = document.getElementById("varInfo");

// Init slider range
const layerIndices = DATA.layerIndices;
slider.min = 0;
slider.max = layerIndices.length - 1;
slider.value = Math.floor(layerIndices.length / 2);

modelInfo.textContent = DATA.model + " | position=" + DATA.position + " | " +
    DATA.points.length + " samples, " + layerIndices.length + " layers";

// 3D projection
function rotMatrix(az, el) {
    const ca = Math.cos(az), sa = Math.sin(az);
    const ce = Math.cos(el), se = Math.sin(el);
    return [
        [ca, -sa * se, sa * ce],
        [0, ce, se],
        [-sa, -ca * se, ca * ce]
    ];
}

function project3D(pts, az, el, ranges) {
    const R = rotMatrix(az, el);
    const n = pts.length;
    const out = new Array(n);
    // Normalize to [-1, 1] using axis ranges
    const cx = (ranges[0][0] + ranges[0][1]) / 2;
    const cy = (ranges[1][0] + ranges[1][1]) / 2;
    const cz = (ranges[2][0] + ranges[2][1]) / 2;
    const sx = (ranges[0][1] - ranges[0][0]) / 2 || 1;
    const sy = (ranges[1][1] - ranges[1][0]) / 2 || 1;
    const sz = (ranges[2][1] - ranges[2][0]) / 2 || 1;
    const scale = Math.max(sx, sy, sz);

    for (let i = 0; i < n; i++) {
        const x = (pts[i][0] - cx) / scale;
        const y = (pts[i][1] - cy) / scale;
        const z = (pts[i][2] - cz) / scale;
        const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
        const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
        const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
        out[i] = {
            sx: W/2 + px * (W/2 - PAD),
            sy: H/2 - py * (H/2 - PAD),
            depth: pz,
            idx: i
        };
    }
    return out;
}

// SVG helpers
function svgEl(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
}

function starPath(cx, cy, r) {
    const pts = [];
    for (let i = 0; i < 10; i++) {
        const angle = Math.PI/2 + i * Math.PI/5;
        const rad = i % 2 === 0 ? r : r * 0.4;
        pts.push((cx + rad * Math.cos(angle)).toFixed(1) + "," +
                 (cy - rad * Math.sin(angle)).toFixed(1));
    }
    return pts.join(" ");
}

// Render
function render() {
    const sliderIdx = parseInt(slider.value);
    const layerIdx = layerIndices[sliderIdx];
    currentLayer = layerIdx;
    layerNum.textContent = "Layer " + layerIdx;

    const pts3d = DATA.layers[String(layerIdx)];
    if (!pts3d) return;

    const vr = DATA.varRatios[String(layerIdx)];
    if (vr) {
        varInfo.textContent = "Variance: contrastive=" + (vr[0]*100).toFixed(1) + "%" +
            "  residPC1=" + (vr[1]*100).toFixed(1) + "%" +
            "  residPC2=" + (vr[2]*100).toFixed(1) + "%";
    }

    const projected = project3D(pts3d, azimuth, elevation, DATA.axisRanges);

    // Depth sort (painter's algorithm: render far objects first)
    const sorted = projected.slice().sort((a, b) => a.depth - b.depth);

    // Clear SVG
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    // Draw axis indicator
    drawAxes();

    // Draw pair lines
    if (showLines.checked) {
        const lineGroup = svgEl("g", {opacity: "0.25"});
        for (const [si, fi] of DATA.pairMap) {
            const sp = projected[si], fp = projected[fi];
            const pid = DATA.points[si].pair_id;
            const isSelected = selectedPair === pid;
            const line = svgEl("line", {
                x1: sp.sx.toFixed(1), y1: sp.sy.toFixed(1),
                x2: fp.sx.toFixed(1), y2: fp.sy.toFixed(1),
                stroke: isSelected ? "#fff" : "#666",
                "stroke-width": isSelected ? 2 : 0.8,
                opacity: isSelected ? 1 : 0.5
            });
            lineGroup.appendChild(line);

            // Label at midpoint
            if (showLabels.checked) {
                const mx = (sp.sx + fp.sx) / 2;
                const my = (sp.sy + fp.sy) / 2;
                const label = DATA.pairLabels[String(pid)] || "";
                const txt = svgEl("text", {
                    x: mx.toFixed(1), y: my.toFixed(1),
                    fill: "#666", "font-size": "8", "text-anchor": "middle",
                    "dominant-baseline": "middle"
                });
                txt.textContent = label;
                lineGroup.appendChild(txt);
            }
        }
        svg.appendChild(lineGroup);
    }

    // Draw points
    const pointGroup = svgEl("g", {});
    for (const p of sorted) {
        const pt = DATA.points[p.idx];
        const color = pt.type === "funny" ? COLOR_F : COLOR_S;
        const isSelected = selectedPair === pt.pair_id;
        const alpha = isSelected ? 1.0 : 0.8;
        const baseR = isSelected ? 7 : 5;

        if (pt.has_boost) {
            const r = baseR * 1.6;
            const star = svgEl("polygon", {
                points: starPath(p.sx, p.sy, r),
                fill: color,
                stroke: isSelected ? "#fff" : "#f0c040",
                "stroke-width": isSelected ? 2 : 1,
                opacity: alpha,
                "data-idx": p.idx
            });
            star.addEventListener("mouseenter", (e) => showTooltip(e, p.idx));
            star.addEventListener("mouseleave", hideTooltip);
            star.addEventListener("click", () => togglePair(pt.pair_id));
            pointGroup.appendChild(star);
        } else {
            const circle = svgEl("circle", {
                cx: p.sx.toFixed(1), cy: p.sy.toFixed(1), r: baseR,
                fill: color,
                stroke: isSelected ? "#fff" : "rgba(255,255,255,0.3)",
                "stroke-width": isSelected ? 2 : 0.5,
                opacity: alpha,
                "data-idx": p.idx
            });
            circle.addEventListener("mouseenter", (e) => showTooltip(e, p.idx));
            circle.addEventListener("mouseleave", hideTooltip);
            circle.addEventListener("click", () => togglePair(pt.pair_id));
            pointGroup.appendChild(circle);
        }
    }
    svg.appendChild(pointGroup);
}

function drawAxes() {
    const len = 40;
    const origin = [W - 70, H - 50];
    const R = rotMatrix(azimuth, elevation);
    const axes = [
        {dx: R[0][0], dy: -R[1][0], label: "C", color: "#7B68EE"},
        {dx: R[0][1], dy: -R[1][1], label: "R1", color: "#4A90D9"},
        {dx: R[0][2], dy: -R[1][2], label: "R2", color: "#E85D75"},
    ];
    const g = svgEl("g", {opacity: "0.6"});
    for (const a of axes) {
        const x2 = origin[0] + a.dx * len;
        const y2 = origin[1] + a.dy * len;
        g.appendChild(svgEl("line", {
            x1: origin[0], y1: origin[1], x2: x2.toFixed(1), y2: y2.toFixed(1),
            stroke: a.color, "stroke-width": 1.5
        }));
        g.appendChild((() => {
            const t = svgEl("text", {
                x: (origin[0] + a.dx * (len + 10)).toFixed(1),
                y: (origin[1] + a.dy * (len + 10)).toFixed(1),
                fill: a.color, "font-size": "10", "text-anchor": "middle",
                "dominant-baseline": "middle"
            });
            t.textContent = a.label;
            return t;
        })());
    }
    svg.appendChild(g);
}

// Tooltip
function showTooltip(event, idx) {
    const pt = DATA.points[idx];
    const pts3d = DATA.layers[String(currentLayer)];
    const contrastVal = pts3d ? pts3d[idx][0].toFixed(2) : "?";
    const contextLabel = pt.type === "funny" ? "funny" : "straight";
    const boostStr = pt.has_boost ?
        '<span class="tt-boost">2x+ pun boost (' + pt.boost_ratio + 'x)</span>' :
        'no significant boost (' + pt.boost_ratio + 'x)';
    const pun_words = pt.pun_words.length ? pt.pun_words.join(", ") : "?";
    const straight_words = pt.straight_words.length ?
        pt.straight_words.slice(0, 4).join(", ") +
        (pt.straight_words.length > 4 ? "..." : "") : "?";
    const pun_prob_str = pt.pun_prob !== null ? "P(pun)=" + pt.pun_prob : "";
    const top1_str = pt.top1 ? 'Top-1 predicted: "' + pt.top1 + '"' : "";

    tooltip.innerHTML =
        '<div class="tt-title">Pair ' + pt.pair_id + ': ' +
            (DATA.pairLabels[String(pt.pair_id)] || "") + '</div>' +
        '<div class="tt-context">' + contextLabel + ' | ' + boostStr + '</div>' +
        '<div class="tt-sentence">"' + pt.sentence + '"</div>' +
        '<div class="tt-detail">Pun words: ' + pun_words + '  ' + pun_prob_str + '</div>' +
        '<div class="tt-detail">Straight words: ' + straight_words + '</div>' +
        (top1_str ? '<div class="tt-detail">' + top1_str + '</div>' : '') +
        '<div class="tt-detail">Contrastive axis: ' + contrastVal +
            (parseFloat(contrastVal) > 0 ? ' (funny direction)' : ' (straight direction)') +
        '</div>';

    const rect = svgWrap.getBoundingClientRect();
    let tx = event.clientX - rect.left + 15;
    let ty = event.clientY - rect.top - 10;
    if (tx + 350 > W) tx = event.clientX - rect.left - 360;
    if (ty + 180 > H) ty = event.clientY - rect.top - 180;
    tooltip.style.left = tx + "px";
    tooltip.style.top = ty + "px";
    tooltip.style.display = "block";
}

function hideTooltip() {
    tooltip.style.display = "none";
}

function togglePair(pairId) {
    selectedPair = selectedPair === pairId ? null : pairId;
    render();
}

// Drag rotation
svgWrap.addEventListener("mousedown", (e) => {
    if (e.target.closest(".controls")) return;
    dragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragAz0 = azimuth;
    dragEl0 = elevation;
});

window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;
    azimuth = dragAz0 + dx * 0.005;
    elevation = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1,
        dragEl0 - dy * 0.005));
    render();
});

window.addEventListener("mouseup", () => { dragging = false; });

// Layer slider
slider.addEventListener("input", () => {
    render();
});

// Play button
playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.classList.toggle("active", playing);
    playBtn.innerHTML = playing ? "Pause &#9646;&#9646;" : "Play &#9654;";
    if (playing) {
        playTimer = setInterval(() => {
            let val = parseInt(slider.value) + 1;
            if (val > parseInt(slider.max)) {
                val = 0;
            }
            slider.value = val;
            render();
        }, 200);
    } else {
        clearInterval(playTimer);
    }
});

// Checkboxes
showLines.addEventListener("change", render);
showLabels.addEventListener("change", render);

// Initial render
render();
</script>
</body>
</html>"""


def layer_scatter_3d(projections_result, meta, detailed_preds=None,
                     width=900, height=700):
    """
    Generate interactive 3D scatterplot HTML.

    Parameters:
        projections_result: output of stable_contrastive_projections()
        meta: metadata dict from load_activations()
        detailed_preds: optional dict from load_detailed_predictions()
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        HTML string.  Use IPython.display.HTML(html) in notebooks,
        or write to a file for standalone viewing.
    """
    payload = _build_data_payload(projections_result, meta, detailed_preds)
    data_json = json.dumps(payload, separators=(",", ":"))

    html = _HTML_TEMPLATE
    html = html.replace("__WIDTH__", str(width))
    html = html.replace("__HEIGHT__", str(height))
    html = html.replace("__DATA_JSON__", data_json)
    return html


def make_layer_viz(meta_file, pred_file=None, width=900, height=700):
    """
    Convenience function: load data, compute projections, return HTML string.

    Parameters:
        meta_file: path to *_meta.json
        pred_file: optional path to *_detailed_preds.json
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        HTML string with self-contained interactive visualization.
    """
    meta_file = Path(meta_file)
    meta, layer_data, layer_indices = load_activations(meta_file)

    print(f"Computing stable 3D projections across {len(layer_indices)} layers...")
    proj_result = stable_contrastive_projections(layer_data, meta, n_components=3)

    detailed_preds = None
    if pred_file is None:
        # Auto-detect pred file
        stem = meta_file.stem.replace("_meta", "")
        candidate = meta_file.parent / f"{stem}_detailed_preds.json"
        if candidate.exists():
            pred_file = candidate
    if pred_file is not None:
        print(f"Loading detailed predictions: {Path(pred_file).name}")
        detailed_preds = load_detailed_predictions(pred_file)

    html = layer_scatter_3d(proj_result, meta, detailed_preds, width, height)
    print(f"Generated HTML: {len(html)} bytes")
    return html


# ── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    meta_file = RAW_DIR / "llama31_70b_instruct_pred_c_meta.json"
    if not meta_file.exists():
        print(f"Not found: {meta_file}")
        raise SystemExit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    html = make_layer_viz(meta_file)
    out_path = FIGURES_DIR / "pred_c_3d_layers.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")

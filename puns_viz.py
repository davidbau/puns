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
       background: #fff; color: #333; display: flex; justify-content: center; padding: 16px; }
.container { width: __WIDTH__px; }
.header { margin-bottom: 8px; }
.header h1 { font-size: 18px; font-weight: 600; color: #111; }
.header .subtitle { font-size: 13px; color: #666; margin-top: 2px; }
.var-info { font-size: 12px; color: #777; margin-top: 4px; font-family: monospace; }

.svg-wrap { position: relative; border: 1px solid #ccc; border-radius: 8px;
            background: #fafafa; overflow: hidden; cursor: grab; }
.svg-wrap:active { cursor: grabbing; }
svg { display: block; }

.controls { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
.controls label { font-size: 13px; color: #555; white-space: nowrap; }
.slider-group { flex: 1; display: flex; align-items: center; gap: 8px; min-width: 200px; }
.slider-group input[type=range] { flex: 1; accent-color: #7B68EE; }
.layer-num { font-family: monospace; font-size: 14px; color: #222; min-width: 60px; }
.btn { background: #eee; color: #444; border: 1px solid #bbb; border-radius: 4px;
       padding: 4px 12px; font-size: 12px; cursor: pointer; }
.btn:hover { background: #ddd; color: #222; }
.btn.active { background: #7B68EE; color: #fff; border-color: #7B68EE; }

.legend { display: flex; gap: 16px; margin-top: 8px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 12px; color: #555; }
.legend-swatch { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #aaa; }
.legend-star { font-size: 16px; line-height: 12px; }

.tooltip { position: absolute; pointer-events: none; background: rgba(255,255,255,0.97);
           border: 1px solid #bbb; border-radius: 6px; padding: 10px 12px;
           font-size: 12px; line-height: 1.5; max-width: 380px; z-index: 100;
           display: none; box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
.tooltip .tt-title { font-weight: 600; color: #111; margin-bottom: 4px; }
.tooltip .tt-context { color: #555; }
.tooltip .tt-sentence { color: #333; font-style: italic; margin: 4px 0; }
.tooltip .tt-detail { color: #666; font-family: monospace; font-size: 11px; }
.tooltip .tt-boost { color: #c08800; }

.checkbox-group { display: flex; align-items: center; gap: 4px; }
.checkbox-group input { accent-color: #7B68EE; }
.checkbox-group label { font-size: 12px; color: #555; cursor: pointer; }
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
    <button class="btn" id="resetBtn">Reset view</button>
    <span class="layer-num" id="zoomInfo" style="min-width:40px; font-size:12px; color:#888;"></span>
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
    <div class="legend-item" style="color:#666; margin-left:auto;">Drag=rotate | Scroll/pinch=zoom | Shift-drag=pan</div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

const W = __WIDTH__, H = __HEIGHT__;
const PAD = 50;
const COLOR_S = "#4A90D9", COLOR_F = "#E85D75";

// State
let azimuth = -0.5, elevation = 0.35;
let zoom = 1.0;
let panX = 0, panY = 0;
let currentLayer = 0;
let playing = false;
let playTimer = null;
let selectedPair = null;
let dragging = false, dragStartX = 0, dragStartY = 0, dragAz0 = 0, dragEl0 = 0;
let panning = false, panStartX = 0, panStartY = 0, panX0 = 0, panY0 = 0;
let pinching = false, pinchDist0 = 0, pinchZoom0 = 1;

// DOM
const svg = document.getElementById("mainSvg");
const svgWrap = document.getElementById("svgWrap");
const tooltip = document.getElementById("tooltip");
const slider = document.getElementById("layerSlider");
const layerNum = document.getElementById("layerNum");
const playBtn = document.getElementById("playBtn");
const showLines = document.getElementById("showLines");
const showLabels = document.getElementById("showLabels");
const resetBtn = document.getElementById("resetBtn");
const zoomInfoEl = document.getElementById("zoomInfo");
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

    const zf = zoom;
    for (let i = 0; i < n; i++) {
        const x = (pts[i][0] - cx) / scale;
        const y = (pts[i][1] - cy) / scale;
        const z = (pts[i][2] - cz) / scale;
        const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
        const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
        const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
        out[i] = {
            sx: W/2 + panX + px * (W/2 - PAD) * zf,
            sy: H/2 + panY - py * (H/2 - PAD) * zf,
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

    zoomInfoEl.textContent = zoom !== 1.0 ? (zoom.toFixed(1) + "x") : "";

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
                stroke: isSelected ? "#333" : "#aaa",
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
                    fill: "#999", "font-size": "8", "text-anchor": "middle",
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
                stroke: isSelected ? "#333" : "#c08800",
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
                stroke: isSelected ? "#333" : "rgba(0,0,0,0.15)",
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

// Project a single normalized 3D point to screen coords
function proj1(nx, ny, nz) {
    const R = rotMatrix(azimuth, elevation);
    const zf = zoom;
    const px = R[0][0]*nx + R[0][1]*ny + R[0][2]*nz;
    const py = R[1][0]*nx + R[1][1]*ny + R[1][2]*nz;
    return {
        sx: W/2 + panX + px * (W/2 - PAD) * zf,
        sy: H/2 + panY - py * (H/2 - PAD) * zf
    };
}

function drawAxes() {
    const g = svgEl("g", {});

    // Compute normalized extents: axis ranges mapped to [-1,1] using same scale as project3D
    const ranges = DATA.axisRanges;
    const sx = (ranges[0][1] - ranges[0][0]) / 2 || 1;
    const sy = (ranges[1][1] - ranges[1][0]) / 2 || 1;
    const sz = (ranges[2][1] - ranges[2][0]) / 2 || 1;
    const scale = Math.max(sx, sy, sz);
    const ex = sx / scale;  // extent in normalized coords
    const ey = sy / scale;
    const ez = sz / scale;

    // Wireframe box: 12 edges connecting 8 corners
    const corners = [];
    for (const ix of [-1, 1])
        for (const iy of [-1, 1])
            for (const iz of [-1, 1])
                corners.push([ix * ex, iy * ey, iz * ez]);

    // Edges: pairs of corners that differ in exactly one coordinate
    const edges = [];
    for (let i = 0; i < 8; i++)
        for (let j = i + 1; j < 8; j++) {
            let diffs = 0;
            for (let k = 0; k < 3; k++) if (corners[i][k] !== corners[j][k]) diffs++;
            if (diffs === 1) edges.push([i, j]);
        }

    for (const [i, j] of edges) {
        const a = proj1(corners[i][0], corners[i][1], corners[i][2]);
        const b = proj1(corners[j][0], corners[j][1], corners[j][2]);
        g.appendChild(svgEl("line", {
            x1: a.sx.toFixed(1), y1: a.sy.toFixed(1),
            x2: b.sx.toFixed(1), y2: b.sy.toFixed(1),
            stroke: "#444", "stroke-width": 0.5, opacity: 0.4
        }));
    }

    // Axis lines through origin, with labels at positive end
    const axisInfo = [
        {dir: [ex, 0, 0], label: "Contrastive", color: "#7B68EE"},
        {dir: [0, ey, 0], label: "Resid PC1",   color: "#4A90D9"},
        {dir: [0, 0, ez], label: "Resid PC2",   color: "#E85D75"},
    ];
    for (const ax of axisInfo) {
        const neg = proj1(-ax.dir[0], -ax.dir[1], -ax.dir[2]);
        const pos = proj1(ax.dir[0], ax.dir[1], ax.dir[2]);
        // Axis line
        g.appendChild(svgEl("line", {
            x1: neg.sx.toFixed(1), y1: neg.sy.toFixed(1),
            x2: pos.sx.toFixed(1), y2: pos.sy.toFixed(1),
            stroke: ax.color, "stroke-width": 1, opacity: 0.5,
            "stroke-dasharray": "4,3"
        }));
        // Label at positive end (offset slightly outward)
        const lbl = proj1(ax.dir[0] * 1.1, ax.dir[1] * 1.1, ax.dir[2] * 1.1);
        const t = svgEl("text", {
            x: lbl.sx.toFixed(1), y: lbl.sy.toFixed(1),
            fill: ax.color, "font-size": "10", "text-anchor": "middle",
            "dominant-baseline": "middle", opacity: 0.7
        });
        t.textContent = ax.label;
        g.appendChild(t);
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

// Rotation pivot: when you rotate, we orbit around the 3D point currently
// at screen center.  For the ambiguous depth coordinate, use mean data depth (≈0).
// This means if you pan to focus on a cluster, rotation orbits around it.
let pivotData = [0, 0, 0];  // pivot in normalized data space

function matTranspose3(R) {
    return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]]
    ];
}

function mat3Mul(M, v) {
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
    ];
}

function computePivot() {
    // Find what data-space point is at screen center, at depth = 0 in view space
    const S = (W/2 - PAD) * zoom;
    const viewX = -panX / (S || 1);
    const viewY = panY / (S || 1);
    const viewZ = 0;  // mean depth of centered data ≈ 0
    const Rt = matTranspose3(rotMatrix(azimuth, elevation));
    return mat3Mul(Rt, [viewX, viewY, viewZ]);
}

function panFromPivot(pivot, az, el) {
    const R = rotMatrix(az, el);
    const viewPos = mat3Mul(R, pivot);
    const S = (W/2 - PAD) * zoom;
    panX = -viewPos[0] * S;
    panY =  viewPos[1] * S;
}

// Drag rotation + shift-drag panning
svgWrap.addEventListener("mousedown", (e) => {
    if (e.target.closest(".controls")) return;
    if (e.shiftKey || e.button === 1) {
        // Shift+drag or middle-click = pan
        panning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        panX0 = panX;
        panY0 = panY;
    } else {
        dragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragAz0 = azimuth;
        dragEl0 = elevation;
        pivotData = computePivot();
    }
});

window.addEventListener("mousemove", (e) => {
    if (dragging) {
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        azimuth = dragAz0 + dx * 0.005;
        elevation = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1,
            dragEl0 - dy * 0.005));
        // Adjust pan so the pivot stays at screen center
        panFromPivot(pivotData, azimuth, elevation);
        render();
    } else if (panning) {
        panX = panX0 + (e.clientX - panStartX);
        panY = panY0 + (e.clientY - panStartY);
        render();
    }
});

window.addEventListener("mouseup", () => { dragging = false; panning = false; });

// Scroll wheel zoom (centered on cursor)
// screen_x = W/2 + panX + data_x * (W/2-PAD) * zoom
// To keep cursor point fixed: panX_new = (1-ratio)*(mx - W/2) + ratio*panX_old
svgWrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = svgWrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const delta = -e.deltaY * 0.001;
    const newZoom = Math.max(0.2, Math.min(20, zoom * (1 + delta)));
    const ratio = newZoom / zoom;

    panX = (1 - ratio) * (mx - W/2) + ratio * panX;
    panY = (1 - ratio) * (my - H/2) + ratio * panY;
    zoom = newZoom;

    render();
}, {passive: false});

// Touch: pinch-to-zoom (centered on midpoint) + single-finger rotate
function touchDist(t) {
    const dx = t[0].clientX - t[1].clientX;
    const dy = t[0].clientY - t[1].clientY;
    return Math.sqrt(dx*dx + dy*dy);
}
function touchMid(t, rect) {
    return [(t[0].clientX + t[1].clientX) / 2 - rect.left,
            (t[0].clientY + t[1].clientY) / 2 - rect.top];
}

let pinchPanX0 = 0, pinchPanY0 = 0, pinchMid0 = [0, 0];

svgWrap.addEventListener("touchstart", (e) => {
    if (e.touches.length === 2) {
        e.preventDefault();
        pinching = true;
        dragging = false;
        pinchDist0 = touchDist(e.touches);
        pinchZoom0 = zoom;
        pinchPanX0 = panX;
        pinchPanY0 = panY;
        const rect = svgWrap.getBoundingClientRect();
        pinchMid0 = touchMid(e.touches, rect);
    } else if (e.touches.length === 1) {
        dragging = true;
        dragStartX = e.touches[0].clientX;
        dragStartY = e.touches[0].clientY;
        dragAz0 = azimuth;
        dragEl0 = elevation;
        pivotData = computePivot();
    }
}, {passive: false});

svgWrap.addEventListener("touchmove", (e) => {
    if (pinching && e.touches.length === 2) {
        e.preventDefault();
        const dist = touchDist(e.touches);
        const newZoom = Math.max(0.2, Math.min(20, pinchZoom0 * dist / pinchDist0));
        const ratio = newZoom / pinchZoom0;
        const rect = svgWrap.getBoundingClientRect();
        const mid = touchMid(e.touches, rect);
        // Zoom centered on pinch midpoint (using original pinch start as reference)
        panX = (1 - ratio) * (pinchMid0[0] - W/2) + ratio * pinchPanX0 + (mid[0] - pinchMid0[0]);
        panY = (1 - ratio) * (pinchMid0[1] - H/2) + ratio * pinchPanY0 + (mid[1] - pinchMid0[1]);
        zoom = newZoom;
        render();
    } else if (dragging && e.touches.length === 1) {
        e.preventDefault();
        const dx = e.touches[0].clientX - dragStartX;
        const dy = e.touches[0].clientY - dragStartY;
        azimuth = dragAz0 + dx * 0.005;
        elevation = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1,
            dragEl0 - dy * 0.005));
        panFromPivot(pivotData, azimuth, elevation);
        render();
    }
}, {passive: false});

svgWrap.addEventListener("touchend", (e) => {
    if (e.touches.length < 2) pinching = false;
    if (e.touches.length < 1) dragging = false;
});

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

// Reset view
resetBtn.addEventListener("click", () => {
    azimuth = -0.5; elevation = 0.35;
    zoom = 1.0; panX = 0; panY = 0;
    render();
});

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

"""Prepare sparse drone bbox annotation assets from a replay run.

This exports sampled video frames plus a CSV template seeded from diagnostics.
The reviewer corrects x1/y1/x2/y2 and sets review_status=ok. Negative frames
can be labeled with label=negative.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare sparse drone bbox annotation packet")
    p.add_argument("--video", required=True, help="Input video")
    p.add_argument("--diagnostics", default="", help="Optional diagnostics.csv from a run")
    p.add_argument("--out-dir", default="annotations/drone_review", help="Annotation packet output")
    p.add_argument("--samples", type=int, default=80, help="Uniform sample count")
    p.add_argument("--stride", type=int, default=15, help="Prefer every Nth frame when diagnostics exists")
    p.add_argument("--include-frames", default="", help="Comma-separated extra frame IDs")
    p.add_argument("--negative-frames", default="", help="Comma-separated hard-negative/no-drone frame IDs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    diag = read_diagnostics(args.diagnostics) if args.diagnostics else {}
    negative_frames = set(parse_frame_list(args.negative_frames))
    selected = choose_frames(
        total,
        diag,
        args.samples,
        args.stride,
        parse_frame_list(args.include_frames) + list(negative_frames),
    )
    rows = []
    for frame_id in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        clean = f"frame_{frame_id:05d}.jpg"
        cv2.imwrite(str(frames_dir / clean), frame)
        is_negative = frame_id in negative_frames
        draft = None if is_negative else diag.get(frame_id)
        x1 = y1 = x2 = y2 = ""
        if draft is not None:
            x = float(draft["x"])
            y = float(draft["y"])
            w = float(draft["w"])
            h = float(draft["h"])
            x1, y1, x2, y2 = f"{x:.2f}", f"{y:.2f}", f"{x + w:.2f}", f"{y + h:.2f}"
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 220, 255), 2)
            cv2.imwrite(str(frames_dir / f"frame_{frame_id:05d}_draft.jpg"), overlay)
        rows.append(
            {
                "frame_id": frame_id,
                "image": f"frames/{clean}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "review_status": "draft",
                "visibility": "",
                "occluded": "",
                "label": "negative" if is_negative else "drone",
                "notes": "",
            }
        )
    cap.release()

    csv_path = out_dir / "drone_sparse_gt_template.csv"
    write_csv(csv_path, rows)
    html_path = out_dir / "bbox_annotator.html"
    write_html(html_path, rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "video": args.video,
                "diagnostics": args.diagnostics,
                "frames": len(rows),
                "instructions": (
                    "Correct boxes around the drone. Set review_status=ok for usable labels, "
                    "skip for ambiguous frames, and label=negative for hard-negative/no-drone frames."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote frames to {frames_dir}")
    return 0


def read_diagnostics(path: str) -> Dict[int, Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[int, Dict[str, str]] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row["frame_index"])
                vals = [float(row[k]) for k in ("x", "y", "w", "h")]
            except Exception:
                continue
            if vals[2] > 0 and vals[3] > 0:
                out[frame] = row
    return out


def choose_frames(total: int, diag: Dict[int, Dict[str, str]], samples: int, stride: int, extras: List[int]) -> List[int]:
    target_count = max(1, samples)
    extra_set = {f for f in extras if 0 <= f < total}
    if diag:
        candidates = sorted(diag)
        stride_candidates = candidates[::max(1, stride)]
        hard = []
        for frame, row in diag.items():
            score = 0.0
            if row.get("lock_state") in {"LOST", "ACQUIRED", "SEARCHING"}:
                score += 3.0
            try:
                score += max(0.0, 0.75 - float(row.get("confidence") or 0)) * 2.0
            except Exception:
                pass
            if score > 0:
                hard.append((score, frame))
        hard.sort(reverse=True)
        hard_frames = [frame for _, frame in hard[: max(15, target_count // 3)]]
        selected = set(extra_set)
        selected.update(hard_frames)
        remaining = max(0, target_count - len(selected))
        if remaining:
            pool = [f for f in stride_candidates if f not in selected]
            if len(pool) <= remaining:
                selected.update(pool)
            else:
                idxs = np.linspace(0, len(pool) - 1, num=remaining, dtype=int)
                selected.update(pool[int(i)] for i in idxs)
    else:
        selected = set(int(v) for v in np.linspace(0, max(0, total - 1), num=target_count, dtype=int))
        selected.update(extra_set)
    return sorted(selected)


def parse_frame_list(value: str) -> List[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "image",
                "x1",
                "y1",
                "x2",
                "y2",
                "review_status",
                "visibility",
                "occluded",
                "label",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: List[Dict[str, object]]) -> None:
    payload = base64.b64encode(json.dumps(rows).encode("utf-8")).decode("ascii")
    cards = "\n".join(_card_html(row) for row in rows)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Skyscouter Drone BBox Review</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f4f6f8; color: #111; }}
    header {{ position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #cdd4dc; padding: 14px 22px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 4px 0; line-height: 1.35; }}
    code {{ background: #eef2f6; padding: 1px 4px; border-radius: 4px; }}
    main {{ margin: 18px 22px 40px; }}
    button {{ border-radius: 6px; border: 1px solid #222; padding: 8px 12px; background: #111; color: white; cursor: pointer; }}
    button.secondary {{ background: #fff; color: #111; border-color: #8c98a4; }}
    button.warn {{ background: #7a1f1f; border-color: #7a1f1f; }}
    input, select {{ width: 100%; box-sizing: border-box; padding: 6px; border: 1px solid #aeb7c1; border-radius: 5px; background: #fff; }}
    input[readonly] {{ background: #eef1f4; color: #222; }}
    .guide {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 10px 0; font-size: 13px; }}
    .guide div {{ background: #f7f9fb; border: 1px solid #d7dee6; border-radius: 6px; padding: 8px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }}
    .toolbar label {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
    .stats {{ font-size: 13px; color: #39434d; }}
    .warn-text {{ color: #8a3000; font-weight: 700; }}
    .card {{ background: #fff; border: 1px solid #cfd6de; border-radius: 8px; padding: 14px; margin: 14px 0; }}
    .card[data-status="ok"] {{ border-left: 5px solid #187a3b; }}
    .card[data-status="skip"] {{ border-left: 5px solid #777; opacity: 0.82; }}
    .card[data-label="negative"] {{ border-left: 5px solid #9b1c1c; }}
    .card-title {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .card-title h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .pill {{ border: 1px solid #aab4be; border-radius: 999px; padding: 4px 8px; color: #303942; font-size: 12px; white-space: nowrap; }}
    .images {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 13px; font-weight: 700; margin: 0 0 6px; color: #29313a; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #8e9aa6; background: #d6dbe0; }}
    .canvas-wrap {{ position: relative; }}
    .canvas-wrap img {{ position: relative; z-index: 1; }}
    .canvas-wrap canvas {{ position: absolute; inset: 0; z-index: 2; width: 100%; height: 100%; cursor: crosshair; touch-action: none; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 12px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr; }} main {{ margin: 12px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Skyscouter Drone BBox Review</h1>
    <p>Draw or correct the box around the <strong>drone</strong> on the clean frame. The draft frame is only a starting point and may be wrong.</p>
    <p class="warn-text">Rows marked <code>draft</code> are ignored by evaluation. Mark usable drone boxes as <code>ok</code>; mark ambiguous rows as <code>skip</code>; use <code>negative</code> only when there is no drone target in the frame.</p>
    <div class="guide">
      <div><strong>How to correct:</strong> drag a new box on either image. The left clean image and the right draft image both update the same x/y values.</div>
      <div><strong>OK Drone:</strong> use when the box tightly encloses the visible drone and you are confident.</div>
      <div><strong>Occluded OK:</strong> use when the drone is partly hidden or cropped but you can still box the visible/inferable drone.</div>
      <div><strong>Negative:</strong> use only if this frame has no drone target or is a hard false-positive frame. It clears the box.</div>
      <div><strong>Skip:</strong> use when you cannot confidently label the frame. Skipped rows are ignored.</div>
    </div>
    <div class="toolbar">
      <button onclick="downloadCsv()">Download corrected CSV</button>
      <button class="secondary" onclick="markVisibleDraftsOk()">Mark visible drafts OK</button>
      <button class="secondary" onclick="showAll()">Show all</button>
      <button class="secondary" onclick="showOnlyDrafts()">Show drafts</button>
      <button class="secondary" onclick="showOnlyProblems()">Show skip/negative</button>
      <label>Jump frame <input id="jumpFrame" type="number" min="0" style="width: 90px"></label>
      <button class="secondary" onclick="jumpToFrame()">Go</button>
      <span id="stats" class="stats"></span>
    </div>
  </header>
  <main>
{cards}
  </main>
  <script>
    const initialRows = JSON.parse(decodeURIComponent(Array.prototype.map.call(atob('{payload}'), c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join('')));
    const rowByFrame = new Map(initialRows.map(r => [String(r.frame_id), r]));

    function fields(card) {{
      const out = {{}};
      card.querySelectorAll('[data-field]').forEach(el => out[el.dataset.field] = el);
      return out;
    }}
    function setCardState(card) {{
      const f = fields(card);
      card.dataset.status = f.review_status.value;
      card.dataset.label = f.label.value;
      updateStats();
    }}
    function setBox(card, box) {{
      const f = fields(card);
      f.x1.value = Number.isFinite(box.x1) ? box.x1.toFixed(2) : '';
      f.y1.value = Number.isFinite(box.y1) ? box.y1.toFixed(2) : '';
      f.x2.value = Number.isFinite(box.x2) ? box.x2.toFixed(2) : '';
      f.y2.value = Number.isFinite(box.y2) ? box.y2.toFixed(2) : '';
      drawCanvases(card);
    }}
    function getBox(card) {{
      const f = fields(card);
      return {{x1:+f.x1.value, y1:+f.y1.value, x2:+f.x2.value, y2:+f.y2.value}};
    }}
    function hasBox(card) {{
      const b = getBox(card);
      return Number.isFinite(b.x1) && Number.isFinite(b.y1) && Number.isFinite(b.x2) && Number.isFinite(b.y2) && b.x2 > b.x1 && b.y2 > b.y1;
    }}
    function drawCanvas(card, canvas) {{
      const img = canvas.closest('.canvas-wrap').querySelector('img');
      const ctx = canvas.getContext('2d');
      const w = img.naturalWidth || 1920;
      const h = img.naturalHeight || 1080;
      canvas.width = w;
      canvas.height = h;
      ctx.clearRect(0, 0, w, h);
      if (!hasBox(card)) return;
      const b = getBox(card);
      ctx.lineWidth = Math.max(3, w / 520);
      ctx.strokeStyle = fields(card).label.value === 'negative' ? '#ff3333' : '#ff00cc';
      ctx.fillStyle = fields(card).label.value === 'negative' ? 'rgba(255,51,51,0.10)' : 'rgba(255,0,204,0.10)';
      ctx.fillRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }}
    function drawCanvases(card) {{
      card.querySelectorAll('canvas').forEach(canvas => drawCanvas(card, canvas));
    }}
    function pointerToImage(e, canvas) {{
      const r = canvas.getBoundingClientRect();
      return {{ x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height }};
    }}
    document.querySelectorAll('.card').forEach(card => {{
      let start = null;
      card.querySelectorAll('.canvas-wrap img').forEach(img => img.addEventListener('load', () => drawCanvases(card)));
      card.querySelectorAll('canvas').forEach(canvas => {{
        canvas.addEventListener('pointerdown', e => {{
          start = pointerToImage(e, canvas);
          canvas.setPointerCapture(e.pointerId);
        }});
        canvas.addEventListener('pointermove', e => {{
          if (!start) return;
          const p = pointerToImage(e, canvas);
          setBox(card, {{ x1: Math.min(start.x, p.x), y1: Math.min(start.y, p.y), x2: Math.max(start.x, p.x), y2: Math.max(start.y, p.y) }});
        }});
        canvas.addEventListener('pointerup', () => {{
          if (!start) return;
          const f = fields(card);
          if (f.review_status.value === 'draft') f.review_status.value = 'ok';
          if (!f.occluded.value) f.occluded.value = '0';
          if (!f.visibility.value) f.visibility.value = '1.0';
          start = null;
          setCardState(card);
          drawCanvases(card);
        }});
      }});
      card.querySelectorAll('[data-field]').forEach(el => el.addEventListener('change', () => {{ setCardState(card); drawCanvases(card); }}));
      setCardState(card);
      drawCanvases(card);
    }});
    function cardFromButton(btn) {{ return btn.closest('.card'); }}
    function markOk(btn) {{
      const card = cardFromButton(btn);
      const f = fields(card);
      f.review_status.value = 'ok';
      f.label.value = 'drone';
      if (!f.occluded.value) f.occluded.value = '0';
      if (!f.visibility.value) f.visibility.value = '1.0';
      setCardState(card);
    }}
    function markNegative(btn) {{
      const card = cardFromButton(btn);
      const f = fields(card);
      f.review_status.value = 'ok';
      f.label.value = 'negative';
      f.x1.value = f.y1.value = f.x2.value = f.y2.value = '';
      f.occluded.value = '0';
      f.visibility.value = '0';
      setCardState(card);
      drawCanvases(card);
    }}
    function markOccluded(btn) {{
      const card = cardFromButton(btn);
      const f = fields(card);
      f.review_status.value = 'ok';
      f.label.value = 'drone';
      f.occluded.value = '1';
      if (!f.visibility.value) f.visibility.value = '0.7';
      setCardState(card);
    }}
    function markSkip(btn) {{
      const card = cardFromButton(btn);
      fields(card).review_status.value = 'skip';
      setCardState(card);
    }}
    function resetDraft(btn) {{
      const card = cardFromButton(btn);
      const r = rowByFrame.get(card.dataset.frame);
      setBox(card, {{x1:+r.x1, y1:+r.y1, x2:+r.x2, y2:+r.y2}});
      const f = fields(card);
      f.review_status.value = 'draft';
      f.label.value = r.label || 'drone';
      f.visibility.value = r.visibility || '';
      f.occluded.value = r.occluded || '';
      f.notes.value = r.notes || '';
      setCardState(card);
    }}
    function collectRows() {{
      return Array.from(document.querySelectorAll('.card')).map(card => {{
        const row = {{ frame_id: card.dataset.frame, image: `frames/frame_${{String(card.dataset.frame).padStart(5, '0')}}.jpg` }};
        card.querySelectorAll('[data-field]').forEach(el => row[el.dataset.field] = el.value);
        return row;
      }});
    }}
    function downloadCsv() {{
      const cols = ['frame_id','image','x1','y1','x2','y2','review_status','visibility','occluded','label','notes'];
      const lines = [cols.join(',')];
      for (const row of collectRows()) {{
        lines.push(cols.map(c => `"${{String(row[c] ?? '').replaceAll('"', '""')}}"`).join(','));
      }}
      const blob = new Blob([lines.join('\\n') + '\\n'], {{type: 'text/csv'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'drone_sparse_gt_corrected.csv';
      a.click();
    }}
    function updateStats() {{
      const rows = collectRows();
      const counts = rows.reduce((acc, r) => {{ acc[r.review_status] = (acc[r.review_status] || 0) + 1; acc[r.label] = (acc[r.label] || 0) + 1; return acc; }}, {{}});
      document.getElementById('stats').textContent = `total=${{rows.length}} ok=${{counts.ok||0}} draft=${{counts.draft||0}} skip=${{counts.skip||0}} negative=${{counts.negative||0}}`;
    }}
    function showAll() {{ document.querySelectorAll('.card').forEach(c => c.classList.remove('hidden')); }}
    function showOnlyDrafts() {{ document.querySelectorAll('.card').forEach(c => c.classList.toggle('hidden', fields(c).review_status.value !== 'draft')); }}
    function showOnlyProblems() {{ document.querySelectorAll('.card').forEach(c => c.classList.toggle('hidden', !(fields(c).review_status.value === 'skip' || fields(c).label.value === 'negative'))); }}
    function markVisibleDraftsOk() {{
      document.querySelectorAll('.card:not(.hidden)').forEach(card => {{
        const f = fields(card);
        if (f.review_status.value === 'draft' && hasBox(card)) {{
          f.review_status.value = 'ok';
          f.label.value = 'drone';
          if (!f.occluded.value) f.occluded.value = '0';
          if (!f.visibility.value) f.visibility.value = '1.0';
          setCardState(card);
        }}
      }});
    }}
    function jumpToFrame() {{
      const frame = String(document.getElementById('jumpFrame').value);
      const card = document.querySelector(`.card[data-frame="${{frame}}"]`);
      if (card) card.scrollIntoView({{behavior: 'smooth', block: 'start'}});
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _card_html(row: Dict[str, object]) -> str:
    frame = int(row["frame_id"])
    img = html.escape(str(row["image"]))
    draft = f"frames/frame_{frame:05d}_draft.jpg"
    x1 = html.escape(str(row.get("x1", "")))
    y1 = html.escape(str(row.get("y1", "")))
    x2 = html.escape(str(row.get("x2", "")))
    y2 = html.escape(str(row.get("y2", "")))
    status = str(row.get("review_status", "draft"))
    visibility = html.escape(str(row.get("visibility", "")))
    occluded = str(row.get("occluded", ""))
    label = str(row.get("label", "drone"))
    notes = html.escape(str(row.get("notes", "")))
    status_options = "".join(_option(v, status) for v in ("draft", "ok", "skip"))
    label_options = "".join(_option(v, label) for v in ("drone", "negative"))
    occ_options = "".join(_option(v, occluded) for v in ("", "0", "1"))
    return f"""
    <section class="card" data-frame="{frame}">
      <div class="card-title">
        <h2>Frame {frame}</h2>
        <span class="pill">draft is not truth</span>
      </div>
      <div class="images">
        <figure>
          <figcaption>Draw corrected drone box here</figcaption>
          <div class="canvas-wrap">
            <img loading="lazy" src="{img}" alt="clean frame {frame}">
            <canvas data-frame="{frame}"></canvas>
          </div>
        </figure>
        <figure>
          <figcaption>Draft tracker box - you can redraw here too</figcaption>
          <div class="canvas-wrap">
            <img loading="lazy" src="{html.escape(draft)}" alt="draft frame {frame}" onerror="this.src='{img}'">
            <canvas data-frame="{frame}" data-side="draft"></canvas>
          </div>
        </figure>
      </div>
      <div class="grid">
        <label>x1 <input value="{x1}" data-field="x1" readonly></label>
        <label>y1 <input value="{y1}" data-field="y1" readonly></label>
        <label>x2 <input value="{x2}" data-field="x2" readonly></label>
        <label>y2 <input value="{y2}" data-field="y2" readonly></label>
        <label>status <select data-field="review_status">{status_options}</select></label>
        <label>label <select data-field="label">{label_options}</select></label>
        <label>occluded <select data-field="occluded">{occ_options}</select></label>
        <label>visibility <input value="{visibility}" data-field="visibility" placeholder="0.0-1.0"></label>
        <label>notes <input value="{notes}" data-field="notes"></label>
      </div>
      <div class="actions">
        <button type="button" onclick="markOk(this)">OK Drone</button>
        <button type="button" onclick="markOccluded(this)">Occluded OK</button>
        <button type="button" class="warn" onclick="markNegative(this)">Negative</button>
        <button type="button" class="secondary" onclick="markSkip(this)">Skip</button>
        <button type="button" class="secondary" onclick="resetDraft(this)">Reset Draft</button>
      </div>
    </section>
"""


def _option(value: str, selected: str) -> str:
    sel = " selected" if value == selected else ""
    return f'<option value="{html.escape(value)}"{sel}>{html.escape(value)}</option>'


if __name__ == "__main__":
    raise SystemExit(main())

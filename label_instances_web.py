# -*- coding: utf-8 -*-
import os, csv, argparse, io
from typing import Tuple, List, Dict, Set, Optional
from filelock import FileLock
from flask import Flask, request, redirect, url_for, send_file, render_template_string, abort
import numpy as np
import cv2

NPZ_DIR = "data/sam_npz"
IMG_DIR = "data/raw_images"
LABELED_IMG_DIR = "data/labeled_images"
LABEL_DIR = "labels"
os.makedirs(LABEL_DIR, exist_ok=True)
CSV_PATH = os.path.join(LABEL_DIR, "labels.csv")
LOCK_PATH = CSV_PATH + ".lock"
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

CLASS_NAME_TO_ID = {
    "T": 1,
    "V": 2,
    "I": 3,
    "Stela": 6,
    "Fibrosis": 7,
    "Others": 0
}

LABEL_ORDER = [
    "T", "V", "I",
    "Stela", "Fibrosis",
    "Others"
]
FOLLICLE_GROUP = {"T", "V", "I"}
STRUCTURE_GROUP = {"Stela", "Fibrosis"}

CLASSES = LABEL_ORDER
EXTRA_BTNS = [("skip", -1)]

SKIP_AREA_RATIO_HIGH = 0.45
SKIP_AREA_PIX_LOW = 500

PREVIEW_MAX_WIDTH = 1600
JPEG_QUALITY = 80

IMG_CACHE: Dict[str, Optional[np.ndarray]] = {}
NPZ_CACHE: Dict[str, Tuple[np.ndarray, str, int, int]] = {}

TOTAL_VALID_MASKS: Optional[int] = None

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Follicle Labeler (T/V/I + Others)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 12px; }
    .grid { display:flex; gap:8px; flex-wrap:wrap; }
    button { padding:10px 14px; font-size:15px; cursor:pointer; border-radius:8px; border:1px solid #bbb; background:#fafafa; }
    .follicle-btn { background:#e8f5e9; border-color:#4caf50; }
    .other-btn { background:#fff3e0; border-color:#ff9800; }
    .selected { outline: 3px solid #2962ff; font-weight: 700; }
    img { max-width:98vw; height:auto; }
    .top { margin-bottom:10px; display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; }
    .muted { color:#777; font-size:14px; }
    .warning { color: #ff6600; font-weight: bold; margin-top: 6px; }
    .info { color: #2196f3; margin-top: 8px; padding: 8px; background: #e3f2fd; border-radius: 4px; }
    a { text-decoration:none; color:#3366cc; }
    a:hover { text-decoration:underline; }
    .controls { margin-left:auto; display:flex; gap:10px; align-items:center; }
    select { padding:6px 8px; font-size:14px; }
    .pill { padding:6px 10px; border-radius:8px; background:#f1f3f4; border:1px solid #d0d7de; }
    .row { margin-top:8px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  </style>
</head>
<body>
  <div class="top">
    <div><b>{{ image_name }}</b> &nbsp; [mask {{ idx+1 }}/{{ total }}] &nbsp; | &nbsp; {{ npz_name }}</div>
    <div class="muted">LABELED: {{ done_count }} &nbsp; / &nbsp; REMAIN (unlabeled): {{ remain_count }}</div>
    <div class="muted">Area: {{ mask_area }} px ({{ area_ratio }}%)</div>
    {% if overlap_warning %}
      <div class="warning">WARNING: This mask may overlap with previously labeled mask!</div>
    {% endif %}
    <form class="controls" method="GET" action="{{ url_for('index') }}">

      <label class="pill">Scope
        <select name="scope" onchange="this.form.submit()">
          <option value="unlabeled" {% if scope=='unlabeled' %}selected{% endif %}>Unlabeled</option>
          <option value="labeled" {% if scope=='labeled' %}selected{% endif %}>Labeled</option>
          <option value="all" {% if scope=='all' %}selected{% endif %}>All</option>
        </select>
      </label>
      <label class="pill">Filter
        <select name="filter_class" onchange="this.form.submit()">
          <option value="any" {% if filter_class=='any' %}selected{% endif %}>Any</option>
          {% for cname in classes %}
            <option value="{{ cname }}" {% if filter_class==cname %}selected{% endif %}>{{ cname }}</option>
          {% endfor %}
        </select>
      </label>
      <label class="pill">View
        <select name="mode" onchange="this.form.submit()">
          <option value="cur" {% if current_mode=='cur' %}selected{% endif %}>current only</option>
          <option value="all" {% if current_mode=='all' %}selected{% endif %}>show all</option>
        </select>
      </label>
    </form>
  </div>

  <div>
    <img src="{{ url_for('preview', npz=npz_name, idx=idx, mode=current_mode) }}" />
  </div>

  <div class="row muted">
    <div>Current label:
      {% if current_label %}
        <b>{{ current_label }}</b>
      {% else %}
        <i>None</i>
      {% endif %}
    </div>
  </div>

  <div style="margin-top:12px" class="grid">
    {% for cname in classes %}
      <form method="POST" action="{{ url_for('submit') }}">
        <input type="hidden" name="npz" value="{{ npz_name }}"/>
        <input type="hidden" name="idx" value="{{ idx }}"/>
        <input type="hidden" name="mode" value="{{ current_mode }}"/>
        <input type="hidden" name="scope" value="{{ scope }}"/>
        <input type="hidden" name="filter_class" value="{{ filter_class }}"/>
        <input type="hidden" name="class_name" value="{{ cname }}"/>
        <input type="hidden" name="class_id" value="{{ class_map[cname] }}"/>
        <button class="{% if cname == 'Others' %}other-btn{% elif cname in follicle_group %}follicle-btn{% else %}other-btn{% endif %} {% if current_label==cname %}selected{% endif %}">{{ cname }}</button>
      </form>
    {% endfor %}
    {% if current_label %}
      <form method="POST" action="{{ url_for('unlabel') }}">
        <input type="hidden" name="npz" value="{{ npz_name }}"/>
        <input type="hidden" name="idx" value="{{ idx }}"/>
        <input type="hidden" name="mode" value="{{ current_mode }}"/>
        <input type="hidden" name="scope" value="{{ scope }}"/>
        <input type="hidden" name="filter_class" value="{{ filter_class }}"/>
        <button>Unlabel</button>
      </form>
    {% endif %}
    {% for ex,cid in extra %}
      <form method="POST" action="{{ url_for('submit') }}">
        <input type="hidden" name="npz" value="{{ npz_name }}"/>
        <input type="hidden" name="idx" value="{{ idx }}"/>
        <input type="hidden" name="mode" value="{{ current_mode }}"/>
        <input type="hidden" name="scope" value="{{ scope }}"/>
        <input type="hidden" name="filter_class" value="{{ filter_class }}"/>
        <input type="hidden" name="class_name" value="{{ ex }}"/>
        <input type="hidden" name="class_id" value="{{ cid }}"/>
        <button>{{ ex }}</button>
      </form>
    {% endfor %}
  </div>

  <div style="margin-top:12px">
    <form method="GET" action="{{ url_for('prev') }}" style="display:inline;">
      <input type="hidden" name="npz" value="{{ npz_name }}"/>
      <input type="hidden" name="idx" value="{{ idx }}"/>
      <input type="hidden" name="mode" value="{{ current_mode }}"/>
      <input type="hidden" name="scope" value="{{ scope }}"/>
      <input type="hidden" name="filter_class" value="{{ filter_class }}"/>
      <button>&larr; prev</button>
    </form>
    <form method="GET" action="{{ url_for('skip_route') }}" style="display:inline;">
      <input type="hidden" name="npz" value="{{ npz_name }}"/>
      <input type="hidden" name="idx" value="{{ idx }}"/>
      <input type="hidden" name="mode" value="{{ current_mode }}"/>
      <input type="hidden" name="scope" value="{{ scope }}"/>
      <input type="hidden" name="filter_class" value="{{ filter_class }}"/>
      <button>next &rarr;</button>
    </form>
  </div>
  
  <div class="info">
    <b>Guide</b>
    <ul>
      <li><b>Goal:</b> Assign one label to the highlighted mask of a single follicle region: T, V, I, Stela, Fibrosis or Others. You can also review and correct existing labels.</li>
    </ul>
  
    <b>What you see</b>
    <ul>
      <li><b>Left panel:</b> The saved labeled image if available, otherwise the raw image.</li>
      <li><b>Right panel:</b> The current image with masks. In "current only" view, only the selected mask is emphasized; other areas are dimmed. In "show all," all masks are color blended.</li>
      <li><b>Header counters:</b> LABELED shows how many instances already have labels. REMAIN shows how many valid masks are still unlabeled.</li>
      <li><b>Mask info:</b> Area and percentage help you spot masks that are too small or too large.</li>
      <li><b>Overlap warning:</b> If shown, the current mask overlaps a previously labeled mask in the same image. Verify you are labeling the correct ring or contour of the follicle.</li>
    </ul>
  
    <b>Labels</b>
    <ul>
      <li><b>T (Terminal):</b> Coarse, thick hair follicle with clear terminal features.</li>
      <li><b>V (Vellus):</b> Fine, thin hair follicle with clear vellus features.</li>
      <li><b>I (Indeterminate):</b> It is a follicle, but the type is uncertain.</li>
      <li><b>Stela:</b> Hair germ/stream-like structure around follicle.</li>
      <li><b>Fibrosis:</b> Fibrotic tissue around or between follicles.</li>
      <li><b>Others:</b> Non-target regions or artifacts not covered above; kept for compatibility.</li>
    </ul>
  
    <b>Scope selector</b>
    <ul>
      <li><b>Unlabeled:</b> Show only masks that have no label yet. Use this to annotate new data.</li>
      <li><b>Labeled:</b> Show only masks that already have a label. Use this to review and correct.</li>
      <li><b>All:</b> Show both labeled and unlabeled masks.</li>
    </ul>
  
    <b>Filter (for review)</b>
    <ul>
      <li>Choose Any to see all items in the selected scope.</li>
      <li>Choose T, V, I, Stela, Fibrosis or Others to review only that label. For example, pick T to audit all masks labeled as T.</li>
      <li>In All scope, the filter still applies only to labeled items; unlabeled items are shown when Filter is Any.</li>
    </ul>
  
    <b>View mode</b>
    <ul>
      <li><b>current only:</b> Emphasizes the selected mask and dims the rest for clear inspection.</li>
      <li><b>show all:</b> Blends all masks with colors to show global context and boundaries.</li>
    </ul>
  
    <b>How to label</b>
    <ol>
      <li>Select the desired Scope and (optionally) a Filter.</li>
      <li>Use Next or Prev to move between masks that match the current Scope and Filter.</li>
      <li>Inspect the highlighted mask on the right. If needed, switch View to check surrounding context.</li>
      <li>Click one label button. The choice is saved immediately.</li>
      <li>If the item already had a label, clicking another label updates it. Use Unlabel to remove the label completely.</li>
    </ol>
  
    <b>Keyboard</b>
    <ul>
      <li><b>Space:</b> Next (same as the Next button). This skips without changing any label.</li>
    </ul>
  
    <b>When to skip</b>
    <ul>
      <li>Inner ring or duplicate ring of the same follicle that should not be labeled separately.</li>
      <li>Image region is too ambiguous and you prefer not to decide now.</li>
    </ul>
  
    <b>Quality checks</b>
    <ul>
      <li>Confirm the mask truly corresponds to a follicle before choosing T, V, or I.</li>
      <li>Use I when it is a follicle but type is unclear; do not force T or V.</li>
      <li>Use Others for non-follicle structures. Do not label artifacts as follicles.</li>
      <li>Use Labeled scope with a Filter (for example T) to quickly audit and correct a specific class.</li>
    </ul>
  
    <b>Troubleshooting</b>
    <ul>
      <li>If overlap warning appears, verify you are labeling the intended ring.</li>
      <li>If the highlighted area is extremely small or huge, double check it is a meaningful mask for labeling.</li>
      <li>If no items appear, adjust Scope or set Filter to Any.</li>
    </ul>
  </div>

  <script>
    document.addEventListener('keydown', (e)=>{
      if(e.code === 'Space'){
        const forms = document.getElementsByTagName('form');
        for (let f of forms) {
          if (f.action.endsWith('/skip')) { f.submit(); return; }
        }
      }
    });
  </script>
</body>
</html>
"""

def load_labels_map() -> Dict[Tuple[str, int], Tuple[str, int]]:
    labels = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    key = (row["image_name"], int(row["mask_idx"]))
                    labels[key] = (row["class_name"], int(row["class_id"]))
                except Exception:
                    continue
    return labels

def write_labels(rows: List[Dict[str, str]]) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "mask_idx", "class_name", "class_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def set_label(image_name: str, mask_idx: int, class_name: str, class_id: int) -> None:
    lock = FileLock(LOCK_PATH)
    with lock:
        rows = []
        found = False
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if row["image_name"] == image_name and int(row["mask_idx"]) == mask_idx:
                        row["class_name"] = class_name
                        row["class_id"] = str(class_id)
                        found = True
                    rows.append(row)
        if not found:
            rows.append({
                "image_name": image_name,
                "mask_idx": str(mask_idx),
                "class_name": class_name,
                "class_id": str(class_id)
            })
        write_labels(rows)

def delete_label(image_name: str, mask_idx: int) -> None:
    lock = FileLock(LOCK_PATH)
    with lock:
        rows = []
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not (row["image_name"] == image_name and int(row["mask_idx"]) == mask_idx):
                        rows.append(row)
        write_labels(rows)

def list_npz() -> List[str]:
    if not os.path.isdir(NPZ_DIR):
        return []
    return sorted([p for p in os.listdir(NPZ_DIR) if p.endswith(".npz")])

def load_npz(npz_name: str):
    if npz_name in NPZ_CACHE:
        return NPZ_CACHE[npz_name]
    z = np.load(os.path.join(NPZ_DIR, npz_name), allow_pickle=False)
    masks = z["masks"]
    image_name = str(z["image_name"])
    H = int(z["H"])
    W = int(z["W"])
    NPZ_CACHE[npz_name] = (masks, image_name, H, W)
    return masks, image_name, H, W

def is_valid_mask(masks: np.ndarray, idx: int) -> bool:
    m = masks[idx].astype(np.uint8)
    area = int(m.sum())
    H, W = m.shape
    if area < SKIP_AREA_PIX_LOW:
        return False
    if area > SKIP_AREA_RATIO_HIGH * H * W:
        return False
    return True

def sorted_indices_by_area(masks: np.ndarray, descending: bool = True) -> List[int]:
    areas = [(i, int(masks[i].sum())) for i in range(masks.shape[0])]
    areas.sort(key=lambda x: x[1], reverse=descending)
    return [i for i, _ in areas]

def check_overlap(masks: np.ndarray, idx: int, done_keys: Set[Tuple[str, int]], image_name: str) -> bool:
    current_mask = masks[idx].astype(bool)
    current_area = current_mask.sum()
    for other_idx in range(masks.shape[0]):
        if other_idx == idx:
            continue
        if (image_name, other_idx) in done_keys:
            other_mask = masks[other_idx].astype(bool)
            overlap = (current_mask & other_mask).sum()
            if overlap > 0.3 * current_area:
                return True
    return False

def imread_cached(path: str) -> Optional[np.ndarray]:
    if path in IMG_CACHE:
        return IMG_CACHE[path]
    img = cv2.imread(path)
    IMG_CACHE[path] = img
    return img

def make_overlay(image_name: str, masks: np.ndarray, idx: int, mode: str = "cur"):
    left_path = resolve_labeled_path(image_name)
    right_path = resolve_raw_path(image_name)

    left_bgr = imread_cached(left_path)
    right_bgr = imread_cached(right_path)
    if right_bgr is None:
        return None
    if left_bgr is None:
        left_bgr = right_bgr.copy()

    M, H, W = masks.shape
    if idx < 0 or idx >= M:
        return None

    left_bgr = cv2.resize(left_bgr, (W, H), interpolation=cv2.INTER_AREA)
    right_bgr = cv2.resize(right_bgr, (W, H), interpolation=cv2.INTER_AREA)

    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    cur = masks[idx].astype(bool)

    if mode == "all":
        overlay = right_rgb.astype(np.float32).copy()
        for i in range(M):
            m = masks[i].astype(bool)
            c = (np.random.RandomState(i + 123).rand(3) * 255).astype(np.float32)
            overlay[m] = overlay[m] * 0.5 + c * 0.5
        vis = cv2.addWeighted(right_rgb, 0.6, overlay.astype(np.uint8), 0.4, 0)
    else:
        hsv = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        S, V = hsv[..., 1], hsv[..., 2]
        not_cur = ~cur
        S[not_cur] *= 0.25
        V[not_cur] *= 0.70
        hsv[..., 1], hsv[..., 2] = S, V
        base = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        fill = np.array([30, 200, 255], np.float32)
        alpha = 0.70
        out = base.astype(np.float32)
        out[cur] = out[cur] * (1 - alpha) + fill * alpha
        vis = np.clip(out, 0, 255).astype(np.uint8)

    cur_u8 = (cur.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(cur_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.drawContours(vis_bgr, contours, -1, (255, 120, 40), thickness=4, lineType=cv2.LINE_AA)

    concat = cv2.hconcat([left_bgr, vis_bgr])

    h, w = concat.shape[:2]
    if w > PREVIEW_MAX_WIDTH:
        scale = PREVIEW_MAX_WIDTH / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        concat = cv2.resize(concat, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return concat

def resolve_with_exts(root_dir: str, image_name: str) -> str:
    p = os.path.join(root_dir, image_name)
    if os.path.isfile(p):
        return p
    base, ext = os.path.splitext(image_name)
    for e in IMG_EXTS:
        cand = os.path.join(root_dir, base + e)
        if os.path.isfile(cand):
            return cand
    return os.path.join(root_dir, image_name)

def resolve_labeled_path(image_name: str) -> str:
    p = resolve_with_exts(LABELED_IMG_DIR, image_name)
    if os.path.isfile(p):
        return p
    return resolve_with_exts(IMG_DIR, image_name)

def resolve_raw_path(image_name: str) -> str:
    return resolve_with_exts(IMG_DIR, image_name)

def is_candidate(image_name: str, masks: np.ndarray, i: int,
                 scope: str, filter_class: str,
                 labels_map: Dict[Tuple[str, int], Tuple[str, int]]) -> bool:
    if not is_valid_mask(masks, i):
        return False
    labeled = (image_name, i) in labels_map
    if scope == "unlabeled":
        return not labeled
    elif scope == "labeled":
        if not labeled:
            return False
        if filter_class and filter_class != "any":
            return labels_map[(image_name, i)][0] == filter_class
        return True
    elif scope == "all":
        if filter_class and filter_class != "any":
            if not labeled:
                return False
            return labels_map[(image_name, i)][0] == filter_class
        return True
    return False

def compute_total_valid_masks() -> int:
    global TOTAL_VALID_MASKS
    if TOTAL_VALID_MASKS is not None:
        return TOTAL_VALID_MASKS
    cnt = 0
    for zname in list_npz():
        masks, image_name, H, W = load_npz(zname)
        for i in range(masks.shape[0]):
            if is_valid_mask(masks, i):
                cnt += 1
    TOTAL_VALID_MASKS = cnt
    return cnt

def count_remaining_unlabeled(done_keys: Set[Tuple[str, int]]) -> int:
    total_valid = compute_total_valid_masks()
    labeled_valid = len(done_keys)
    rem = total_valid - labeled_valid
    if rem < 0:
        rem = 0
    return rem

def first_candidate(scope: str, filter_class: str,
                    labels_map: Dict[Tuple[str, int], Tuple[str, int]]):
    for npz_name in list_npz():
        masks, image_name, H, W = load_npz(npz_name)
        for idx in sorted_indices_by_area(masks, descending=True):
            if is_candidate(image_name, masks, idx, scope, filter_class, labels_map):
                return npz_name, image_name, masks.shape[0], idx
    return None, None, 0, -1

def find_next(npz_name: str, idx: int, forward: bool,
              scope: str, filter_class: str,
              labels_map: Dict[Tuple[str, int], Tuple[str, int]]):
    npzs = list_npz()
    if not npzs:
        return None
    try:
        start_j = npzs.index(npz_name)
    except ValueError:
        start_j = 0

    if forward:
        masks, image_name, H, W = load_npz(npz_name)
        order = sorted_indices_by_area(masks, descending=True)
        pos = order.index(idx) if idx in order else -1
        for i in order[pos + 1:]:
            if is_candidate(image_name, masks, i, scope, filter_class, labels_map):
                return (npz_name, i)
        for j in range(start_j + 1, len(npzs)):
            m2, img2, _, _ = load_npz(npzs[j])
            for k in sorted_indices_by_area(m2, descending=True):
                if is_candidate(img2, m2, k, scope, filter_class, labels_map):
                    return (npzs[j], k)
        return None
    else:
        masks, image_name, H, W = load_npz(npz_name)
        order = sorted_indices_by_area(masks, descending=True)
        pos = order.index(idx) if idx in order else len(order)
        for i in reversed(order[:pos]):
            if is_candidate(image_name, masks, i, scope, filter_class, labels_map):
                return (npz_name, i)
        for j in range(start_j - 1, -1, -1):
            m2, img2, _, _ = load_npz(npzs[j])
            order2 = sorted_indices_by_area(m2, descending=True)
            for k in reversed(order2):
                if is_candidate(img2, m2, k, scope, filter_class, labels_map):
                    return (npzs[j], k)
        return None

app = Flask(__name__)

@app.route("/")
def index():
    npz = request.args.get("npz", default=None, type=str)
    idx = request.args.get("idx", default=None, type=int)
    mode = request.args.get("mode", default="cur", type=str)
    scope = request.args.get("scope", default="unlabeled", type=str)
    filter_class = request.args.get("filter_class", default="any", type=str)
    if filter_class not in (["any"] + CLASSES):
        filter_class = "any"

    labels_map = load_labels_map()
    done_keys = set(labels_map.keys())

    if npz is None:
        npz, image_name, M, idx = first_candidate(scope, filter_class, labels_map)
        if npz is None:
            return f"No items match the current filter (scope={scope}, filter={filter_class})."
    else:
        masks, image_name, H, W = load_npz(npz)
        M = masks.shape[0]
        if idx is None:
            found_local = None
            for i in sorted_indices_by_area(masks, descending=True):
                if is_candidate(image_name, masks, i, scope, filter_class, labels_map):
                    found_local = i
                    break
            if found_local is None:
                nxt = find_next(npz, -1, True, scope, filter_class, labels_map)
                if not nxt:
                    return f"No items match the current filter (scope={scope}, filter={filter_class})."
                npz, idx = nxt[0], nxt[1]
            else:
                idx = found_local

    masks, image_name, H, W = load_npz(npz)
    if not is_candidate(image_name, masks, idx, scope, filter_class, labels_map):
        nxt = find_next(npz, idx, True, scope, filter_class, labels_map)
        if nxt:
            return redirect(url_for("index", npz=nxt[0], idx=nxt[1], mode=mode, scope=scope, filter_class=filter_class))
        else:
            return f"No items match the current filter (scope={scope}, filter={filter_class})."

    remain = count_remaining_unlabeled(done_keys)
    mask_area = int(masks[idx].sum())
    area_ratio = round(100 * mask_area / (H * W), 2)
    overlap_warning = check_overlap(masks, idx, done_keys, image_name)
    current_label = labels_map.get((image_name, idx), (None, None))[0]

    return render_template_string(
        TEMPLATE,
        npz_name=npz,
        idx=len(masks) if idx is None else idx,
        total=masks.shape[0],
        image_name=image_name,
        classes=CLASSES,
        class_map=CLASS_NAME_TO_ID,
        follicle_group=FOLLICLE_GROUP,
        structure_group=STRUCTURE_GROUP,
        done_count=len(done_keys),
        remain_count=remain,
        current_mode=mode,
        mask_area=mask_area,
        area_ratio=area_ratio,
        overlap_warning=overlap_warning,
        scope=scope,
        filter_class=filter_class,
        current_label=current_label,
        extra=EXTRA_BTNS
    )

@app.route("/preview")
def preview():
    npz_name = request.args.get("npz", type=str)
    idx = request.args.get("idx", type=int)
    mode = request.args.get("mode", default="cur", type=str)
    if npz_name is None:
        abort(404)
    masks, image_name, H, W = load_npz(npz_name)
    vis = make_overlay(image_name, masks, idx, mode=mode)
    if vis is None:
        abort(404)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, buf = cv2.imencode(".jpg", vis, encode_param)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")

@app.route("/submit", methods=["POST"])
def submit():
    npz_name = request.form.get("npz")
    idx = int(request.form.get("idx"))
    mode = request.form.get("mode", "cur")
    scope = request.form.get("scope", "unlabeled")
    filter_class = request.form.get("filter_class", "any")
    class_name = request.form.get("class_name")
    class_id = int(request.form.get("class_id"))

    masks, image_name, H, W = load_npz(npz_name)

    if class_id == -1:
        labels_map = load_labels_map()
        nxt = find_next(npz_name, idx, True, scope, filter_class, labels_map)
        if not nxt:
            return redirect(url_for("index", mode=mode, scope=scope, filter_class=filter_class))
        return redirect(url_for("index", npz=nxt[0], idx=nxt[1], mode=mode, scope=scope, filter_class=filter_class))

    set_label(image_name, idx, class_name, class_id)
    labels_map = load_labels_map()
    nxt = find_next(npz_name, idx, True, scope, filter_class, labels_map)
    if nxt:
        return redirect(url_for("index", npz=nxt[0], idx=nxt[1], mode=mode, scope=scope, filter_class=filter_class))
    return redirect(url_for("index", mode=mode, scope=scope, filter_class=filter_class))

@app.route("/unlabel", methods=["POST"])
def unlabel():
    npz_name = request.form.get("npz")
    idx = int(request.form.get("idx"))
    mode = request.form.get("mode", "cur")
    scope = request.form.get("scope", "unlabeled")
    filter_class = request.form.get("filter_class", "any")

    masks, image_name, H, W = load_npz(npz_name)
    delete_label(image_name, idx)
    labels_map = load_labels_map()
    nxt = find_next(npz_name, idx, True, scope, filter_class, labels_map)
    if nxt:
        return redirect(url_for("index", npz=nxt[0], idx=nxt[1], mode=mode, scope=scope, filter_class=filter_class))
    return redirect(url_for("index", mode=mode, scope=scope, filter_class=filter_class))

@app.route("/prev")
def prev():
    npz_name = request.args.get("npz")
    idx = int(request.args.get("idx", 0))
    mode = request.args.get("mode", "cur")
    scope = request.args.get("scope", "unlabeled")
    filter_class = request.args.get("filter_class", "any")

    labels_map = load_labels_map()
    prv = find_next(npz_name, idx, False, scope, filter_class, labels_map)
    if prv:
        return redirect(url_for("index", npz=prv[0], idx=prv[1], mode=mode, scope=scope, filter_class=filter_class))
    return redirect(url_for("index", mode=mode, scope=scope, filter_class=filter_class))

@app.route("/skip")
def skip_route():
    npz_name = request.args.get("npz")
    idx = int(request.args.get("idx", 0))
    mode = request.args.get("mode", "cur")
    scope = request.args.get("scope", "unlabeled")
    filter_class = request.args.get("filter_class", "any")

    labels_map = load_labels_map()
    nxt = find_next(npz_name, idx, True, scope, filter_class, labels_map)
    if nxt:
        return redirect(url_for("index", npz=nxt[0], idx=nxt[1], mode=mode, scope=scope, filter_class=filter_class))
    return redirect(url_for("index", mode=mode, scope=scope, filter_class=filter_class))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)

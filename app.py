from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io

app = FastAPI()

# تحميل الموديل
print("Loading model...")
model = tf.keras.models.load_model('/content/best_oral_model.h5')
class_names = ['Calculus', 'Caries', 'Gingivitis', 'Tooth Discoloration', 'Ulcer']

# شرح مبسط لكل حالة (بيظهر للمستخدم مع النتيجة)
descriptions = {
    'Calculus': 'Hardened plaque (tartar) on the teeth, usually near the gum line.',
    'Caries': 'Tooth decay: areas where the enamel has been damaged by acid.',
    'Gingivitis': 'Inflamed, red or swollen gums, often caused by plaque build-up.',
    'Tooth Discoloration': 'Teeth that look yellow, brown or stained compared with normal enamel.',
    'Ulcer': 'A sore or open spot on the soft tissue inside the mouth.',
}

# ملحوظة: القالب مش f-string عشان أقواس CSS/JS تفضل زي ما هي
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oral disease classification</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --paper: #f1f7f6;
    --sheet: #ffffff;
    --ink: #12292d;
    --muted: #5b7378;
    --line: #d3e2df;
    --teal: #0f766e;
    --teal-dark: #0b5953;
    --mist: #e3efed;
    --warn: #b45309;
    --warn-bg: #fdf3e3;
    --err: #b42318;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { color-scheme: light; }
  body {
    font-family: 'Manrope', system-ui, sans-serif;
    background: var(--paper);
    color: var(--ink);
    min-height: 100vh;
    line-height: 1.55;
    padding: 28px 20px 40px;
  }
  .wrap { max-width: 1040px; margin: 0 auto; }

  header { display: flex; align-items: center; gap: 10px; margin-bottom: 22px; }
  header .mark {
    width: 34px; height: 34px; border-radius: 10px;
    background: var(--teal); color: #fff;
    display: grid; place-items: center; font-size: 18px;
  }
  header span { font-weight: 700; letter-spacing: .01em; }

  .sheet {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 22px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    overflow: hidden;
  }
  .pane { padding: 40px; }
  .pane + .pane { border-left: 1px solid var(--line); background: #fafdfc; }

  h1 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: clamp(30px, 4vw, 42px);
    line-height: 1.1;
    letter-spacing: -.01em;
    margin-bottom: 12px;
  }
  .lead { color: var(--muted); max-width: 40ch; margin-bottom: 28px; }

  /* منطقة رفع الصورة */
  .drop {
    position: relative;
    display: grid; place-items: center;
    aspect-ratio: 4 / 3;
    border: 2px dashed #9cc3bd;
    border-radius: 16px;
    background: var(--mist);
    cursor: pointer;
    overflow: hidden;
    text-align: center;
    transition: border-color .2s, background .2s;
  }
  .drop:hover, .drop.over { border-color: var(--teal); background: #d6e9e6; }
  .drop:focus-within { outline: 3px solid var(--teal); outline-offset: 3px; }
  .drop input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .drop .hint { padding: 20px; pointer-events: none; }
  .drop .hint strong { display: block; font-size: 17px; margin: 8px 0 2px; }
  .drop .hint small { color: var(--muted); }
  .drop .hint .icon { font-size: 34px; }
  .drop img {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; display: none;
  }
  .drop.has-image img { display: block; }
  .drop.has-image .hint { display: none; }

  .actions { display: flex; gap: 10px; margin-top: 16px; }
  button {
    font: inherit; font-weight: 600; cursor: pointer;
    border-radius: 12px; padding: 13px 20px; border: 0;
    transition: background .2s, opacity .2s;
  }
  button:focus-visible { outline: 3px solid var(--teal); outline-offset: 2px; }
  .primary { background: var(--teal); color: #fff; flex: 1; }
  .primary:hover:not(:disabled) { background: var(--teal-dark); }
  .primary:disabled { opacity: .45; cursor: not-allowed; }
  .ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
  .ghost:hover { background: var(--mist); }
  .ghost[hidden] { display: none; }

  /* لوحة النتيجة */
  .empty { color: var(--muted); max-width: 34ch; }
  .empty h2 { font-size: 18px; color: var(--ink); margin-bottom: 6px; }

  .loading { display: none; align-items: center; gap: 12px; color: var(--muted); }
  .spinner {
    width: 22px; height: 22px; border-radius: 50%;
    border: 3px solid var(--mist); border-top-color: var(--teal);
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .result { display: none; }
  .result .label { color: var(--muted); font-size: 14px; }
  .result .name {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: clamp(34px, 5vw, 50px);
    line-height: 1.05;
    margin: 4px 0 6px;
  }
  .result .conf { font-weight: 700; color: var(--teal); font-size: 18px; }
  .result .desc { color: var(--muted); margin: 14px 0 22px; max-width: 42ch; }

  .notice {
    background: var(--warn-bg); color: var(--warn);
    border-radius: 12px; padding: 12px 14px; font-size: 14px; margin-bottom: 20px;
  }
  .notice[hidden] { display: none; }

  .bars { list-style: none; display: grid; gap: 12px; }
  .bars li { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; font-size: 14px; }
  .bars .pct { font-variant-numeric: tabular-nums; color: var(--muted); }
  .bars .track {
    grid-column: 1 / -1; height: 8px; border-radius: 99px; background: var(--mist); overflow: hidden;
  }
  .bars .fill { height: 100%; width: 0; background: #9cc3bd; border-radius: 99px; transition: width .7s cubic-bezier(.2,.7,.2,1); }
  .bars li.top .fill { background: var(--teal); }
  .bars li.top .nm { font-weight: 700; }

  .error { display: none; color: var(--err); font-weight: 600; }

  .disclaimer {
    margin-top: 22px; font-size: 12.5px; color: var(--muted); max-width: 60ch;
  }

  @media (max-width: 820px) {
    .sheet { grid-template-columns: 1fr; }
    .pane { padding: 26px 22px; }
    .pane + .pane { border-left: 0; border-top: 1px solid var(--line); }
  }
  @media (prefers-reduced-motion: reduce) {
    .spinner { animation-duration: 2s; }
    .bars .fill { transition: none; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="mark" aria-hidden="true">🦷</div>
    <span>Oral Disease AI</span>
  </header>

  <main class="sheet">
    <section class="pane">
      <h1>Check a dental photo</h1>
      <p class="lead">Upload a clear, well-lit photo of the teeth or gums and the model will suggest the most likely condition.</p>

      <label class="drop" id="drop">
        <input type="file" id="file" accept="image/*">
        <img id="preview" alt="Selected photo preview">
        <div class="hint">
          <div class="icon" aria-hidden="true">📷</div>
          <strong>Drop a photo here or browse</strong>
          <small>JPG or PNG</small>
        </div>
      </label>

      <div class="actions">
        <button class="primary" id="analyze" disabled>Analyze photo</button>
        <button class="ghost" id="reset" hidden>Choose another</button>
      </div>
    </section>

    <section class="pane" aria-live="polite">
      <div class="empty" id="empty">
        <h2>No result yet</h2>
        <p>Add a photo and select Analyze photo. The result and the model's confidence for each condition will show up here.</p>
      </div>

      <div class="loading" id="loading"><div class="spinner"></div><span>Analyzing photo…</span></div>
      <p class="error" id="error"></p>

      <div class="result" id="result">
        <div class="label">Most likely condition</div>
        <div class="name" id="name"></div>
        <div class="conf" id="conf"></div>
        <p class="desc" id="desc"></p>
        <div class="notice" id="notice" hidden>The model is not very sure about this one. Try a closer, sharper photo with good light.</div>
        <ul class="bars" id="bars"></ul>
      </div>

      <p class="disclaimer">This tool is for screening and learning only. It does not replace an examination by a dentist.</p>
    </section>
  </main>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const drop = $('drop'), fileInput = $('file'), preview = $('preview');
  const analyzeBtn = $('analyze'), resetBtn = $('reset');
  const emptyEl = $('empty'), loadingEl = $('loading'), errorEl = $('error'), resultEl = $('result');
  let currentFile = null;

  function showOnly(which) {
    emptyEl.style.display = which === 'empty' ? 'block' : 'none';
    loadingEl.style.display = which === 'loading' ? 'flex' : 'none';
    resultEl.style.display = which === 'result' ? 'block' : 'none';
    errorEl.style.display = which === 'error' ? 'block' : 'none';
  }

  function setFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    currentFile = file;
    preview.src = URL.createObjectURL(file);
    drop.classList.add('has-image');
    analyzeBtn.disabled = false;
    resetBtn.hidden = false;
    showOnly('empty');
  }

  fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));

  resetBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    preview.removeAttribute('src');
    drop.classList.remove('has-image');
    analyzeBtn.disabled = true;
    resetBtn.hidden = true;
    showOnly('empty');
  });

  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    analyzeBtn.disabled = true;
    showOnly('loading');
    try {
      const body = new FormData();
      body.append('file', currentFile);
      const res = await fetch('/predict', { method: 'POST', body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Something went wrong.');
      render(data);
    } catch (err) {
      errorEl.textContent = err.message + ' Check the photo and try again.';
      showOnly('error');
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  function render(data) {
    $('name').textContent = data.prediction;
    $('conf').textContent = data.confidence.toFixed(1) + '% confidence';
    $('desc').textContent = data.description;
    $('notice').hidden = data.confidence >= 60;

    const bars = $('bars');
    bars.innerHTML = '';
    data.all.forEach((item, i) => {
      const li = document.createElement('li');
      if (i === 0) li.className = 'top';
      li.innerHTML =
        '<span class="nm"></span><span class="pct">' + item.score.toFixed(1) + '%</span>' +
        '<div class="track"><div class="fill"></div></div>';
      li.querySelector('.nm').textContent = item.label;
      bars.appendChild(li);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        li.querySelector('.fill').style.width = item.score + '%';
      }));
    });
    showOnly('result');
  }

  showOnly('empty');
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_TEMPLATE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "This file could not be read as an image."}, status_code=400)

    # معالجة الصورة
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized)
    img_reshape = np.expand_dims(img_array, axis=0)

    # التنبؤ
    probs = model.predict(img_reshape, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = int(order[0])

    return {
        "prediction": class_names[top],
        "confidence": float(100 * probs[top]),
        "description": descriptions[class_names[top]],
        "all": [{"label": class_names[int(i)], "score": float(100 * probs[int(i)])} for i in order],
    }

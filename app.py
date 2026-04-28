"""
Brain Tumour MRI Detection — Flask Web Application (with Dataset Upload)
Major Project (BCAMP23601) — TMU Moradabad
Team: Mobin Shaikh (TCA2301361)
"""
import os, zipfile, shutil, threading, subprocess, sys
import numpy as np 
import webbrowser 
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = "tmu_brain_tumor_2025"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB ZIP allowed

UPLOAD_FOLDER  = os.path.join("static", "uploads")
DATASET_FOLDER = "dataset_sample"
MODEL_PATH     = "model/brain_tumor_model.h5"
ALLOWED_IMG    = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_ZIP    = {"zip"}
IMG_SIZE       = 150
CLASSES        = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
DESCRIPTIONS = {
    'Glioma Tumor':     "Gliomas develop from glial cells. Requires immediate medical consultation.",
    'Meningioma Tumor': "Meningiomas form on membranes covering the brain. Most are benign but need monitoring.",
    'No Tumor':         "No tumor detected. The brain appears healthy. ✅",
    'Pituitary Tumor':  "Pituitary tumors develop in the pituitary gland. Often treatable."
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- Training state (in-memory) ----------
TRAIN_STATE = {"running": False, "log": "", "done": False}

# ---------- Lazy-load TF model ----------
_model = None
def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        print("🧠 Loading model...")
        _model = load_model(MODEL_PATH)
    return _model

def reload_model():
    global _model
    _model = None
    return get_model()

def allowed(name, exts):
    return "." in name and name.rsplit(".", 1)[1].lower() in exts

# ---------- Dataset stats ----------
def dataset_stats():
    stats = {}
    if not os.path.isdir(DATASET_FOLDER):
        return stats
    for cls in sorted(os.listdir(DATASET_FOLDER)):
        p = os.path.join(DATASET_FOLDER, cls)
        if os.path.isdir(p):
            count = sum(1 for f in os.listdir(p)
                        if allowed(f, ALLOWED_IMG))
            stats[cls] = count
    return stats

# ---------- Predict ----------
def predict_image(filepath):
    model = get_model()
    img = Image.open(filepath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img) / 255.0, 0)
    if model is None:
        np.random.seed(abs(hash(filepath)) % (2**32))
        probs = np.random.dirichlet(np.ones(4) * 2)
    else:
        probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]) * 100, {c: float(p)*100 for c, p in zip(CLASSES, probs)}

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html", stats=dataset_stats(), model_ready=os.path.exists(MODEL_PATH))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: flash("No file uploaded"); return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "" or not allowed(file.filename, ALLOWED_IMG):
        flash("Invalid file. Use PNG/JPG/JPEG/BMP"); return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    label, conf, probs = predict_image(filepath)
    return render_template("result.html",
                           image_path=filepath.replace("\\","/"),
                           label=label, confidence=round(conf,2),
                           description=DESCRIPTIONS.get(label,""),
                           probs=probs, demo_mode=get_model() is None)

# ===================== DATASET UPLOAD =====================
@app.route("/dataset", methods=["GET"])
def dataset_page():
    return render_template("dataset.html",
                           stats=dataset_stats(),
                           train_state=TRAIN_STATE)

@app.route("/dataset/upload", methods=["POST"])
def dataset_upload():
    """Upload a ZIP containing class folders → extract into dataset_sample/."""
    if "zipfile" not in request.files:
        flash("No ZIP file selected"); return redirect(url_for("dataset_page"))
    f = request.files["zipfile"]
    if f.filename == "" or not allowed(f.filename, ALLOWED_ZIP):
        flash("Please upload a .zip file"); return redirect(url_for("dataset_page"))

    tmp_zip = os.path.join("static", "uploads", secure_filename(f.filename))
    f.save(tmp_zip)
    replace = request.form.get("replace") == "yes"

    try:
        if replace and os.path.isdir(DATASET_FOLDER):
            for item in os.listdir(DATASET_FOLDER):
                p = os.path.join(DATASET_FOLDER, item)
                if os.path.isdir(p): shutil.rmtree(p)

        with zipfile.ZipFile(tmp_zip, "r") as z:
            members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
            # Detect single top-level wrapper folder (e.g. "BrainTumor/glioma/...")
            tops = {m.split("/")[0] for m in members if "/" in m}
            strip = ""
            if len(tops) == 1:
                only = list(tops)[0]
                # Check that subdirs exist (i.e. it's a wrapper, not a class itself)
                has_subdirs = any(m.startswith(f"{only}/") and m.count("/") >= 2 for m in members)
                if has_subdirs: strip = only + "/"

            extracted_imgs = 0
            for m in members:
                if m.endswith("/"): continue
                rel = m[len(strip):] if strip and m.startswith(strip) else m
                if not rel or "/" not in rel: continue
                if not allowed(rel, ALLOWED_IMG): continue
                cls = rel.split("/")[0].lower().strip()
                fname = os.path.basename(rel)
                if not fname: continue
                out_dir = os.path.join(DATASET_FOLDER, cls)
                os.makedirs(out_dir, exist_ok=True)
                with z.open(m) as src, open(os.path.join(out_dir, fname), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_imgs += 1

        os.remove(tmp_zip)
        flash(f"✅ Successfully extracted {extracted_imgs} images into dataset_sample/")
    except zipfile.BadZipFile:
        flash("❌ Invalid ZIP file")
    except Exception as e:
        flash(f"❌ Error: {e}")
    return redirect(url_for("dataset_page"))

@app.route("/dataset/clear", methods=["POST"])
def dataset_clear():
    if os.path.isdir(DATASET_FOLDER):
        for item in os.listdir(DATASET_FOLDER):
            p = os.path.join(DATASET_FOLDER, item)
            if os.path.isdir(p): shutil.rmtree(p)
    flash("🗑️ Dataset cleared")
    return redirect(url_for("dataset_page"))

# ===================== TRAINING =====================
def _train_worker():
    TRAIN_STATE.update(running=True, done=False, log="🚀 Training started...\n")
    try:
        proc = subprocess.Popen([sys.executable, "train_model.py"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            TRAIN_STATE["log"] += line
        proc.wait()
        TRAIN_STATE["log"] += f"\n✅ Finished (exit {proc.returncode})\n"
        reload_model()
    except Exception as e:
        TRAIN_STATE["log"] += f"\n❌ Error: {e}\n"
    finally:
        TRAIN_STATE.update(running=False, done=True)

@app.route("/dataset/train", methods=["POST"])
def dataset_train():
    if TRAIN_STATE["running"]:
        flash("⚠️ Training already in progress"); return redirect(url_for("dataset_page"))
    stats = dataset_stats()
    if not stats or sum(stats.values()) < 20:
        flash("❌ Dataset too small. Upload more images first (min 20)."); return redirect(url_for("dataset_page"))
    TRAIN_STATE.update(running=True, done=False, log="")
    threading.Thread(target=_train_worker, daemon=True).start()
    flash("🧠 Training started in background. Check log below.")
    return redirect(url_for("dataset_page"))

@app.route("/dataset/train-status")
def train_status():
    return jsonify(TRAIN_STATE)
import requests

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    API_KEY = "YOUR_API_KEY"

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + API_KEY

    payload = {
        "contents": [{
            "parts": [{"text": user_msg}]
        }]
    }

    response = requests.post(url, json=payload)
    data = response.json()

    try:
        reply = data['candidates'][0]['content']['parts'][0]['text']
    except:
        reply = "Sorry, I couldn't understand."

    return jsonify({"reply": reply})
# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Brain Tumour MRI Detection — Flask App")
    print("📍 TMU Moradabad | Major Project BCAMP23601")
    print("👥 Team: Hasan Kadari, Chirag Jain, Mobin Shaikh")
    print("=" * 60)
    print("🚀 Server: http://127.0.0.1:5000")
    print("📂 Dataset upload page: http://127.0.0.1:5000/dataset")
    print("=" * 60)

    webbrowser.open("http://127.0.0.1:5001")   

    app.run(debug=True, host="0.0.0.0", port=5001)
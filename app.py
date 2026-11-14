"""Web demo cho dự án gki_AI: python app.py  ->  http://localhost:5000"""
import os, threading
import numpy as np, torch
from flask import Flask, request, jsonify, send_from_directory
import backend
backend.use_graphics = False          # tắt matplotlib khi chạy web
from models import PerceptronModel, RegressionModel, DigitClassificationModel, LanguageIDModel
from train import train_perceptron, train_regression, train_digitclassifier, train_languageid

app = Flask(__name__, static_folder="static")
W = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(W, exist_ok=True)
MODELS, STATUS, LOCK = {}, {}, threading.Lock()

# Tạo dataset *chỉ khi thật sự cần huấn luyện* để tiết kiệm RAM trên hosting.
def _build(name):
    if name == "digit": return DigitClassificationModel(), backend.DigitClassificationDataset, train_digitclassifier
    if name == "reg":   return RegressionModel(), backend.RegressionDataset, train_regression
    if name == "lang":  return LanguageIDModel(), backend.LanguageIDDataset, train_languageid
    raise ValueError("Unknown model: " + name)

def load_or_train(name):
    STATUS[name] = "loading"
    try:
        model, dataset_class, train_func = _build(name)
        path = os.path.join(W, name + ".pt")
        if os.path.exists(path):
            # Máy chủ Render chỉ chạy CPU, không cần mở dữ liệu huấn luyện MNIST.
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        elif os.environ.get("TRAIN_IF_MISSING", "1") == "1":
            STATUS[name] = "training"
            dataset = dataset_class(model)
            train_func(model, dataset)
            torch.save(model.state_dict(), path)
        else:
            STATUS[name] = "missing_weights"
            print(f"[DEPLOY] Thiếu {path}; hãy đưa file trọng số từ máy vào weights/", flush=True)
            return
        model.eval()
        MODELS[name] = model
        STATUS[name] = "ready"
    except Exception as exc:
        STATUS[name] = "error"
        print(f"[DEPLOY] Lỗi tải {name}: {exc}", flush=True)

def boot():
    for name in ("digit", "reg", "lang"):
        load_or_train(name)

for name in ("digit", "reg", "lang"):
    STATUS[name] = "loading"
threading.Thread(target=boot, daemon=True).start()

def need(name):
    state = STATUS.get(name)
    if state == "ready":
        return None
    if state == "missing_weights":
        return jsonify(error=f"Thiếu file weights/{name}.pt trên máy chủ"), 503
    if state == "error":
        return jsonify(error=f"Mô hình {name} không tải được; kiểm tra Logs trên Render"), 503
    return jsonify(error="Mô hình đang được tải hoặc huấn luyện, thử lại sau"), 503

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/status")
def status(): return jsonify(STATUS)

@app.post("/api/digit")
def digit():
    if (e := need("digit")): return e
    x = torch.tensor(request.json["pixels"], dtype=torch.float32).reshape(1, 784)
    with torch.no_grad(): p = torch.softmax(MODELS["digit"](x), 1)[0]
    return jsonify(probs=p.tolist(), pred=int(p.argmax()))

@app.post("/api/lang")
def lang():
    if (e := need("lang")): return e
    m = MODELS["lang"]
    chars = list(np.load(backend.get_data_path("lang_id.npz"))["chars"])
    word = "".join(c for c in request.json["word"].lower() if c in chars)[:10]
    if not word: return jsonify(error="Nhập từ có chữ cái Latin (a-z, ą, ł, ñ...)"), 400
    xs = torch.zeros(10, 1, 47)                      # giống lúc huấn luyện: đệm 0 đến 10 bước
    for t, c in enumerate(word): xs[t, 0, chars.index(c)] = 1
    with torch.no_grad(): p = torch.softmax(m(xs), 1)[0]
    return jsonify(langs=m.languages, probs=p.tolist(), word=word)

@app.get("/api/regression")
def regression():
    if (e := need("reg")): return e
    x = np.linspace(-2*np.pi, 2*np.pi, 200)
    with torch.no_grad(): y = MODELS["reg"](torch.tensor(x, dtype=torch.float32).reshape(-1, 1)).flatten()
    return jsonify(x=x.tolist(), real=np.sin(x).tolist(), pred=y.tolist())

@app.get("/api/perceptron")
def perceptron():
    n = 150; X = np.hstack([np.random.randn(n, 2), np.ones((n, 1))]).astype(np.float32)
    y = np.where(X[:, 0] + 2*X[:, 1] - 1 >= 0, 1, -1)
    m = PerceptronModel(3); hist = [m.get_weights().detach().flatten().tolist()]
    with torch.no_grad():
        for _ in range(1000):
            err = 0
            for i in np.random.permutation(n):
                xi = torch.tensor(X[i])
                if m.get_prediction(xi) != y[i]:
                    m.w += int(y[i]) * xi; err += 1; hist.append(m.w.flatten().tolist())
            if err == 0: break
    return jsonify(points=X[:, :2].tolist(), labels=y.tolist(), history=hist)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)

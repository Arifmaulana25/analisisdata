from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import subprocess
import os
from pathlib import Path

app = FastAPI(title="Analisis Data Python API")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "tmp"
TEMP_DIR.mkdir(exist_ok=True)


def run_script(args, cwd=None, timeout=300):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        stdout = proc.stdout
        stderr = proc.stderr
        return proc.returncode, stdout, stderr
    except Exception as e:
        return 1, "", str(e)


@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...), file_id: int = Form(...)):
    # Save uploaded file to temp
    temp_path = TEMP_DIR / file.filename
    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    script = BASE_DIR / "preprocess.py"
    if not script.exists():
        return JSONResponse(status_code=500, content={"error": f"Script not found: {script}"})

    # Run the existing script: python preprocess.py <file_path> <file_id>
    args = ["python", str(script), str(temp_path), str(file_id)]
    rc, out, err = run_script(args)
    if rc != 0:
        return JSONResponse(status_code=500, content={"error": "Python script failed", "stderr": err, "stdout": out})

    # Try to parse output as JSON (script prints JSON)
    try:
        import json
        result = json.loads(out)
    except Exception:
        result = {"stdout": out, "stderr": err}

    # Optionally remove temp file
    try:
        temp_path.unlink()
    except Exception:
        pass

    return JSONResponse(content=result)


@app.post("/process")
async def process(file: UploadFile = File(...), n_clusters: int = Form(...), file_id: int = Form(...), init_method: str = Form(None)):
    temp_path = TEMP_DIR / file.filename
    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    script = BASE_DIR / "clustering_only.py"
    if not script.exists():
        return JSONResponse(status_code=500, content={"error": f"Script not found: {script}"})

    # clustering_only.py usage: clustering_only.py <file_path> <n_clusters> <file_id> [init_method]
    args = ["python", str(script), str(temp_path), str(n_clusters), str(file_id)]
    if init_method:
        args.append(str(init_method))
    rc, out, err = run_script(args)
    if rc != 0:
        return JSONResponse(status_code=500, content={"error": "Python script failed", "stderr": err, "stdout": out})

    try:
        import json
        result = json.loads(out)
    except Exception:
        result = {"stdout": out, "stderr": err}

    try:
        temp_path.unlink()
    except Exception:
        pass

    return JSONResponse(content=result)


@app.post("/prediction")
async def prediction(tipe_file: str = Form(...), bulan: int = Form(...), tahun: int = Form(...), window_size: int = Form(...), prediction_id: int = Form(...)):
    script = BASE_DIR / "prediction_ma.py"
    if not script.exists():
        return JSONResponse(status_code=500, content={"error": f"Script not found: {script}"})

    args = ["python", str(script), str(tipe_file), str(bulan), str(tahun), str(window_size), str(prediction_id)]
    rc, out, err = run_script(args, timeout=600)
    if rc != 0:
        return JSONResponse(status_code=500, content={"error": "Python script failed", "stderr": err, "stdout": out})

    # Return stdout/stderr (script writes to DB and returns success)
    try:
        import json
        result = json.loads(out)
    except Exception:
        result = {"stdout": out, "stderr": err}

    return JSONResponse(content=result)

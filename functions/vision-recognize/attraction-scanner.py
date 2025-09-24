import os, json, base64, time, uuid, re, sys
import boto3

# ---- Константы ----
ALLOWED_MIME = {b"image/jpeg": "jpeg", b"image/png": "png"}
MAX_BYTES = 5 * 1024 * 1024

# ---- ENV ----
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
MODEL_ID       = os.getenv("MODEL_ID", "meta.llama3-2-90b-instruct-v1:0")
LANDMARKS      = json.loads(os.getenv("LANDMARKS_JSON", "[]"))
DRY_RUN        = os.getenv("DRY_RUN", "0") == "1"

# Ленивая и безопасная инициализация клиента
_bedrock = None
def bedrock_client():
    global _bedrock
    if _bedrock is None:
        region = (os.getenv("BEDROCK_REGION", "us-east-1") or "us-east-1").strip().replace(" ", "")
        _bedrock = boto3.client("bedrock-runtime", region_name=region)
    return _bedrock

# ---- Lambda entrypoint ----
def lambda_handler(event, context):
    t0 = time.time()
    request_id = str(uuid.uuid4())
    try:
        headers = {(k or "").lower(): v for k, v in (event.get("headers") or {}).items()}

        # Тело должно быть base64 (REST+proxy + binary types)
        if not event.get("isBase64Encoded"):
            return _resp(400, {"message": "Body must be base64", "request_id": request_id})

        content_type = headers.get("content-type") or headers.get("Content-Type")
        if not content_type or "multipart/form-data" not in content_type:
            return _resp(400, {"message": "Expected multipart/form-data", "request_id": request_id})

        body_bytes = base64.b64decode(event["body"])
        fields = parse_multipart(body_bytes, content_type)

        lang = (fields.get("lang") or "ru").lower()
        persona = (fields.get("persona") or "formal").lower()
        file = fields.get("file")
        if not file:
            return _resp(400, {"message": "file is required", "request_id": request_id})

        mime = file["content_type"]
        img_bytes = file["content"]
        if len(img_bytes) > MAX_BYTES:
            return _resp(413, {"message": "File too large", "request_id": request_id})
        if mime not in ALLOWED_MIME:
            return _resp(415, {"message": "Unsupported media type", "request_id": request_id})

        # Отладка окружения
        print(f"DBG landmarks={len(LANDMARKS)} region={BEDROCK_REGION} model={MODEL_ID} dry_run={DRY_RUN}", file=sys.stderr)

        # ---- Классификация ----
        if DRY_RUN:
            det = {"id": LANDMARKS[0]["id"], "confidence": 0.92} if LANDMARKS else None
        else:
            try:
                det = classify_with_llama(img_bytes, ALLOWED_MIME[mime], LANDMARKS)
            except Exception as e:
                print("ERROR classify_with_llama:", repr(e), file=sys.stderr)
                det = None

        latency = int((time.time() - t0) * 1000)

        if not det:
            return _resp(200, {
                "lang": lang, "persona": persona, "latency_ms": latency, "request_id": request_id,
                "detections": [], "answer": msg_no_object(lang)
            })

        name_local = localized_name(det["id"], lang)
        confidence = float(det.get("confidence", 0.0))

        if confidence < 0.7:
            return _resp(200, {
                "lang": lang, "persona": persona, "latency_ms": latency, "request_id": request_id,
                "detections": [{"name": name_local, "id": det["id"], "confidence": round(confidence, 2)}],
                "answer": msg_try_closer(lang)
            })

        return _resp(200, {
            "lang": lang, "persona": persona, "latency_ms": latency, "request_id": request_id,
            "detections": [{"name": name_local, "id": det["id"], "confidence": round(confidence, 2)}],
            "answer": msg_success(name_local, lang, persona)
        })

    except Exception as e:
        return _resp(500, {"message": f"Internal error: {e}", "request_id": request_id})

# ---- Helpers ----
def _resp(code: int, obj: dict):
    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,X-Requested-With,Accept,Origin,x-api-key"
        },
        "body": json.dumps(obj, ensure_ascii=False)
    }

def localized_name(landmark_id: str, lang: str) -> str:
    for lm in LANDMARKS:
        if lm.get("id") == landmark_id:
            return lm.get(f"name_{lang}", lm.get("name_ru", landmark_id))
    return landmark_id

def msg_no_object(lang: str) -> str:
    return {
        "ru": "Не удалось распознать достопримечательность.",
        "kk": "Нысанды тану мүмкін болмады.",
        "en": "Could not recognize the landmark."
    }.get(lang, "Не удалось распознать достопримечательность.")

def msg_try_closer(lang: str) -> str:
    return {
        "ru": "Попробуйте сфотографировать ближе.",
        "kk": "Жақынырақ түсіріп көріңіз.",
        "en": "Try taking a closer photo."
    }.get(lang, "Попробуйте сфотографировать ближе.")

def msg_success(name: str, lang: str, persona: str) -> str:
    if lang == "kk":
        return f"Супер! Бұл {name} – квест тапсырмасы орындалды!"
    if lang == "en":
        return f"Great! That’s {name} — quest checkpoint completed!"
    return f"Супер! Это {name} — квест засчитан!"

def classify_with_llama(image_bytes: bytes, img_format: str, landmarks: list) -> dict | None:
    if not landmarks:
        print("DBG no landmarks → skip model", file=sys.stderr)
        return None

    lines = []
    for lm in landmarks:
        name_en = lm.get("name_en", lm["id"])
        hint = lm.get("hint_en") or ""
        lines.append(f'- {name_en} (id: {lm["id"]}){" - " + hint if hint else ""}')

    instruction = (
        "You are an extremely strict image classifier for landmarks in Astana, Kazakhstan.\n"
        "Choose ONE best match from the list or return id=\"none\" if no match.\n"
        "Return ONLY compact JSON: {\"id\":\"...\",\"confidence\":0..1}.\n"
        "Candidates:\n" + "\n".join(lines)
    )

    messages = [{
        "role": "user",
        "content": [
            {"text": instruction},
            {"image": {"format": img_format, "source": {"bytes": image_bytes}}}
        ]
    }]

    bedrock = bedrock_client()
    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 150, "temperature": 0}
    )

    text_out = ""
    for b in resp["output"]["message"]["content"]:
        if "text" in b:
            text_out += b["text"]

    m = re.search(r'\{[^{}]*"id"\s*:\s*"([^"]+)"[^{}]*"confidence"\s*:\s*([0-9.]+)[^{}]*\}', text_out)
    if not m:
        return None
    det_id = m.group(1).strip()
    conf = float(m.group(2))
    if det_id.lower() == "none":
        return None

    known_ids = {lm["id"] for lm in landmarks if "id" in lm}
    if det_id not in known_ids:
        det_low = det_id.lower()
        for kid in known_ids:
            if kid in det_low:
                det_id = kid
                break
        else:
            return None
    return {"id": det_id, "confidence": conf}

def parse_multipart(body: bytes, content_type: str) -> dict:
    """
    Разбор multipart/form-data. Возвращает:
      {"file":{"filename":..., "content_type": b"image/jpeg", "content": b"..."},
       "lang":"ru","persona":"formal"}
    """
    m = re.search(r'boundary="?([^";]+)"?', content_type, re.IGNORECASE)
    if not m:
        raise ValueError("boundary not found")
    boundary = m.group(1).encode()
    delimiter = b"--" + boundary
    end_delimiter = b"--" + boundary + b"--"

    out = {}
    chunks = body.split(delimiter)
    for ch in chunks:
        ch = ch.strip()
        if not ch or ch == b"--" or ch == end_delimiter:
            continue

        h_end = ch.find(b"\r\n\r\n")
        if h_end == -1:
            continue
        raw_headers = ch[:h_end].decode("utf-8", "ignore")
        data = ch[h_end + 4:].rstrip(b"\r\n")

        disp = None
        ctype = None
        for line in raw_headers.split("\r\n"):
            L = line.lower()
            if L.startswith("content-disposition:"):
                disp = line
            elif L.startswith("content-type:"):
                ctype = line.split(":", 1)[1].strip().encode()

        if not disp:
            continue

        name_m = re.search(r'name="([^"]+)"', disp, re.IGNORECASE)
        if not name_m:
            continue
        name = name_m.group(1)

        file_m = re.search(r'filename="([^"]+)"', disp, re.IGNORECASE)
        if file_m:
            out["file"] = {
                "filename": file_m.group(1),
                "content_type": ctype or b"application/octet-stream",
                "content": data
            }
        else:
            out[name] = data.decode("utf-8", "ignore")

    return out

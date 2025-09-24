# lambda_function.py
# Runtime: Python 3.11
# Env required:
#   KNOWLEDGE_BASE_ID       (e.g. 4AG1Y91PXO)
#   MODEL_ID                (e.g. meta.llama3-8b-instruct-v1:0)
# Optional:
#   AWS_REGION              (default: us-east-1)
#   LLAMA_TEMPERATURE       (default: 0.3)
#   LLAMA_MAX_TOKENS        (default: 512)
#   RETRIEVAL_TOP_K         (default: 6)

import json
import os
import re
import time
from typing import Any, Dict, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ---------- ENV ----------
REGION = os.getenv("AWS_REGION", "us-east-1").strip()
KB_ID = (os.getenv("KNOWLEDGE_BASE_ID") or "").strip()
MODEL_ID = (os.getenv("MODEL_ID") or "meta.llama3-8b-instruct-v1:0").strip()

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

BASE_TEMP = _env_float("LLAMA_TEMPERATURE", 0.3)
MAX_TOKENS = _env_int("LLAMA_MAX_TOKENS", 512)
TOP_K = _env_int("RETRIEVAL_TOP_K", 6)

if not KB_ID:
    raise RuntimeError("KNOWLEDGE_BASE_ID env var is not set")

FOUNDATION_MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}"

# ---------- CLIENTS ----------
bedrock_agent_rt = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "standard"}),
)

translate = boto3.client(
    "translate",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "standard"}),
)

# ---------- PROMPT TEMPLATE (исправлены плейсхолдеры: $query$ и $search_results$) ----------
PROMPT_RU_TEMPLATE = """Ты — ассистент для туристов Астаны. Отвечай ТОЛЬКО по фактам из базы знаний.
Соблюдай выбранный стиль общения (persona):

- formal — нейтральный, сухой, деловой тон.
- friendly — тёплый, приветливый, доброжелательный.
- humorous — лёгкий, безобидный юмор (1 короткая шутка максимум), без перехода на личности.

Пример одного и того же ответа в разных стилях:
Вопрос: «Какие есть базары в Астане?»
formal → "Лид: Базары в Астане. Факты: … Итог: Для уточнения деталей свяжитесь…"
friendly → "Лид: В Астане вас ждут колоритные базары! Факты: … Итог: Уточняйте по телефонам и заглядывайте в гости."
humorous → "Лид: Настроены на шопинг с казахским колоритом? Базары уже ждут! Факты: … Итог: Позвоните и загляните — кошелёк держите покрепче."

Стиль общения (persona): {persona}

Формат ответа (строго):
Лид (1 строка): короткое описание темы.
Факты (3–5 пунктов): каждый пункт в формате "ключевая сущность — факт (адрес, время, цена, контакты, длительность)".
Если цифр/данных нет в базе — явно пиши: “в базе нет данных по цене/адресу/...”.
Итог (1 строка): что делать дальше — добавляй только если в базе есть конкретные данные для действия.

Правила:
- Объём ответа: максимум 120–150 слов.
- Без приветствий и прощаний.
- Без собственных рассуждений и ссылок на источники.
- Используй ТОЛЬКО факты из фрагментов.
- Только простой текст: без markdown, без таблиц, без символа '|'.

Вопрос пользователя: $query$

Фрагменты базы (используй ТОЛЬКО их):
$search_results$
"""

# ---------- CLEANUP ----------
NOISE_RE = re.compile(r"(?:</?SYS>|</?INST>|\[/?SYS]|\[/INST])", re.IGNORECASE)
OUT_TAG_RE = re.compile(r"<out>(.*?)</out>", re.DOTALL | re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[^>]+>")
DANGEROUS_SECTIONS_RE = re.compile(
    r"(?:^|\n)\s*(?:</out>|\*{0,2}\s*output\s*:|\*{0,2}\s*note\s*:|output\s*:|note\s*:).*$",
    re.IGNORECASE | re.DOTALL,
)

def _clean_noise(text: str) -> str:
    if not text:
        return text
    t = NOISE_RE.sub("", text)
    t = HTML_TAG_RE.sub("", t)
    return t.strip()

def _extract_out(text: str) -> str:
    if not text:
        return ""
    m = OUT_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    end = text.lower().find("</out>")
    if end != -1:
        return text[:end].strip()
    return text.strip()

def _strip_after_markers(text: str) -> str:
    return DANGEROUS_SECTIONS_RE.sub("", text).strip()

def _strip_markdown_symbols(text: str) -> str:
    t = text.replace("|", " ")
    t = t.replace("**", "").replace("*", "")
    t = t.replace("`", "")
    t = re.sub(r"^[ \t]*#{1,6}\s*", "", t, flags=re.MULTILINE)
    return t

def _final_sanitize(text: str) -> str:
    t = _extract_out(text)
    t = _strip_after_markers(t)
    t = _clean_noise(t)
    t = _strip_markdown_symbols(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

# ---------- Citations ----------
def _extract_citations(rag_response: Dict[str, Any]) -> List[Dict[str, str]]:
    cites: List[Dict[str, str]] = []
    for c in rag_response.get("citations", []):
        for prov in c.get("generatedResponsePart", {}).get("citations", []):
            for ref in prov.get("retrievedReferences", []):
                uri = (
                    ref.get("location", {}).get("s3Location", {}).get("uri")
                    or ref.get("location", {}).get("webLocation", {}).get("url")
                    or ""
                )
                frag = ref.get("content", {}).get("text", "")
                if uri:
                    cites.append({"uri": uri, "snippet": frag[:600]})
        for ref in c.get("retrievedReferences", []):
            uri = (
                ref.get("location", {}).get("s3Location", {}).get("uri")
                or ref.get("location", {}).get("webLocation", {}).get("url")
                or ""
            )
            frag = ref.get("content", {}).get("text", "")
            if uri:
                cites.append({"uri": uri, "snippet": frag[:600]})
    if not cites and "retrievedReferences" in rag_response:
        for ref in rag_response.get("retrievedReferences", []):
            uri = (
                ref.get("location", {}).get("s3Location", {}).get("uri")
                or ref.get("location", {}).get("webLocation", {}).get("url")
                or ""
            )
            frag = ref.get("content", {}).get("text", "")
            if uri:
                cites.append({"uri": uri, "snippet": frag[:600]})
    return cites[:8]

# ---------- Persona helpers ----------
def _persona_temperature(base: float, persona: str) -> float:
    p = (persona or "").lower().strip()
    if p == "formal":
        return max(0.1, base - 0.15)
    if p == "humorous":
        return min(1.0, base + 0.4)
    return min(1.0, base + 0.2)  # friendly

def _persona_or_default(p: str) -> str:
    p = (p or "").lower().strip()
    if p in ("friendly", "formal", "humorous"):
        return p
    return "friendly"

# ---------- Bedrock KB RAG (на русском) ----------
def _retrieve_and_generate_ru(query: str, persona: str, temperature: float) -> Dict[str, Any]:
    prompt_text = PROMPT_RU_TEMPLATE.format(persona=persona)
    req = {
        "input": {"text": query},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KB_ID,
                "modelArn": FOUNDATION_MODEL_ARN,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": int(TOP_K)}
                },
                "generationConfiguration": {
                    "promptTemplate": {"textPromptTemplate": prompt_text},
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": int(MAX_TOKENS),
                            "temperature": float(temperature),
                            "topP": 0.9,
                        }
                    },
                },
            }
        }
    }
    return bedrock_agent_rt.retrieve_and_generate(**req)

# ---------- Translation ----------
def _translate_from_ru(text: str, target_lang: str) -> str:
    tl = target_lang.lower()
    tgt = "kk" if tl.startswith("kk") else "en"
    resp = translate.translate_text(
        Text=text,
        SourceLanguageCode="ru",
        TargetLanguageCode=tgt,
    )
    return resp.get("TranslatedText", "").strip() or text

# ---------- HTTP helpers ----------
def _parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if isinstance(body, str):
            try:
                return json.loads(body)
            except Exception:
                return {}
        elif isinstance(body, dict):
            return body
        return {}
    return event if isinstance(event, dict) else {}

def _resp(status: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*",
            "access-control-allow-headers": "Content-Type, x-api-key, Authorization",
            "access-control-allow-methods": "POST, OPTIONS",
            "access-control-max-age": "3600",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }

# ---------- Handler ----------
def lambda_handler(event, context):
    t0 = time.time()
    try:
        data = _parse_event(event)
        text = (data.get("text") or "").strip()
        lang = (data.get("lang") or "ru").strip().lower()
        persona_in = _persona_or_default(data.get("persona"))

        if not text:
            return _resp(400, {"error": "bad_request", "detail": "Missing 'text' in request body"})

        temp_used = _persona_temperature(BASE_TEMP, persona_in)

        rag = _retrieve_and_generate_ru(text, persona_in, temp_used)
        raw_answer_ru = rag.get("output", {}).get("text", "") or ""
        cleaned_ru = _final_sanitize(raw_answer_ru)

        if lang.startswith("kk") or lang.startswith("en"):
            final_answer = _translate_from_ru(cleaned_ru, lang)
        else:
            final_answer = cleaned_ru

        citations = _extract_citations(rag)
        latency_ms = int((time.time() - t0) * 1000)

        return _resp(200, {
            "lang": lang,
            "persona": persona_in,
            "temperature_used": round(temp_used, 3),
            "answer": final_answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "request_id": getattr(context, "aws_request_id", None),
        })

    except ClientError as e:
        return _resp(500, {"error": "inference_failed", "detail": str(e)})
    except Exception as e:
        return _resp(500, {"error": "inference_failed", "detail": str(e)})

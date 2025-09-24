# Baiterek Demo ML

Serverless AI services for the Astana guide, built on **AWS API Gateway + Lambda + Amazon Bedrock**.

This repo contains two Lambdas exposed via API Gateway:

* **`/ask`** — text Q\&A (Meta Llama Instruct)
* **`/recognize`** — image landmark recognition (Meta Llama 3.2 Vision via **inference profile**)

---

## Repository layout

```
functions/
  ai-guide/                  # /ask Lambda (text)
    env/.env.example
    src/lambda_function.py
    src/requirements.txt
  vision-recognize/          # /recognize Lambda (vision)
    env/.env.example
    src/lambda_function.py
    src/requirements.txt
LICENSE
README.md
```

`.gitignore` should include:

```
__pycache__/
*.pyc
.venv/
.env
```

---

## Prerequisites

* AWS account with access to **API Gateway**, **Lambda**, **CloudWatch**, **IAM**, **Amazon Bedrock**
* Python 3.11+ for local installs
* For the **vision** Lambda:

  * Region: **us-east-1**
  * **Inference profile enabled** for Meta Llama **3.2 90B Vision Instruct** (ID: `us.meta.llama3-2-90b-instruct-v1:0`)
  * Lambda IAM role allowing:

    * `bedrock:InvokeModel`
    * `bedrock:InvokeModelWithResponseStream`

---

## Environment variables

### `/ask` (functions/ai-guide/env/.env.example)

```dotenv
# Recommended: call via inference profile in us-east-1
BEDROCK_REGION=us-east-1
MODEL_ID=us.meta.llama3-2-70b-instruct-v1:0
```

### `/recognize` (functions/vision-recognize/env/.env.example)

```dotenv
# Bedrock via inference profile (required in us-east-1 for Llama 3.2 Vision)
BEDROCK_REGION=us-east-1
MODEL_ID=us.meta.llama3-2-90b-instruct-v1:0

# Landmarks list (SINGLE LINE JSON, UTF-8)
LANDMARKS_JSON=[{"id":"bayterek","name_ru":"Байтерек","name_kk":"Бәйтерек","name_en":"Baiterek","hint_en":"tall white monument with observation deck"},{"id":"khan_shatyr","name_ru":"Хан Шатыр","name_kk":"Хан Шатыр","name_en":"Khan Shatyr","hint_en":"giant tent-shaped mall"},{"id":"nur_alem","name_ru":"Нур Алем","name_kk":"Нұр Әлем","name_en":"Nur Alem","hint_en":"spherical museum from Expo 2017"},{"id":"astana_mall","name_ru":"Астана Молл","name_kk":"Астана Молл","name_en":"Astana Mall","hint_en":"Modern shopping mall with entertainment and dining"},{"id":"mega_silk_way","name_ru":"Мега Силк Уэй","name_kk":"Мега Силк Уэй","name_en":"Mega Silk Way","hint_en":"Huge shopping and entertainment complex"},{"id":"victory_park","name_ru":"Парк Победы","name_kk":"Жеңіс Паркі","name_en":"Victory Park","hint_en":"Memorial park dedicated to the victory in World War II"},{"id":"central_park","name_ru":"Центральный парк культуры и отдыха","name_kk":"Орталық мәдениет және демалыс саябағы","name_en":"Central Park of Culture and Leisure","hint_en":"Green space with amusement rides and recreational areas"},{"id":"kazakhstan_opera_theater","name_ru":"Казахстанский театр оперы и балета","name_kk":"Қазақстан опера және балет театры","name_en":"Kazakhstan Opera and Ballet Theatre","hint_en":"Leading theater for opera and ballet performances"},{"id":"musrepov_theater","name_ru":"Театр имени А. Мусрепова","name_kk":"Ә. Мусрепов атындағы театр","name_en":"A. Musrepov Theater","hint_en":"Famous theater for drama and performances"},{"id":"astana_ballet_theater","name_ru":"Театр 'Астана Балет'","name_kk":"'Астана Балет' театры","name_en":"Astana Ballet Theater","hint_en":"A ballet theater showcasing classical and contemporary ballet"},{"id":"botanical_garden","name_ru":"Ботанический сад","name_kk":"Ботаникалық бақ","name_en":"Botanical Garden","hint_en":"Large green space with diverse plant species and educational exhibits"}]
```

> Note: keep `LANDMARKS_JSON` as a single line. You can later move it to SSM Parameter Store or S3 if it grows.

---

## Installation (local)

> Lambdas don’t start a web server locally; this is just to install deps and allow ad-hoc handler tests/imports.

```bash
# /ask
pip install -r functions/ai-guide/src/requirements.txt
export $(cat functions/ai-guide/env/.env.example | xargs)

# /recognize
pip install -r functions/vision-recognize/src/requirements.txt
export $(cat functions/vision-recognize/env/.env.example | xargs)
```

---

## Deployment (manual)

1. **Create/Update Lambda**

   * Zip `lambda_function.py` (and any extra modules if added) per function and upload, or paste code in console.
   * Memory: 1024 MB, Timeout: 15s (vision).
   * Add environment variables from `.env.example`.

2. **Amazon Bedrock**

   * Region **us-east-1** → enable inference profile `us.meta.llama3-2-90b-instruct-v1:0` (vision).
   * (Optional) enable `us.meta.llama3-2-70b-instruct-v1:0` for `/ask`.

3. **IAM role** for each Lambda

   * Allow `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`, and CloudWatch Logs.

4. **API Gateway (REST)**

   * Create resources `/ask` and `/recognize`.
   * Methods: `POST` → **Lambda proxy**; `OPTIONS` → Enable CORS.
   * **Binary media types** (for vision): `multipart/form-data`, `image/jpeg`, `image/png`.
   * Create **Usage plan** + **API key** and attach the stage (e.g., `prod`).

5. **Deploy API**

   * API Gateway → **Actions → Deploy API** → Stage `prod`.

---

## Endpoints

### `POST /ask`

**Headers**

```
Content-Type: application/json
x-api-key: <API_KEY>
```

**Request**

```json
{
  "text": "List night tours in Astana",
  "lang": "en",
  "persona": "formal",
  "prompt_type": "ask"
}
```

**Response**

```json
{
  "lang": "en",
  "persona": "formal",
  "prompt_type": "ask",
  "answer": "short structured text...",
  "latency_ms": 1234
}
```

**cURL**

```bash
curl -s -X POST 'https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/ask' \
  -H 'Content-Type: application/json' -H 'x-api-key: <API_KEY>' \
  -d '{"text":"Какие есть базары в Астане?","lang":"ru","persona":"humorous"}' | jq
```

---

### `POST /recognize`

**Headers**

```
x-api-key: <API_KEY>
Content-Type: multipart/form-data
```

**Form fields**

* `file` — image bytes (`image/jpeg` or `image/png`, ≤ 5 MB)
* `lang` — `ru | kk | en` (default `ru`)
* `persona` — `formal | humorous` (optional)

**Success (recognized)**

```json
{
  "lang": "kk",
  "persona": "formal",
  "latency_ms": 1310,
  "request_id": "4486faca-5f6c-4adf-8db6-4276330172d4",
  "detections": [
    { "name": "Ботаникалық бақ", "id": "botanical_garden", "confidence": 0.90 }
  ],
  "answer": "Супер! Бұл Ботаникалық бақ – квест тапсырмасы орындалды!"
}
```

**Success (low confidence < 0.7)**
Returns a detection with low confidence and a “try closer” message.

**Not recognized**

```json
{
  "lang": "ru",
  "persona": "formal",
  "latency_ms": 560,
  "request_id": "…",
  "detections": [],
  "answer": "Не удалось распознать достопримечательность."
}
```

**cURL**

```bash
curl -s "https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/recognize" \
  -H "x-api-key: <API_KEY>" \
  -F "file=@/absolute/path/photo.jpg;type=image/jpeg" \
  -F "lang=kk" -F "persona=formal" | jq
```

**Common HTTP codes**

* `200` — OK (including empty `detections`)
* `400` — wrong content type / missing `file`
* `413` — file too large
* `415` — unsupported media type (set `;type=image/jpeg|image/png` in multipart)
* `5xx` — internal error (see CloudWatch)

---

## Frontend example (Telegram Mini App)

```js
async function recognize(file, lang='ru', persona='formal') {
  const fd = new FormData();
  fd.append('file', file, file.name || 'photo.jpg');
  fd.append('lang', lang);
  fd.append('persona', persona);

  const res = await fetch('https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/recognize', {
    method: 'POST',
    headers: { 'x-api-key': '<API_KEY>' }, // in production proxy this on server
    body: fd
  });
  const data = await res.json();

  if (!data.detections?.length) return {status:'no_match', message:data.answer};
  const top = data.detections[0];
  if (top.confidence < 0.7) return {status:'low_conf', detection:top, message:data.answer};
  return {status:'ok', checkpoint:top.id, label:top.name, message:data.answer};
}
```

---

## Troubleshooting

* **415 Unsupported Media Type**
  Include explicit `;type=image/jpeg` or `;type=image/png` in the multipart `file=@...`.

* **200 OK but `detections` empty, `latency_ms` < \~200 ms**
  Model likely wasn’t invoked. Check ENV for `/recognize`:

  * `BEDROCK_REGION=us-east-1`
  * `MODEL_ID=us.meta.llama3-2-90b-instruct-v1:0`
  * `LANDMARKS_JSON` set (single line)

* **ValidationException: use inference profile**
  You’re using a direct model ID. Switch to `us.meta.llama3-2-90b-instruct-v1:0` and enable the profile.

* **AccessDeniedException / Model not allowed**
  Enable the profile in Bedrock (us-east-1) and make sure the Lambda role has `bedrock:InvokeModel`.

* **CloudWatch Logs**
  Group: `/aws/lambda/<function-name>`
  Look for `DBG ...` and `ERROR classify_with_llama:` lines.


## Security

* Don’t expose production `x-api-key` in client apps. Prefer a thin backend proxy or Cognito/JWT.
* Use Usage Plans (rate/burst/quota) and rotate keys.
* Validate file size (current limit: 5 MB) and MIME.

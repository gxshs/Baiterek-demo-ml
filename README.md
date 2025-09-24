# Baiterek-demo-ml

Serverless AI Guide for Astana built on AWS Bedrock.

## Endpoints

POST `/ask`

### Request
```json
{
  "text": "List night tours in Astana",
  "lang": "en",
  "persona": "formal",      
  "prompt_type": "ask"     
}
````

### Response

```json
{
  "lang": "en",
  "persona": "formal",
  "prompt_type": "ask",
  "answer": "short structured text...",
  "latency_ms": 1234
}
```

## Run locally

```bash
pip install -r functions/ai-guide/src/requirements.txt
export $(cat functions/ai-guide/env/.env.example | xargs)
python functions/ai-guide/src/lambda_function.py
```

## Deploy

Package and deploy Lambda to AWS, set env vars from `.env.example`.

## Test

```bash
curl -s -X POST 'https://n7ijgjzcka.execute-api.us-east-1.amazonaws.com/prod/ask' \
  -H 'Content-Type: application/json' -H 'x-api-key:  <your api key>' \
  -d '{"text":"Какие есть базары в Астане?","lang":"ru","persona":"humorous"}' | jq
```

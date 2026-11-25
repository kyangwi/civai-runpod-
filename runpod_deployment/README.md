# CivIA on Runpod Serverless

One queue-based endpoint runs Qwen in vLLM on the GPU and NLLB in Transformers on the CPU. French chat skips translation. Lingala and Swahili chat translate the incoming question to French, generate the reply in French, then translate it into the selected reply language.

The default chat checkpoint is `Qwen/Qwen3.5-9B`. Set `QWEN_MODEL=Qwen/Qwen3-14B` on the worker if 14B is required. This deployment uses the requested `facebook/nllb-200-1.3B` checkpoint for prototype evaluation. Meta releases it under CC BY-NC 4.0 and lists production and legal-domain translation outside its intended scope. Review the translation model choice before a public deployment. [NLLB model card](https://huggingface.co/facebook/nllb-200-1.3B)

## Build and deploy

1. Push these files to GitHub. Open **Actions → Build and deploy CivIA Runpod worker → Run workflow** with **deploy** unchecked. Changes to the deployment files on `main` or `master` also publish an image. The action summary shows the commit-tagged image at `ghcr.io/<owner>/<repo>-runpod:<commit-sha>`.
2. In Runpod, create a **queue-based Serverless** endpoint from that image. Leave the container start command blank. If the GHCR package is private, configure registry credentials with package read access in Runpod.
3. Start with one 48 GB NVIDIA GPU, zero active workers, and one maximum worker. Start with 80 GB of container disk if weights are stored on the worker. Optionally attach a 50 GB network volume for the Hugging Face cache, which defaults to `/runpod-volume/huggingface`. The first worker startup downloads both checkpoints. Qwen uses an 8,192-token context by default, including input and output.
4. Copy the endpoint ID for client calls. For GitHub-triggered updates, add it as the repository variable `RUNPOD_ENDPOINT_ID` and add a Runpod API key with `serverless:write` as the repository secret `RUNPOD_DEPLOY_API_KEY`. Run the workflow with **deploy** checked to publish the image and update the endpoint. Check the endpoint's active release and test it after updating. To roll back, set the endpoint image to the previous commit-tagged image. [Runpod endpoint update API](https://docs.runpod.io/api-reference-v2/serverless/update-a-serverless-endpoint)

## Test the endpoint

Set the following in PowerShell before calling the client. On a hosted backend, set the same names as environment variables.

```powershell
$env:RUNPOD_ENDPOINT_ID = "your-endpoint-id"
$env:RUNPOD_API_KEY = "your-inference-api-key"
python qwenendpoint.py "Bonjour, comment peux-tu m'aider ?" --language fr
python qwenendpoint.py "Mbote, okoki kosalisa ngai?" --language ln
python qwenendpoint.py "Habari, unaweza kunisaidia?" --language sw
```

The client submits to `/run` and polls `/status/{id}` so the smoke test can wait through a cold start. The worker returns a complete reply per job. [Runpod job operations](https://docs.runpod.io/serverless/endpoints/job-operations)

## Request format

Send an `input` object to Runpod. The client function `run_pipeline` wraps a payload in `input` for you.

```json
{
  "input": {
    "task": "chat",
    "text": "Mbote, okoki kosalisa ngai?",
    "language": "ln",
    "reply_language": "ln",
    "context_fr": "",
    "history_fr": [
      {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider ?"}
    ]
  }
}
```

Language values are `fr`, `ln` and `sw`. The reply language defaults to the question language. Chat returns `text`, `language`, `question_fr`, `answer_fr` and `finish_reason`. Store the French question and answer as history for subsequent calls, and keep citation URLs in the app's source metadata so they stay outside translation.

CivIA can use the same endpoint to translate a question before retrieval. Submit `{"task":"translate","text":"…","source_language":"ln","target_language":"fr"}`, retrieve sources in Django using the returned `text`, then call chat with the French question, `language: "fr"`, the desired `reply_language`, and `context_fr`. The current Django provider selection is independent of this endpoint. The included `qwenendpoint.py` exposes the backend call for the app to integrate.

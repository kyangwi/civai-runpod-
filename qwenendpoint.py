import os

from openai import OpenAI

endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
client = OpenAI(
    api_key=os.environ["RUNPOD_API_KEY"],
    base_url=f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[
        {
            "role": "system",
            "content": "Réponds en français simple, sur un ton naturel et chaleureux.",
        },
        {
            "role": "user",
            "content": "Bonjour, peux-tu m’aider à trouver la démarche que je dois suivre ?",
        },
    ],
    max_tokens=400,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response.choices[0].message.content)
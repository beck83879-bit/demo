from openai import OpenAI
client = OpenAI(api_key='sk-a6c965a9f7f54f80adbd811386af4834')

for m in client.models.list().data[:20]:
    print(m.id)

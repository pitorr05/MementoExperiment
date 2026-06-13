from openai import OpenAI


client = OpenAI(
    base_url="http://74.48.140.178:27231/v1",  
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="hiudev/gpt-oss-20b-VietMindAI-4bit",
    messages=[{"role": "user", "content": "Xin chào"}],
)

print(response.choices[0].message.content)
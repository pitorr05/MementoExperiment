from openai import OpenAI

# Khởi tạo client với API của bạn
client = OpenAI(
    base_url="https://foot-career-frederick-emotions.trycloudflare.com/v1",
    api_key="EMPTY"  # Không cần key thật
)

print("=" * 50)
print("TESTING QWEN API WITH OPENAI CLIENT")
print("=" * 50)

# Test 1: List models
print("\n📋 Test 1: List available models")
try:
    models = client.models.list()
    print(f"✅ Models: {[m.id for m in models.data]}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Chat completion
print("\n💬 Test 2: Chat completion")
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {"role": "user", "content": "Trần Quốc Toản là ai trong lịch sử Việt Nam viết về lịch sử của ông ấy "}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    print(f"✅ Response received!")
    print(f"💬 Answer: {response.choices[0].message.content}")
    print(f"📊 Usage: {response.usage}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Multiple turns
print("\n🔄 Test 3: Multi-turn conversation")
try:
    messages = [
        {"role": "user", "content": "My name is John."},
        {"role": "assistant", "content": "Nice to meet you, John!"},
        {"role": "user", "content": "What's my name?"}
    ]
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=messages,
        max_tokens=50
    )
    
    print(f"✅ Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("✅ API TEST COMPLETED")
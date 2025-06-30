import requests

response = requests.post("http://127.0.0.1:8000/api/chat", json={
    "message": "tell me about sprout",
    "session_id": "test_1234w"
}, stream=True)

final_text = ""
for line in response.iter_lines():
    if line:
        if b"content" in line:
            try:
                content = eval(line.decode().split("data: ")[1])
                final_text += content.get("content", "")
            except:
                pass
print(final_text)

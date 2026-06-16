import requests

response = requests.post(
    "https://SUA-URL.onrender.com/analyze",
    json={
        "filename": "Main.java",
        "content": "System.out.println('teste');"
    }
)

print(response.json())
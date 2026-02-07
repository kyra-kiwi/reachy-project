import ollama

response = ollama.chat(model='phi3:mini', messages=[
    {'role': 'user', 'content': 'Say hello in a friendly way'}
])
print(response['message']['content'])
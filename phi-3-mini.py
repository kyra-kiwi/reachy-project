import ollama

def llama(message):
    print(f"LLAMA: {message}")
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': message}
    ])
    return response['message']['content']

def phi3(message):
    print(f"PHI3: {message}")
    response = ollama.chat(model='phi3:mini', messages=[
        {'role': 'user', 'content': message}
    ])
    return response['message']['content']


if __name__ == "__main__":
    print(llama("Hello, how are you?"))
    print("--------------------------------")
    print(phi3("Hello, how are you?"))
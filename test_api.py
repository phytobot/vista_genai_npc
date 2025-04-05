

import requests

print("Welcome to the Fantasy NPC Chat!")
print("Type 'exit' or 'quit' to end the session.\n")

while True:
    message = input("You: ")
    if message.lower() in ['exit', 'quit']:
        print("Exiting...")
        break

    try:
        response = requests.post(
            "http://127.0.0.1:5000/analyze",
            json={"message": message}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"NPC: {data['npc_reply']}")
        else:
            print("Error:", response.json().get("error", "Unknown error"))
    except Exception as e:
        print("Exception occurred:", str(e))

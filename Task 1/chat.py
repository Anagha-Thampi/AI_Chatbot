import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

msg=[{"role":"system","content":"be rude"}]

while(True):
    imsg=input("You:")
    msg.append({"role":"user","content":imsg})

    if imsg.lower()=="exit":
        break

    chat_completion = client.chat.completions.create(
    messages=msg,
    model="llama-3.1-8b-instant",
    temperature=0.8,
    #stream=True
    )

    print(chat_completion.choices[0].message.content)

    msg.append({"role":"assistant","content":chat_completion.choices[0].message.content})

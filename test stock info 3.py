import socketio
import json
import asyncio

endpoint = "https://ws-api.iextrading.com/1.0/tops"
symbols = [
    "AAPL",
    "MSFT",
    "SNAP"
]

@client.on("connect", namespace=namespace)
async def on_connect():
    for symbol in symbols:
        await client.emit("subscribe", symbol, namespace=namespace)
        print(f"Subscribed to '{symbol}'")

@client.on("message", namespace=namespace)
def on_message(message):
    data = json.loads(message)
    print(data)

loop = asyncio.get_event_loop()
loop.create_task(task)
loop.run_forever()

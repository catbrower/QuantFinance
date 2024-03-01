import ssl
import time
import asyncio
import websockets
import pandas as pd

t0 = time.time()
# Set this to a perposterous value to get tons of data
max_results = 5e12

async def main():
    # uri = "ws://192.168.1.151:8080/aggregates"
    uri = "ws://localhost:8080/aggregates"

    ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
    async with ws.connect(uri) as websocket:
        message = "{\"tickers\": [\"QQQ\", \"SPY\"], \"from\": 0, \"to\": 1000000000}"
        print(f"Sending message: {message}")
        await websocket.send(message)

        data = []
        headers = ''

        count = 0

        while count < max_results:
            response = await websocket.recv()
            if response == "END":
                break

            response = response.split(',')

            if count == 0:
                headers = response
            else:
                data.append(response)
            count += 1

        df = pd.DataFrame(data, columns=headers)
        print(df)

        print(f'Retreived data in {(time.time() - t0)}s')

asyncio.run(main())
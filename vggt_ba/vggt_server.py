import pickle
from threading import Thread

from fastapi import FastAPI, Request, Response
import uvicorn

from vggt_ba import CudaInference


print("Loading model")
model = CudaInference()
print("Finished loading model")

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    imgs = pickle.loads(body)
    print(imgs)
    predictions = model.run(imgs)
    response = Response(
        content = pickle.dumps(predictions),
        media_type = "application/octet-stream"
    )
    return response


def start_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == '__main__':
    Thread(target=start_api, daemon=True).start()
    input("Press Enter to exit...\n")

import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from src.serve_model import *

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title="Tensorflow FastAPI Start Pack", description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/audio")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("mp3")
    if not extension:
        return "Audio must be mp3"


    audio = read_audio_file(await file.read())
    
    prediction = predict(audio, file.filename)

    # print(prediction)
    return prediction

data = {
    "song": "8.mp3",
    "middle_level": {
        "melody": float(5.9008684),
        "articulation": float(3.958108),
        "rhythm_complexity": float(4.6986804),
        "rhythm_stability": float(5.2627783),
        "dissonance": float(3.6181989),
        "atonality": float(7.2277913),
        "mode": float(5.541093)
    },
    "emotion": {
        "valence": float(4.809655),
        "energy": float(4.3131323),
        "tension": float(3.5942125),
        "anger": float(1.0534579),
        "fear": float(1.2822785),
        "happy": float(3.3106818),
        "sad": float(1.6806943),
        "tender": float(2.8928287)
    }
}

@app.get("/your_endpoint")
async def your_endpoint():
    return data

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.get("/users")
# async def users():
#     users = [
#         {
#             "name": "Mars Kule",
#             "age": 25,
#             "city": "Lagos, Nigeria"
#         },

#         {
#             "name": "Mercury Lume",
#             "age": 23,
#             "city": "Abuja, Nigeria"
#         },

#          {
#             "name": "Jupiter Dume",
#             "age": 30,
#             "city": "Kaduna, Nigeria"
#         }
#     ]

#     return users
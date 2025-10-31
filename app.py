from fastapi import FastAPI, File, HTTPException, UploadFile
from yolov_model import predict  # This import stays the same
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DentAI YOLOv5 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "DentAI API is running!"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image with a valid content-type (e.g., image/jpeg).")

        image_bytes = await file.read()
        
        try:
            prediction_result = predict(image_bytes)
            return prediction_result
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
# main.py
from fastapi import FastAPI
from app.api.predict import router
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI(title="CBC Disease Risk Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local development
    allow_methods=["*"],
    allow_headers=["*"],
)
# Register prediction route
app.include_router(router)

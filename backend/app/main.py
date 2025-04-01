from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import updates

app = FastAPI(title="arXiv Updates API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(updates.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to arXiv Updates API"}
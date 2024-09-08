from fastapi import APIRouter
from fastapi import Body
from model.model import save_model
from model.model import load_model
import torch
router = APIRouter()
model = load_model()


@router.post("/recognize")
async def recognize(data: list = Body(...)):
    data = torch.tensor(data, dtype=torch.float32).view(1, 28, 28).to("cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0)
    print(model(data).squeeze(0).tolist())
    return model(data).squeeze(0).tolist()

@router.get("/retrain")
async def save():
    save_model()
    return {"output": "Model has been updated"}

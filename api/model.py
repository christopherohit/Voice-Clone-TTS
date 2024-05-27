from pydantic import BaseModel
from fastapi import File, UploadFile
from typing import Optional, final, Any

class InputData(BaseModel):
    text: str
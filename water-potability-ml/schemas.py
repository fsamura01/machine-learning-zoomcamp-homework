from pydantic import BaseModel, Field


class WaterSample(BaseModel):
    ph: float = Field(..., ge=0, le=14, description="pH level")
    Hardness: float = Field(..., ge=0, description="Hardness in mg/L")
    Solids: float = Field(..., ge=0, description="Total dissolved solids")
    Chloramines: float = Field(..., ge=0, description="Chloramines in ppm")
    Sulfate: float = Field(..., ge=0, description="Sulfate in mg/L")
    Conductivity: float = Field(..., ge=0, description="Electrical conductivity")
    Organic_carbon: float = Field(..., ge=0, description="Organic carbon in ppm")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalomethanes in µg/L")
    Turbidity: float = Field(..., ge=0, description="Turbidity in NTU")


class PredictionResponse(BaseModel):
    potable: bool
    confidence: float
    recommendation: str

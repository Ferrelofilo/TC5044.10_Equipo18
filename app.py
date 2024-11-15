from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipelines.models import MultiOutCnnHandler
import pandas as pd
import torch
import yaml
import os
from pipelines.transformers import get_flare_transformer


app = FastAPI()

model_type = "linear_cnn"
model_path = os.path.join("data/models", f"{model_type}_model.pth")


cnn_handler = MultiOutCnnHandler(cnn_type=model_type)
cnn_handler.load_model(model_path)


class PredictionRequest(BaseModel):
    modified_zurich_class: str = "H"
    largest_spot_size: str = "A"
    spot_distribution: str = "X"
    activity: int = 1
    evolution: int = 3
    previous_24_hour_flare_activity: int = 1
    historically_complex: int = 1
    became_complex_on_this_pass: int = 1
    area: int = 1

    def to_dataframe(self):

        data = {
            "modified Zurich class": [self.modified_zurich_class],
            "largest spot size": [self.largest_spot_size],
            "spot distribution": [self.spot_distribution],
            "activity": [self.activity],
            "evolution": [self.evolution],
            "previous_24_hour_flare_activity": [self.previous_24_hour_flare_activity],
            "historically_complex": [self.historically_complex],
            "became_complex_on_this_pass": [self.became_complex_on_this_pass],
            "area": [self.area],
        }
        return pd.DataFrame(data)


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:

        input_df = request.to_dataframe()

        pipeline = get_flare_transformer()
        encoded = pipeline.fit_transform(input_df)
        processed_input = pd.DataFrame(
            encoded, index=input_df.index, columns=input_df.columns
        )

        input_tensor = torch.tensor(processed_input.values, dtype=torch.float32)

        cnn_handler.model.eval()
        with torch.no_grad():
            outputs = cnn_handler.model(input_tensor)

        predictions = [int(output[0][0]) for output in outputs]

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de predicción de CNN"}

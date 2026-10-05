import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException , UploadFile, File
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# Input schema for the API request
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(ge=0, description="Median house age in block group")
    AveRooms: float = Field(gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per household")
    Population: float = Field(gt=0, description="Total population in block group")
    AveOccup: float = Field(gt=0, description="Average number of household members")
    Latitude: float = Field(ge=32, le=42, description="Latitude of block group")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude of block group")




# jo price hoga house Medinc me wo divide by 100000 hoga taki humara price ka format sahi ho jaye . 
# houseage ka matlab h ki humara model kya predict kar raha h , ye house ka price hoga jo humara target variable hoga . 
# avgrooms ka matlab h ki humara model kya predict kar raha h , ye house ka price hoga jo humara target variable hoga .




@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API",
        "status": "running",
        "endpoint": "Send POST request to /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000/-",
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([house.model_dump()])

        # Ensure features order matches the trained model
        input_data = input_data[features]

        # Fix 1 & 2: Fixed variable names (input_data & predicted)
        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousand",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}",
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )



# @app.post("/predict_file"): FastAPI me ek naya POST API route banata hai jiska path /predict_file hota hai. Iska use server ko data (file) bhejne ke liye kiya jata hai.

# async def predict_file(...): Fast/Asynchronous function banata hai jo file upload hote waqt server ko freeze ya block hone se bachata hai.

# file: UploadFile: FastAPI ka special type hai jo uploaded file ko disk/memory me efficiently handle karta hai (badi CSV/Excel files ke liye best).

# File(...): FastAPI ko batata hai ki yeh field Required (zaroori) hai; bina file attach kiye API request execute nahi hogi.

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a CSV file only."
        )

    # FIX 1: Indentation fix kiya (pehle ye raise HTTPException ke andar hidden tha)
    contents = await file.read()

    # FIX 2: pd.DataFrame ko badal kar pd.read_csv kiya kyunki CSV buffer read_csv se load hota hai
    df = pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    missing_columns = [
        col for col in required_columns 
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"these columns are Missing from your file{missing_columns}"
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The Uploaded file has no data rows ."
        )


    try:
        predictions = model.predict(df[required_columns])  

        # FIX 3: Pehle predictions ko column me assign kiya, fir $ format apply kiya
        df["predicted_column_usd"] = predictions * 100000
        df["predicted_column_usd"] = df["predicted_column_usd"].apply(lambda x: f"${x:,.0f}")

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


""""
# ---------------------------------------------------------
# FASTAPI CSV PREDICTION & STREAMING FILE RESPONSE NOTES
# ---------------------------------------------------------

# 1. Error handling ke liye try block start hota hai
try:
    # 2. DataFrame ke specific columns par model prediction run karta hai
    predictions = model.predict(df[required_columns])  

    # 3. Prediction values ko currency format ($350,000) me convert karta hai
    df["predicted_column_usd"] = df["predicted_column_usd"].apply(lambda x: f"${x:,.0f}")

    # 4. DataFrame ko CSV string me badalta hai (index=False se extra row numbers nahi aate)
    output = df.to_csv(index=False)

    # 5. Browser/Client ko file stream ke roop me download response bheja jata hai
    return StreamingResponse(
        # 6. String data ko file-like object me convert karta hai stream ke liye
        io.StringIO(output),
        
        # 7. Response type set karta hai ki yeh CSV data hai
        media_type="text/csv",
        
        # 8. Browser ko batata hai ki response ko 'predictions.csv' naam se direct download karna hai
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv"
        }
    )

"""
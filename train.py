from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib  



print("Loading dataset")
data = fetch_california_housing()

 # x ka matlab ye h ki humara input data jo hum dengy . x humara table of features hoga jo  ye dekhega ki humara model  kya predict karega model ko train karne ke liye 
X = pd.DataFrame(data.data, columns=data.feature_names)  




#ye hoga ki model kya try kr raha h predict krne ka , ye house ka price hoga jo humara target variable hoga
Y = data.target 


print(f"total records: {X.shape[0]}")



# train_test_split ka matlab h ki hum apne data ko 2 parts me divide karenge , ek part hoga training data aur dusra hoga testing data .
# test_size=0.2 ka matlab h ki hum 20% data ko testing ke liye use karenge aur 80% data ko training ke liye use karenge . 
# random_state=42 ka matlab h ki hum apne data ko randomly split karenge taki har bar same result mile

# x_train-  features of training data
# x_test-  features of testing data 
# y_train-  target values of training data
# y_test-  target values of testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# training a model
# model name h RandomForestRegressor , ye ek ensemble learning method h jo decision trees ka use karke regression problems ko solve karta h .
# n_estimators=100 ka matlab h ki hum 100 decision trees ka use karenge model ko train karne ke liye .
# random_state=42 ka matlab h ki hum apne model ko randomly initialize karenge taki har bar same result mile
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42

)
model.fit(X_train, y_train)

# is line ka matlab h ki hum apne trained model ko save karenge taki hum future me use kar sake .
Y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, Y_pred)
r2 = r2_score(y_test, Y_pred)


 
print(f"average error: ${mae * 100000:,.0f}")
# 49,000/- 49122.124321

joblib.dump(model, "house_model.joblib")
joblib.dump(list(X.columns), "house_features.joblib")







# cd "House_prediction_api"
# .\venv\Scripts\Activate.ps1
# python explore.py
# python train.py
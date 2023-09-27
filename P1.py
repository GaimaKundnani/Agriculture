import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
#Load Datasets 
df=pd.read_csv(r"/Users/Garima/Desktop/Python/Agriculture/crop_production.csv")#State Wise Major Crop Production
df1=pd.read_csv(r"/Users/Garima/Desktop/Python/Agriculture/cpdata.csv")#Crop Dependence
df2=pd.read_csv(r"/Users/Garima/Desktop/Python/Agriculture/cropph.csv")#Ideal value of pH for various crop
df3=pd.read_csv(r"/Users/Garima/Desktop/Python/Agriculture/state_wise_crop_production.csv")#State Wise Cost of Production
df4=pd.read_csv(r"/Users/Garima/Desktop/Python/Agriculture/cropproductiononvariousfactors.csv")#Actual Crop Yield based on Various Factors
df5=pd.concat([df,df1,df2,df3,df4],axis=0, ignore_index=False)
#First few Rows
print(df5.head())
#Description and Info
print("Description:")
print(df5.describe())
print("Infomation:")
print(df5.info())
#Null Values
print("Missing values:")
print(df5.isnull())
df7= df5.fillna(0,inplace=True)
print("Sort Values:")
print(df5.corr()['Crop_Year'].sort_values(ascending=False))
#The best crop to grow in each State
#Label Encoder
df6 = df5[['State', 'Crop','Yield (Quintal/ Hectare) ']]
State_le=LabelEncoder()
Crop_le=LabelEncoder()
df6['State'] = df6['State'].astype(str)
df6['Crop'] = df6['Crop'].astype(str)
df6['State']=State_le.fit_transform(df6['State'])
df6['Crop']=Crop_le.fit_transform(df6['Crop'])
print(df6)
#Factors affect the respective crops the most
Crop_le=LabelEncoder()
Cropconversion_le=LabelEncoder()
df4['Crop']=Crop_le.fit_transform(df4['Crop'])
df4['Cropconversion']=Cropconversion_le.fit_transform(df4['Cropconversion'])
print(df4.head())
#Bar Plot
#plt.bar(df6['State'], df6['Crop'])
#plt.xlabel('State')  
#plt.ylabel('Crop')  
#plt.show()
#Split Dataset
a=df6[['State']]
b=df6[['Yield (Quintal/ Hectare) ']]
X_train,X_test,y_train,y_test=train_test_split(a,b,test_size=0.3,random_state=20)
#Experiment with various models for this project
model=RandomForestRegressor(n_estimators=20)
model.fit(X_train,y_train)
print("Accuracy of RandomForestClassifier:",model.score(X_test,y_test)*100)
#Create an interface(using statements), where the user inputs their Geographic
#Location(District), and based on that your model should generate output
#prescribing them the best crop to grow in that region

#c=input("Enter your State:")
#c_numeric = char (c)
#c_reshaped = [[c_numeric]]
#d=model.predict(c_reshaped)
#print(d)
#Recommend Crop
def recommend_crop():
    user_district = input("Enter your Geographic Location (District): ")
    user_encoded_district = State_le.transform([user_district])
    predicted_yield = model.predict([user_encoded_district])[0]
    predicted_encoded_crop = model.predict([user_encoded_district])[0]
    if predicted_encoded_crop in Crop_le.classes_:
        predicted_crop = Crop_le.inverse_transform([predicted_encoded_crop])[0]
        print(f"Based on your location, it is recommended to grow: ",{predicted_crop})
    else:
        print("The model predicts an unknown crop for this location.")
    
    print(f"Predicted Yield for this crop: {predicted_yield} Quintal/Hectare")
print(recommend_crop())
#Handeling of Missing Values
def re_crop(user_encoded_district):
    # Collect additional metrics from the user
    rainfall = input("Enter the average annual rainfall (in mm), or leave blank: ")
    temperature = input("Enter the average annual temperature (in degrees Celsius), or leave blank: ")
    soil_ph = input("Enter the soil pH value, or leave blank: ")
    
    # Check if the user provided input for each metric, if not, use None
    rainfall = float(rainfall) if rainfall else None
    temperature = float(temperature) if temperature else None
    soil_ph = float(soil_ph) if soil_ph else None
    
    # Create a dictionary with the user's input
    user_data = {
        'State': user_encoded_district[0],  # Assuming you encoded the state as 'State' during training
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Soil_pH': soil_ph
    }
    
    # Convert the dictionary to a DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    user_df = user_df.fillna(0)

    # Make a prediction based on the user's input
    predicted_yield = model.predict(user_df)[0]
    
    # You may need to decode the crop label for a more user-friendly output using Crop_le
    predicted_crop_encoded = model.predict(user_df)[0]
    predicted_crop = Crop_le.inverse_transform([predicted_crop_encoded])[0]
    
    # Calculate expected profit (you need to define your profit calculation based on your data)
    expected_profit = calculate_profit(predicted_crop, predicted_yield)
    
    # Print the recommended crop, predicted yield, and expected profit
    print(f"Based on your location and input metrics, it is recommended to grow: {predicted_crop}")
    print(f"Predicted Yield for this crop: {predicted_yield} Quintal/Hectare")
    # print(f"Expected Profit for this crop: ${expected_profit}")

# Call the function with the user_encoded_district as an argument
user_district = input("Enter your Geographic Location (District): ")
user_encoded_district = State_le.transform([user_district])
print(re_crop(user_encoded_district))

#final output should be in the form of a dataframe where columns should include the current crop the user is growing, the suggested crop, current profit from the present crop, expected profit from the predicted crop
def reco_crop(user_encoded_district):
    # Collect user input for the current crop and additional metrics
    current_crop = input("Enter the current crop you are growing: ")
    rainfall_input = input("Enter the average annual rainfall (in mm), or leave blank: ")
    temperature_input = input("Enter the average annual temperature (in degrees Celsius), or leave blank: ")
    soil_ph_input = input("Enter the soil pH value, or leave blank: ")
    
    # Convert user inputs to floats if provided, or set them to None if not provided
    rainfall = float(rainfall_input) if rainfall_input else None
    temperature = float(temperature_input) if temperature_input else None
    soil_ph = float(soil_ph_input) if soil_ph_input else None
    
    # Create a dictionary with the user's input, including missing values as None
    user_data = {
        'State': user_encoded_district[0],  # Assuming you encoded the state as 'State' during training
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Soil_pH': soil_ph
    }
    
    # Convert the dictionary to a DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    
    # Handle missing values by filling them with default values (e.g., mean or median of training data)
    user_df.fillna(train_data_mean_or_median, inplace=True)  # Replace train_data_mean_or_median with your computed values
    
    # Make a prediction based on the user's input
    predicted_yield = model.predict(user_df)[0]
    
    # You may need to decode the crop label for a more user-friendly output using Crop_le
    predicted_crop_encoded = model.predict(user_df)[0]
    predicted_crop = Crop_le.inverse_transform([predicted_crop_encoded])[0]
    
    # Calculate expected profit (you need to define your profit calculation based on your data)
    expected_profit = calculate_profit(predicted_crop, predicted_yield)
    
    # Assuming you have a function to calculate current profit from the current crop
    current_profit = calculate_current_profit(current_crop)  # Define this function
    
    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Current Crop': [current_crop],
        'Suggested Crop': [predicted_crop],
        'Current Profit': [current_profit],
        'Expected Profit': [expected_profit]
    })
    
    # Print the recommended crop, predicted yield, and expected profit
    print(results_df)

# Call the function with the user_encoded_district as an argument
user_district = input("Enter your Geographic Location (District): ")
user_encoded_district = State_le.transform([user_district])
re_crop(user_encoded_district)

#the prescriptive actions for the soil improvement if any, and the cost associated with the change
def re_crop(user_encoded_district):
    # Collect user input for the current crop, soil health, and additional metrics
    current_crop = input("Enter the current crop you are growing: ")
    rainfall_input = input("Enter the average annual rainfall (in mm), or leave blank: ")
    temperature_input = input("Enter the average annual temperature (in degrees Celsius), or leave blank: ")
    soil_ph_input = input("Enter the soil pH value, or leave blank: ")
    soil_health_info = input("Enter soil health information (if available): ")
    
    # Convert user inputs to floats if provided, or set them to None if not provided
    rainfall = float(rainfall_input) if rainfall_input else None
    temperature = float(temperature_input) if temperature_input else None
    soil_ph = float(soil_ph_input) if soil_ph_input else None
    
    # Create a dictionary with the user's input, including missing values as None
    user_data = {
        'State': user_encoded_district[0],  # Assuming you encoded the state as 'State' during training
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Soil_pH': soil_ph
    }
    
    # Convert the dictionary to a DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    
    # Handle missing values by filling them with default values (e.g., mean or median of training data)
    user_df.fillna(train_data_mean_or_median, inplace=True)  # Replace train_data_mean_or_median with your computed values
    
    # Make a prediction based on the user's input
    predicted_yield = model.predict(user_df)[0]
    
    # You may need to decode the crop label for a more user-friendly output using Crop_le
    predicted_crop_encoded = model.predict(user_df)[0]
    predicted_crop = Crop_le.inverse_transform([predicted_crop_encoded])[0]
    
    # Calculate expected profit (you need to define your profit calculation based on your data)
    expected_profit = calculate_profit(predicted_crop, predicted_yield)
    
    # Calculate prescriptive actions for soil improvement and associated cost (you need to define this)
    soil_improvement_actions, soil_improvement_cost = suggest_soil_improvement(current_crop, soil_health_info)
    
    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Current Crop': [current_crop],
        'Suggested Crop': [predicted_crop],
        'Current Profit': [current_profit],
        'Expected Profit': [expected_profit],
        'Soil Improvement Actions': [soil_improvement_actions],
        'Soil Improvement Cost': [soil_improvement_cost]
    })
    
    # Print the recommended crop, predicted yield, expected profit, and soil improvement suggestions
    print(results_df)

# Call the function with the user_encoded_district as an argument
user_district = input("Enter your Geographic Location (District): ")
user_encoded_district = State_le.transform([user_district])
re_crop(user_encoded_district)

#

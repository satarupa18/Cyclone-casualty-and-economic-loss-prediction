import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import tkinter as tk
from tkinter import ttk

# Read CSV and store it in MongoDB
csv_file_path = r'C:\Users\Rishav Sengupta\project(srijit clg)\Final Prediction analysis\Final_Data1.csv'
data = pd.read_csv(csv_file_path)

client = MongoClient('mongodb://localhost:27017/')
db = client['Cyclone_Final']
collection = db['DATA1']

# Store data in MongoDB
collection.delete_many({})  # Clear existing data
collection.insert_many(data.to_dict('records'))

# Load the data from MongoDB into a DataFrame
data = pd.DataFrame(list(collection.find()))

# Label encoding for 'Effected_Regions' and 'Effected_District'
label_encoder_region = LabelEncoder()
label_encoder_district = LabelEncoder()
data['effect_region_encoded'] = label_encoder_region.fit_transform(data['Effected_Regions'])
data['effect_district_encoded'] = label_encoder_district.fit_transform(data['Effected_District'])

# Define independent variables for CASUALITIES prediction
independent_vars_casualties = ['WIND_SPEED', 'Total_Population', 'Males_Population', 'Females_Population', 'Households']

# Split the data into training and testing sets for CASUALITIES
X_train_casualties, X_test_casualties, y_train_casualties, y_test_casualties = train_test_split(
    data[independent_vars_casualties], data['CASUALITIES'], test_size=0.2, random_state=42
)

# Create and fit the Random Forest regressor for CASUALITIES prediction
rf_regressor_casualties = RandomForestRegressor(n_estimators=40, max_depth=7, min_samples_split=2,
                                                min_samples_leaf=1, max_features='sqrt', random_state=42)
rf_regressor_casualties.fit(X_train_casualties, y_train_casualties)

# Define independent variables for Economic_Loss prediction
independent_vars_economic_loss = ['WIND_SPEED', 'GDP_VALUE', 'Inhabited_villages', 'Towns']

# Split the data into training and testing sets for Economic_Loss
X_train_economic_loss, X_test_economic_loss, y_train_economic_loss, y_test_economic_loss = train_test_split(
    data[independent_vars_economic_loss], data['ECONOMIC_LOSS'], test_size=0.2, random_state=42
)

# Create and fit the Random Forest regressor for Economic_Loss prediction
rf_regressor_economic_loss = RandomForestRegressor(n_estimators=20, max_depth=8, min_samples_split=3,
                                                   min_samples_leaf=1, max_features='sqrt', random_state=42)
rf_regressor_economic_loss.fit(X_train_economic_loss, y_train_economic_loss)

# Define independent variables for effect_region prediction
independent_vars_effect_region = ['WIND_SPEED', 'state_encoded']

# Split the data into training and testing sets for effect_region
X_train_effect_region, X_test_effect_region, y_train_effect_region, y_test_effect_region = train_test_split(
    data[independent_vars_effect_region], data['effect_region_encoded'], test_size=0.2, random_state=42
)

# Create and fit the Random Forest regressor for effect_region prediction
rf_regressor_effect_region = RandomForestRegressor(n_estimators=1000, max_depth=9, min_samples_split=2,
                                                    min_samples_leaf=1, max_features='sqrt', random_state=42)
rf_regressor_effect_region.fit(X_train_effect_region, y_train_effect_region)

# Define independent variables for effect_district prediction
independent_vars_effect_district = ['WIND_SPEED', 'state_encoded']

# Split the data into training and testing sets for effect_district
X_train_effect_district, X_test_effect_district, y_train_effect_district, y_test_effect_district = train_test_split(
    data[independent_vars_effect_district], data['effect_district_encoded'], test_size=0.2, random_state=42
)

# Create and fit the Random Forest regressor for effect_district prediction
rf_regressor_effect_district = RandomForestRegressor(n_estimators=1000, max_depth=9, min_samples_split=2,
                                                     min_samples_leaf=1, max_features='sqrt', random_state=42)
rf_regressor_effect_district.fit(X_train_effect_district, y_train_effect_district)

# Predict function
def predict():
    wind_speed_val = float(entry_wind_speed.get())
    state_val = dropdown_state.get().replace(" ", "_")
    
    # Prepare input for prediction
    input_data = pd.DataFrame({
        'WIND_SPEED': [wind_speed_val],
        'Total_Population': [data[data['STATE'] == state_val]['Total_Population'].iloc[0]],
        'Males_Population': [data[data['STATE'] == state_val]['Males_Population'].iloc[0]],
        'Females_Population': [data[data['STATE'] == state_val]['Females_Population'].iloc[0]],
        'Households': [data[data['STATE'] == state_val]['Households'].iloc[0]],
        'GDP_VALUE': [data[data['STATE'] == state_val]['GDP_VALUE'].iloc[0]],
        'Inhabited_villages': [data[data['STATE'] == state_val]['Inhabited_villages'].iloc[0]],
        'Towns': [data[data['STATE'] == state_val]['Towns'].iloc[0]],
        'state_encoded': [data[data['STATE'] == state_val]['state_encoded'].iloc[0]]
    })
    
    # Predict casualties
    casualties_prediction = rf_regressor_casualties.predict(input_data[independent_vars_casualties])
    label_casualties.config(text=f"Casualties Prediction: {casualties_prediction[0]:.2f}")
    
    # Predict economic loss
    economic_loss_prediction = rf_regressor_economic_loss.predict(input_data[independent_vars_economic_loss])
    label_economic_loss.config(text=f"Economic Loss Prediction: {economic_loss_prediction[0]:.2f}")
    
    # Predict affected region
    effect_region_prediction = rf_regressor_effect_region.predict(input_data[independent_vars_effect_region])
    effect_region_prediction_label = label_encoder_region.inverse_transform([int(effect_region_prediction[0])])[0]
    label_effect_region.config(text=f"Effect Region Prediction: {effect_region_prediction_label}")
    
    # Predict affected district
    effect_district_prediction = rf_regressor_effect_district.predict(input_data[independent_vars_effect_district])
    effect_district_prediction_label = label_encoder_district.inverse_transform([int(effect_district_prediction[0])])[0]
    label_effect_district.config(text=f"Effect District Prediction: {effect_district_prediction_label}")

# Create GUI
root = tk.Tk()
root.title("Disaster Prediction")
root.geometry("500x500")

# Styling options
style = ttk.Style()
style.configure("TLabel", padding=6)
style.configure("TEntry", padding=6)
style.configure("TButton", padding=6)

# Labels and entry for wind speed
frame_wind_speed = ttk.Frame(root, padding="10")
frame_wind_speed.pack(fill=tk.X)

label_wind_speed = ttk.Label(frame_wind_speed, text="Enter Wind Speed:")
label_wind_speed.pack(side=tk.LEFT)

entry_wind_speed = ttk.Entry(frame_wind_speed)
entry_wind_speed.pack(side=tk.LEFT)

# Dropdown for state selection
frame_state = ttk.Frame(root, padding="10")
frame_state.pack(fill=tk.X)

label_state = ttk.Label(frame_state, text="Select State:")
label_state.pack(side=tk.LEFT)

state_options = data['STATE'].unique()
state_options = [state.replace("_", " ") for state in state_options]
dropdown_state = ttk.Combobox(frame_state, values=state_options)
dropdown_state.pack(side=tk.LEFT)

# Predict button
frame_button = ttk.Frame(root, padding="10")
frame_button.pack()

button_predict = ttk.Button(frame_button, text="Predict", command=predict)
button_predict.pack()

# Labels for displaying predictions
frame_results = ttk.Frame(root, padding="10")
frame_results.pack(fill=tk.X)

label_casualties = ttk.Label(frame_results, text="")
label_casualties.pack()

label_economic_loss = ttk.Label(frame_results, text="")
label_economic_loss.pack()

label_effect_region = ttk.Label(frame_results, text="")
label_effect_region.pack()

label_effect_district = ttk.Label(frame_results, text="")
label_effect_district.pack()

root.mainloop()

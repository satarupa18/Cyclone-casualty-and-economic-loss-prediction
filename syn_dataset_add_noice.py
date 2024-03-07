import pandas as pd
import numpy as np


def modify_duplicate_rows(windspeed, casualties):
    # Generate a random value for modification
    random_value_wind_speed = int(np.random.uniform(low=1, high=10))  #  adjust the range 
    random_value_casulaties = int(np.random.uniform(low=30000, high=40000))
    # Randomly choose whether to add or subtract
    operation = np.random.choice(['add', 'subtract'])
    dice_roll=int(np.random.uniform(low=1, high=6))
    multiply_wind_speed=1
    multiply_casualties=1
    if dice_roll==5:
        multiply_chance=np.random.choice([1,-1])
        if multiply_chance == 1:
            multiply_wind_speed= -1
        else:
            multiply_casualties= -1
    random_value_wind_speed=random_value_wind_speed * multiply_wind_speed
    random_value_casulaties=random_value_casulaties * multiply_casualties

    if operation == 'add':
        windspeed += random_value_wind_speed
        casualties += random_value_casulaties
    else:
        windspeed -= random_value_wind_speed
        casualties -= random_value_casulaties
    if windspeed < 0:
        windspeed=48
    if casualties <0:
        casualties=casualties * (-1)

    return casualties,windspeed


#load the synthethic dataste
synthetic_data = pd.read_csv('Updated_Synthetic_Data.csv')

grouped = synthetic_data.groupby(['WIND_SPEED_X', 'STATE'])
# Apply noise modification function for each group
for group_keys, group_df in grouped:
    # Get the windspeed and casualties values for the group
    windspeed_group = group_df['WIND_SPEED_X']
    casualties_group = group_df['ECONOMIC_LOSS']
    
    # Apply noise modification function to each row in the group
    for index, row in group_df.iterrows():
        windspeed = row['WIND_SPEED_X']
        casualties = row['ECONOMIC_LOSS']
        
        # Apply noise modification function
        modified_casualties, modified_windspeed = modify_duplicate_rows(windspeed, casualties)
        
        # Update the original dataframe with modified values
        synthetic_data.at[index, 'WIND_SPEED_X'] = modified_windspeed
        synthetic_data.at[index, 'ECONOMIC_LOSS'] = modified_casualties

# Save the updated dataset to a new CSV file
synthetic_data.to_csv('Updated_Synthetic_Data.csv', index=False)
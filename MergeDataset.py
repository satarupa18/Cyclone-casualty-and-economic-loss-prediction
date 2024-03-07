import pandas as pd

 # Load the data from the CSV file
df1 = pd.read_csv("Population.csv")
df3 = pd.read_csv("Historical.csv")

# # Merge Population and Historical datasets using STATE and PLACE columns
data = pd.merge(df1,df3, on='STATE' ,how='inner')

data_sorted = data.sort_values(by=['YEAR','STATE'])
# #Save the merge dataset 
data_sorted.to_csv('Historical_Population.csv',index=False)



# Load historical dataset from CSV
historical_df = pd.read_csv('Historical_Population.csv')
# Load GDP dataset from CSV
gdp_df = pd.read_csv('Modified_Gdp.csv')

merged_df = pd.merge_asof(historical_df, gdp_df, 
                         by='STATE', on='YEAR', direction='nearest')


merged_df.to_csv("Final_Data.csv",index=False)


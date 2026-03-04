import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression
import joblib

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="car_project",
    user="postgres",
    password="1234",
    host="postgres"
)

# Fetch data
query = "SELECT hp, vol, wt, mpg FROM cars;"
df = pd.read_sql(query, conn)

print("Data Loaded:")
print(df.head())

# Features and target
X = df[['hp', 'vol', 'wt']]
y = df['mpg']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "car_mpg_model.pkl")

print("Model trained and saved successfully!")

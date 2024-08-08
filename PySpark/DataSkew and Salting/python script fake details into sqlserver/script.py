import pyodbc
import pandas as pd
from faker import Faker
import random

# Initialize the Faker library
fake = Faker()

# Database connection details
server = 'COGNINE-L143'   # Replace with your SQL Server
database = 'Data'   # Replace with your database
username = 'Read'   # Replace with your username
password = 'Welcome2cognine'   # Replace with your password

# Connection string
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Connect to the SQL Server
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Define the table name
table_name = '[220k_awards_by_directors]'

# Read existing data from the table
query = f"SELECT Top(10)* FROM {table_name}"
existing_data = pd.read_sql(query, conn)

# Generate random data
def generate_random_data(existing_data):
    new_data = []
    
    for _ in range(len(existing_data)):
        # Generate random data based on existing data
        row = {
            'director_name': fake.name(),
            'ceremony': random.choice(existing_data['ceremony']),
            'year': random.choice(existing_data['year']),
            'category': random.choice(existing_data['category']),
            'outcome': random.choice(existing_data['outcome']),
            'original_language': random.choice(existing_data['original_language'])
        }
        new_data.append(row)
        print('new data generated: ',new_data)
    
    return pd.DataFrame(new_data)

# Generate new data
new_data_df = generate_random_data(existing_data)

# Append the new data to the existing table
for index, row in new_data_df.iterrows():
    cursor.execute(f"""
        INSERT INTO {table_name} (director_name, ceremony, year, category, outcome, original_language)
        VALUES (?, ?, ?, ?, ?, ?)
    """, row['director_name'], row['ceremony'], row['year'], row['category'], row['outcome'], row['original_language'])

# Commit the transaction
conn.commit()
print('committed to sql server')

# Close the connection
cursor.close()
conn.close()

print(f"Appended {len(new_data_df)} new rows to {table_name}.")

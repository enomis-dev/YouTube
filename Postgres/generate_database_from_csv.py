import psycopg2
import csv
import os
from secrets import DB_PASSWORD

conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres",
                        password=DB_PASSWORD, port=5432)

cur = conn.cursor()

cwd = os.getcwd()
csv_file_path = os.path.join(cwd, 'countries.csv')
table_name = 'countries'

# Create the table in PostgreSQL
cur.execute("""CREATE TABLE IF NOT EXISTS countries (
    ID INT PRIMARY KEY,
    country VARCHAR(255),
    capital VARCHAR(255),
    population INT        
);
""")

# Import data from the CSV into the table
with open(csv_file_path, 'r') as file:
    header = file.readline().strip()
    csv_reader = csv.reader(file)
    insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(['%s'] * len(header.split(',')))})"
    for row in csv_reader:
        cur.execute(insert_query, row)
  
# print all the countries
sql2 = """SELECT * FROM countries WHERE population > 80000000;"""
cur.execute(sql2)
for i in cur.fetchall():
    print(i)
  
# close all connection 
conn.commit()
conn.close()




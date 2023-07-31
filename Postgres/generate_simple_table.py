import psycopg2
from secrets import DB_PASSWORD

password = DB_PASSWORD # replace with your password
# connect to postgres
conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres",
                        password=password, port=5432)

# Create a cursor object
# With the cursor object, you can perform various database operations as sql queries
cur = conn.cursor()

# execute simple query to create a table
cur.execute("""CREATE TABLE IF NOT EXISTS workers (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    age INT,
    job VARCHAR(255)        
);
""")

# fill the table
cur.execute("""INSERT INTO workers (id, name, age, job) VALUES 
(1, 'Luke', 30, 'engineer'),
(2, 'Lisa', 40, 'professor'),
(3, 'Mary', 50, 'doctor'),
(4, 'Robert', 30, 'plumber')
""")

# select rows with name == Mary
cur.execute("""SELECT * FROM workers WHERE name = 'Mary';""")
# print first value of the query
print(cur.fetchone())

print()
# select rows with age < 35
cur.execute("""SELECT * FROM workers WHERE age < 35;""")
# print all found the rows
for row in cur.fetchall():
    print(row)

print()
# custom query with input arguments
sql = cur.mogrify("""SELECT * FROM workers WHERE starts_with(name, %s) AND age <= %s;""", ("L", 40))
print(sql)

print()
# execute query
cur.execute(sql)
# print all the row
print(cur.fetchall())

# method is used to save any changes made to the database after executing
# data manipulation operations  as part of a transaction
conn.commit()

# close cursor
cur.close()
# close connection to database
conn.close()
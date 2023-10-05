from flask import Flask, render_template, request, redirect, url_for
import psycopg2

app = Flask(__name__)

# PostgreSQL configuration from docker-compose
db_config = {
    'host': 'postgres', 
    'database': 'mydb',
    'user': 'myuser',
    'password': 'mypassword',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_table', methods=['GET', 'POST'])
def create_table():
    if request.method == 'POST':
        try:
            # Connect to the PostgreSQL database
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Create a simple table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sample_data (
                    id serial PRIMARY KEY,
                    name VARCHAR(255),
                    age INT
                )
            ''')
            conn.commit()

            return "Table 'sample_data' created successfully!"

        except Exception as e:
            return str(e)

    return render_template('create_table.html')

@app.route('/insert_data', methods=['GET', 'POST'])
def insert_data():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']

        try:
            # Connect to the PostgreSQL database
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Insert data into the table
            cursor.execute("INSERT INTO sample_data (name, age) VALUES (%s, %s)", (name, age))
            conn.commit()

            return redirect(url_for('query_data'))  # Redirect to the query page after insertion

        except Exception as e:
            return str(e)

    return render_template('insert_data.html')

@app.route('/query_data')
def query_data():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Query the sample data from the table
        cursor.execute("SELECT * FROM sample_data")
        data = cursor.fetchall()

        return render_template('query_data.html', data=data)

    except Exception as e:
        return str(e)
    
@app.route('/clean_table', methods=['POST'])
def clean_table():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Clean (truncate) the table
        cursor.execute("TRUNCATE TABLE sample_data RESTART IDENTITY;")
        conn.commit()

        return redirect(url_for('query_data'))  # Redirect to the query page after cleaning

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

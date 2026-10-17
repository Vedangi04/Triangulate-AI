import sqlite3

# This creates the actual database file
conn = sqlite3.connect('company_data.db')
cursor = conn.cursor()

# Create the table
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    units_sold INTEGER,
    revenue REAL,
    quarter TEXT
)
''')

# Insert mock semiconductor data
mock_data = [
    ('H100 GPU', 5000, 150000000.0, '2025-Q3'),
    ('A100 GPU', 3000, 45000000.0, '2025-Q3'),
    ('Blackwell B200', 100, 4000000.0, '2025-Q3'),
    ('H100 GPU', 6200, 186000000.0, '2025-Q4'),
    ('Blackwell B200', 1500, 60000000.0, '2025-Q4')
]

cursor.executemany('INSERT INTO sales (product_name, units_sold, revenue, quarter) VALUES (?, ?, ?, ?)', mock_data)
conn.commit()
conn.close()
print("✅ company_data.db created with mock sales data!")
# How to Run

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder.
3. Create a virtual environment: python -m venv .venv
4. Activate it (Windows): .venv\Scripts\activate
5. Install packages: python -m pip install -r requirements.txt
6. Start the server: python backend/app.py
7. Open http://localhost:5000 in a browser.

## Run the tests
python -m pytest backend/tests -q

## Features to try
- Filter by price, GSM, stretch, colour, rating and stock
- Click a product to see KD-tree recommendations
- Insights tab: tree statistics and the benchmark against brute force

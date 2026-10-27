"""
Dashboard support code: read-only access to the SQLite store (data.py),
colors and labels (theme.py) and Plotly chart builders (charts.py).

app.py only arranges these on the page. Keeping the queries and charts here
means they can be tested without a running Streamlit server, and the app
never re-derives a result the pipeline already stored.
"""

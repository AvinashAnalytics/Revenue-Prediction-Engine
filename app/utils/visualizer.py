import plotly.express as px

def create_pie_chart():
    """Create a pie chart for model accuracy."""
    return px.pie(
        values=[85, 15],
        names=['Model Accuracy', 'Error Margin'],
        title='Prediction Accuracy Breakdown'
    )

def create_trend_chart():
    """Create a line chart for revenue trends."""
    trend_data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'Revenue': [2.5, 2.8, 3.1, 2.9, 3.2]
    }
    return px.line(
        trend_data,
        x='Month',
        y='Revenue',
        title='Monthly Revenue Trend',
        markers=True
    )
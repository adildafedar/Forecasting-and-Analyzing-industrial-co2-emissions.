from flask import Flask, render_template, request, make_response
import joblib
import numpy as np
from xhtml2pdf import pisa
from io import BytesIO

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        company = request.form['company']
        year = int(request.form['year'])
        limit = float(request.form['limit'])
        oil = float(request.form['oil'])
        gas = float(request.form['gas'])
        coal = float(request.form['coal'])
        flared = float(request.form['flared'])
        captured = float(request.form['captured'])

        features = np.array([[oil, gas, coal, flared, captured, year]])
        predicted_emissions = model.predict(features)[0]

        predicted_mt = predicted_emissions / 1_000_000  # convert to million tons
        limit_mt = limit
        over_limit = predicted_mt - limit_mt

        if over_limit > 0:
            comment = f"⚠️ Emissions exceed the limit by {over_limit:.2f} million metric tons."
            credits_needed = f"{over_limit:.2f} million carbon credits required."
        else:
            comment = "✅ Emissions are within the approved limit."
            credits_needed = "No carbon credits required."

        prediction = {
            'company': company,
            'year': year,
            'predicted': predicted_mt,
            'limit': limit_mt,
            'comment': comment,
            'credits': credits_needed
        }

    return render_template('index.html', prediction=prediction)

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.form.to_dict()
    prediction = {
        'company': data.get('company'),
        'year': int(data.get('year')),
        'predicted': float(data.get('predicted')),
        'limit': float(data.get('limit')),
        'comment': data.get('comment'),
        'credits': data.get('credits')
    }

    html = render_template('report_template.html', prediction=prediction)
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)

    if pisa_status.err:
        return "Error generating PDF", 500

    response = make_response(pdf.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=co2_forecast_report.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)

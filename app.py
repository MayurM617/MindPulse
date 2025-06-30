from flask import Flask, render_template, request, jsonify

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Home route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Prediction route with NEGATIVE checked first
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        eeg = float(data['eeg'])

        # ✅ Emotion classification (NEGATIVE first)
        if -1580 <= eeg <= -500:
            emotion = "NEGATIVE 😔"
        elif -500 < eeg < 500:
            emotion = "NEUTRAL 😐"
        elif 500 <= eeg <= 1960:
            emotion = "POSITIVE 😊"
        else:
            emotion = "Unknown / Out of Range ❓"

        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)

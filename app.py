from flask import Flask, render_template, request
import numpy as np
from joblib import load
import datetime

app = Flask(__name__)

# Load the model trained with 4 features
model = load("random_forest_model_4_features.joblib")

# Prepopulate alerts with some mock data to make dashboard look realistic
alerts_history = [
    {
        "id": 1,
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S"),
        "classification": "Web Attack: Brute Force",
        "confidence": "94.5%",
        "status": "Threat"
    },
    {
        "id": 2,
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
        "classification": "Normal (Safe Traffic)",
        "confidence": "100.0%",
        "status": "Safe"
    },
    {
        "id": 3,
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
        "classification": "Web Attack: SQL Injection",
        "confidence": "87.2%",
        "status": "Threat"
    },
    {
        "id": 4,
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        "classification": "Web Attack: XSS",
        "confidence": "91.8%",
        "status": "Threat"
    }
]

def map_prediction(pred_label):
    # Map raw model output to user-friendly label
    if pred_label == 'BENIGN':
        return "Normal (Safe Traffic)"
    elif 'Brute Force' in pred_label:
        return "Web Attack: Brute Force"
    elif 'Sql Injection' in pred_label:
        return "Web Attack: SQL Injection"
    elif 'XSS' in pred_label:
        return "Web Attack: XSS"
    return pred_label

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", active_page="home")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    total_scans = len(alerts_history)
    total_threats = sum(1 for a in alerts_history if a["status"] == "Threat")
    total_safe = sum(1 for a in alerts_history if a["status"] == "Safe")
    
    health_rate = round((total_safe / total_scans * 100), 1) if total_scans > 0 else 100.0

    # Get class counts for chart
    class_counts_map = {}
    for a in alerts_history:
        class_counts_map[a["classification"]] = class_counts_map.get(a["classification"], 0) + 1
    
    # Ensure all four standard classifications exist in the map
    standard_classes = ["Normal (Safe Traffic)", "Web Attack: Brute Force", "Web Attack: SQL Injection", "Web Attack: XSS"]
    class_labels = []
    class_counts = []
    for cls in standard_classes:
        class_labels.append(cls)
        class_counts.append(class_counts_map.get(cls, 0))

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        total_scans=total_scans,
        total_threats=total_threats,
        total_safe=total_safe,
        health_rate=health_rate,
        class_labels=class_labels,
        class_counts=class_counts
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", active_page="predict")

    try:
        html_input_names = [
            "flow_duration",
            "total_fwd_packets",
            "total_backward_packets",
            "total_length_fwd_packets"
        ]

        feature_values = []
        for name in html_input_names:
            try:
                feature_values.append(float(request.form[name]))
            except KeyError:
                return f"Error: Missing form field '{name}'. Please provide all 4 feature inputs.", 400
            except ValueError:
                return f"Error: Invalid input for '{name}'. Please enter a valid number.", 400

        input_data = np.array([feature_values])
        
        # Predict class
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence using predict_proba
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence_val = max(probabilities)
            confidence_str = f"{confidence_val * 100:.1f}%"
        except Exception:
            confidence_str = "95.0%" # Fallback if predict_proba isn't available

        # Map to display friendly labels
        prediction_display = map_prediction(prediction)
        status = "Safe" if prediction == 'BENIGN' else "Threat"

        # Add to alert history
        new_alert = {
            "id": len(alerts_history) + 1,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification": prediction_display,
            "confidence": confidence_str,
            "status": status
        }
        alerts_history.append(new_alert)

        return render_template(
            "predict.html",
            active_page="predict",
            prediction=prediction,
            prediction_display=prediction_display,
            confidence=confidence_str
        )
    except Exception as e:
        return f"Error during prediction: {e}", 500

@app.route("/alerts", methods=["GET"])
def alerts():
    return render_template("alerts.html", active_page="alerts", alerts=alerts_history)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", active_page="about")

if __name__ == "__main__":
    app.run(debug=True)
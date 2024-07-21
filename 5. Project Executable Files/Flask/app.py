from flask import Flask, render_template, request
import pickle


# Load models and scalers

sc = pickle.load(open("ms.pkl", "rb"))  # MinMaxScaler or StandardScaler
model = pickle.load(open("rf_model.pkl", "rb"))  # RandomForest Model


app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("proj.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    try:
        Weight_in_gms = request.form["Weight_in_gms"]
        Cost_of_the_Product = request.form["Cost_of_the_Product"]
        Prior_purchases = request.form["Prior_purchases"]
        Discount_offered = request.form["Discount_offered"]
        Product_importance = request.form["Product_importance"]
        Customer_rating = request.form["Customer_rating"]
        Customer_care_calls = request.form["Customer_care_calls"]

        # Encode Product_importance
        prodimpdict = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
        # Prepare input data
        preds = [[
            float(Cost_of_the_Product), 
            float(Customer_rating), 
            int(Customer_care_calls), 
            int(Prior_purchases), 
            prodimpdict[Product_importance], 0.0, 
            float(Discount_offered), 
            float(Weight_in_gms)
        ]]

        # Transform and predict
        transformed_preds = sc.transform(preds)
        prediction = model.predict(transformed_preds)
        prediction_proba = model.predict_proba(transformed_preds)[0]

        prob = prediction_proba[1]

        prediction_text = 'There is a {:.2f}% chance that your product will reach in time'.format(prob * 100)
        print(prediction_text)
        print(prediction)

        return render_template("proj.html", prediction_text=prediction_text)
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=False)

from flask import Flask, request, jsonify
from flask_cors import CORS
import dill
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.ERROR)

@app.route('/mdnm_ml', methods=['GET'])
def mdnm_ml():
    try:
        input_text = str(request.args.get('verbatim'))

        if not input_text:
            return jsonify({"message": "No text provided"}), 400

        # Load the prediction model
        try:
            with open('static/model/predict_mdnm_pipeline2.pkl', 'rb') as file:
                loaded_pipeline = dill.load(file)
        except FileNotFoundError:
            return jsonify({"message": "Model file not found"}), 500
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return jsonify({"message": "Error loading the model"}), 500

        # Make the prediction
        try:
            y_pred = loaded_pipeline.predict(input_text)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return jsonify({"message": "Error during prediction"}), 500

        # Return the prediction results
        if y_pred is not None:
            predicted_tc = y_pred['response']
            similar_mdnm = y_pred['similar_mdnm']
            filtered_similar_mdnm = list(set(tc for tc in similar_mdnm if tc != predicted_tc))

            logging.info(f"Predicted term: {predicted_tc}")
            return jsonify({
                'result': predicted_tc,
                'confidence': y_pred['confidence_percentage'],
                'similar_mdnm': filtered_similar_mdnm
            })
        else:
            return jsonify({'message': 'No best match found'}), 404

    except Exception as e:
        logging.error(f"General error in /mdnm_ml route: {str(e)}")
        return jsonify({"message": "An internal error occurred"}), 500


@app.route('/tc_ml', methods=['GET'])
def tc_ml():
    try:
        input_text1 = str(request.args.get('verbatim'))
        input_text2 = str(request.args.get('indication'))
        input_text3 = str(request.args.get('route'))

        if not input_text1 or not input_text2 or not input_text3:
            return jsonify({"message": "One or more text inputs not provided"}), 400

        # Load the prediction model
        try:
            with open('static/model/predict_tc_pipeline.pkl', 'rb') as file:
                loaded_pipeline = dill.load(file)
        except FileNotFoundError:
            return jsonify({"message": "Model file not found"}), 500
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return jsonify({"message": "Error loading the model"}), 500

        # Make the prediction
        try:
            y_pred = loaded_pipeline.predict(input_text1, input_text2, input_text3)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return jsonify({"message": "Error during prediction"}), 500

        # Return the prediction results
        if y_pred is not None:
            predicted_tc = y_pred['term1']
            similar_tc = y_pred['similar_TC']
            drug_number = y_pred.get('DRUGNUMBER')
            atc_code = y_pred.get('atc_code')

            filtered_similar_tc = list(set(tc for tc in similar_tc if tc != predicted_tc))

            logging.info(f"Predicted term1: {predicted_tc}")
            return jsonify({
                'result': predicted_tc,
                'confidence': y_pred['confidence_percentage'],
                'similar_tc': filtered_similar_tc,
                'drug_number': drug_number,
                'atc_code': atc_code
            })
        else:
            return jsonify({'message': 'No best match found'}), 404

    except Exception as e:
        logging.error(f"General error in /tc_ml route: {str(e)}")
        return jsonify({"message": "An internal error occurred"}), 500


if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', threaded=True)
    except Exception as e:
        logging.error(f"An error occurred when starting the app: {str(e)}")
        print(f"An error occurred: {str(e)}")

from flask import Flask, request, jsonify
from flask_cors import CORS
import dill


app = Flask(__name__)
CORS(app)


@app.route('/mdnm_ml', methods=['GET'])
def mdnm_ml():
    input_text = str(request.args.get('verbatim'))
    
    if not input_text:
        return "No text provided", 400
    
    with open('static/model/predict_mdnm_pipeline2.pkl', 'rb') as file:
        loaded_pipeline = dill.load(file)


    y_pred = loaded_pipeline.predict(input_text)

    if y_pred is not None:
        predicted_mdm = y_pred['response']
        similar_mdnm = y_pred['similar_MDNM']

        filtered_similar_mdnm = list(set(mdnm for mdnm in similar_mdnm if mdnm != predicted_mdm))
        print(f"Predicted MDNM: {y_pred['response']}")
        print(f"Confidence Percentage: {y_pred['confidence_percentage']}")
        print(f"Similar MDNM: {filtered_similar_mdnm}")
        
        response = jsonify({'result': y_pred['response'], 'confidence': y_pred['confidence_percentage'], 'similar_mdnm':filtered_similar_mdnm})
        
        
    else:
        response = jsonify({'message': 'No best match found'})

    return response



@app.route('/tc_ml', methods=['GET'])
def tc_ml():
    input_text1 = str(request.args.get('verbatim'))
    input_text2 = str(request.args.get('indication'))
    input_text3 = str(request.args.get('route'))
   
   
    
    if not input_text1 or not input_text2 or not input_text3:
        return "One or more text inputs not provided", 400
    
    with open('static/model/predict_tc_pipeline.pkl', 'rb') as file:
        loaded_pipeline = dill.load(file)


    y_pred = loaded_pipeline.predict(input_text1, input_text2, input_text3)

    if y_pred is not None:
        predicted_tc = y_pred['response']
        similar_tc = y_pred['similar_TC']

        filtered_similar_tc = list(set(tc for tc in similar_tc if tc != predicted_tc))
        print(f"Predicted tc: {y_pred['response']}")
        print(f"Confidence Percentage: {y_pred['confidence_percentage']}")
        print(f"similar_TC: {filtered_similar_tc}")
        
        response = jsonify({'result': y_pred['response'], 'confidence': y_pred['confidence_percentage'], 'similar_tc': filtered_similar_tc})
        
        
    else:
        response = jsonify({'message': 'No best match found'})

    return response


if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', threaded=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
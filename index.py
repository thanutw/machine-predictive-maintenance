from flask import Flask, redirect, url_for, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
failure_types = {
    0 : 'Heat Dissipation Failure',
    1 : 'No Failure',
    2 : 'Overstrain Failure',
    3 : 'Power Failure',
    4 : 'Random Failures',
    5 : 'Tool Wear Failure',
}
pickle_model = pickle.load(open('mpm-model.sav', 'rb'))
pickle_scaler = pickle.load(open('mpm-scaler.sav', 'rb'))


@app.route('/')
def home():
    # return render_template('index.html', content='mycustom_content')
    return render_template('form.html')

@app.route('/form', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        torque_data = request.form['torque']
        return redirect(url_for('result', torque=torque_data))
    else:
        return render_template('form.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.values()
    data_arr = np.array(list(data)).reshape(1, -1)
    data_scaled = pickle_scaler.transform(data_arr)
    predict_value = pickle_model.predict(data_scaled)
    prediction_text = failure_types[predict_value[0]]
    return render_template('form.html', prediction_text=prediction_text)

@app.route('/result/<torque>')
def result(torque):
    return f'{torque}'

if __name__ == '__main__':
    debug_mode = True
    app.run(debug=debug_mode)
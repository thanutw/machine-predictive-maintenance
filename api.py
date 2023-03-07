import uvicorn
from fastapi import FastAPI, File
import pickle
from pydantic import BaseModel
from typing import Union
import numpy as np

app = FastAPI()
pickle_model = pickle.load(open('mpm-model.sav', 'rb'))
pickle_scaler = pickle.load(open('mpm-scaler.sav', 'rb'))

class MachineData(BaseModel):
    machine_type: int
    air_temp: float
    torque: float
    tool_wear: float

failure_types = {
    0 : 'Heat Dissipation Failue',
    1 : 'No Failure',
    2 : 'Overstrain Failue',
    3 : 'Power Failure',
    4 : 'Random Failures',
    5 : 'Tool Wear Failure',
}

@app.get('/')
async def index():
    return {'message': 'hello world'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}

@app.post('/items')
async def create_item(data: MachineData):
    data = data.dict()
    m_type = data['machine_type'] # L=0, M=1, H=2
    air_temp = data['air_temp']
    torque = data['torque']
    tool_wear = data['tool_wear']
    input_data = np.array([[m_type, air_temp, torque, tool_wear]])
    input_data_scaled = pickle_scaler.transform(input_data)
    y_predict = pickle_model.predict(input_data_scaled)
    return {'predicted_failure': f'{failure_types[y_predict[0]]}'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import math
# from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
import streamlit as st

#Load lại các giá trị
label_encoders = joblib.load('label_encoders.pkl')
label_scaler = joblib.load('label_scalers.pkl')

column_encoders = ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", 
                   "Festival", "City", "Time_of_Day"]
column_scalers = ["Delivery_person_Age", "Delivery_person_Ratings", "Time_taken(min)", "distance"]

class ProcessingData:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def replace_columns_value(self):
        self.data_frame["Time_taken(min)"] = self.data_frame["Time_taken(min)"].str.replace("(min)", "")
        self.data_frame['Road_traffic_density'] = self.data_frame['Road_traffic_density'].str.strip()
        self.data_frame['Type_of_order'] = self.data_frame['Type_of_order'].str.strip()
        self.data_frame['Type_of_vehicle'] = self.data_frame['Type_of_vehicle'].str.strip()
        self.data_frame['Festival'] = self.data_frame['Festival'].str.strip()
        self.data_frame['City'] = self.data_frame['City'].str.strip()

    def convert_dtype(self):
        self.data_frame["Delivery_person_Age"] = self.data_frame["Delivery_person_Age"].astype("int64")
        self.data_frame["Delivery_person_Ratings"] = self.data_frame["Delivery_person_Ratings"].astype("float")
        self.data_frame['Time_Orderd'] = pd.to_datetime(self.data_frame['Time_Orderd'])
        self.data_frame['Time_taken(min)'] = self.data_frame['Time_taken(min)'].astype("int")

    def get_time(self, hour):
        if 6 <= hour < 12:
            return 'morning'
        elif hour < 13:
            return 'noon'
        elif hour < 18:
            return 'afternoon'
        else:
            return 'evening'

    def haversine(self, lati1, longi1, lati2, longi2):
        R = 6371
        lati1, longi1, lati2, longi2 = map(math.radians, [lati1, longi1, lati2, longi2])
        dlati = lati2 - lati1
        dlongi = longi2 - longi1
        a = math.sin(dlati / 2) ** 2 + math.cos(lati1) * math.cos(lati2) * math.sin(dlongi / 2) ** 2
        distance = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(distance, 2)

    def add_columns(self):
        self.data_frame['Time_of_Day'] = self.data_frame['Time_Orderd'].dt.hour.apply(self.get_time)
        for i in range(len(self.data_frame)):
            self.data_frame.loc[i, 'distance'] = self.haversine(
                self.data_frame.loc[i, 'Restaurant_latitude'],
                self.data_frame.loc[i, 'Restaurant_longitude'],
                self.data_frame.loc[i, 'Delivery_location_latitude'],
                self.data_frame.loc[i, 'Delivery_location_longitude']
            )

    def one_data_frame(self):
        self.replace_columns_value()
        self.convert_dtype()
        self.add_columns()
        self.data_frame = self.data_frame.drop(["Time_Orderd" ,"Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"], axis=1)
        for column in column_encoders:
            self.data_frame[column] = label_encoders[column].transform(self.data_frame[column])
        for column in column_scalers:
            self.data_frame[column] = label_scaler[column].transform(self.data_frame[column].values.reshape(-1, 1))
        print(self.data_frame)
        return self.data_frame


st.title("Food Delivery Time Prediction")

delivery_person_age = st.number_input('Delivery Person Age', min_value=0, value=25)
delivery_person_ratings = st.number_input('Delivery Person Ratings', min_value=0.0, max_value=5.0, value=4.6)
Restaurant_latitude = st.number_input('Restaurant_latitude', value=13.066762)
Restaurant_longitude = st.number_input('Delivery Location Longitude', value=80.251865)
delivery_location_latitude = st.number_input('Delivery Location Latitude', value=13.146762)
delivery_location_longitude = st.number_input('Delivery Location Longitude', value=80.331865)
time_ordered = st.text_input('Time Ordered (HH:MM:SS)', '19:50:00')
weather_conditions = st.selectbox('Weather Conditions', ['conditions Sunny', 'conditions Stormy', 
                                                         'conditions Sandstorms', 'conditions Cloudy',
                                                         'conditions Fog', 'conditions Windy'],index=2)
road_traffic_density = st.selectbox('Road Traffic Density', ['High', 'Jam', 'Low', 'Medium'],index=1)
vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=2, value=2)
type_of_order = st.selectbox('Type of Order', ['Snack', 'Drinks', 'Buffet', 'Meal'],index=1)
type_of_vehicle = st.selectbox('Type of Vehicle', ['motorcycle', 'scooter', 'electric_scooter'], index=1)
festival = st.selectbox('Festival', ['No', 'Yes'], index=0)
city = st.selectbox('City', ['Urban', 'Metropolitian', 'Semi-Urban'], index=1)

if st.button('Predict'):
    # Tạo DataFrame cho dữ liệu đầu vào
    data_test = pd.DataFrame({
        "Delivery_person_Age": [delivery_person_age],
        "Delivery_person_Ratings": [delivery_person_ratings],
        "Restaurant_latitude": [Restaurant_latitude], 
        "Restaurant_longitude": [Restaurant_longitude], 
        "Delivery_location_latitude": [delivery_location_latitude],
        "Delivery_location_longitude": [delivery_location_longitude],
        "Time_Orderd": [time_ordered],
        "Weatherconditions": [weather_conditions],
        "Road_traffic_density": [road_traffic_density],
        "Vehicle_condition": [vehicle_condition],
        "Type_of_order": [type_of_order],
        "Type_of_vehicle": [type_of_vehicle],
        "Festival": [festival],
        "City": [city],
        "Time_taken(min)": ["(min) 25"]
    })

    # Xử lý dữ liệu
    process_data = ProcessingData(data_test)
    print(data_test)
    print(label_encoders)
    print(label_scaler)
    dt = process_data.one_data_frame()
    print(dt)
    model = joblib.load('model_delivery_time.pth')
    # Dự đoán
    pred = model.predict(dt[["Delivery_person_Age", "Delivery_person_Ratings", "Weatherconditions", 
                               "Road_traffic_density", "Vehicle_condition", "Type_of_order", 
                               "Type_of_vehicle", "Festival", "City", "Time_of_Day", "distance"]])
    
    pred_time_taken = label_scaler["Time_taken(min)"].inverse_transform(pred.reshape(-1, 1))
    print(pred)
    print(pred_time_taken)

    st.success(f'Predicted Time Taken: {pred_time_taken[0][0]} minutes')
# streamlit run deploy_Flask.py





# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Lấy dữ liệu từ biểu mẫu
#         data_test = pd.DataFrame({
#             "Delivery_person_Age": [request.form['Delivery_person_Age']],
#             "Delivery_person_Ratings": [request.form['Delivery_person_Ratings']],
#             "Restaurant_latitude": [13.066762], 
#             "Restaurant_longitude": [80.251865], 
#             "Delivery_location_latitude": [request.form['Delivery_location_latitude']],
#             "Delivery_location_longitude": [request.form['Delivery_location_longitude']],
#             "Time_Orderd": [request.form['Time_Orderd']],
#             "Weatherconditions": [request.form['Weatherconditions']],
#             "Road_traffic_density": [request.form['Road_traffic_density']],
#             "Vehicle_condition": [request.form['Vehicle_condition']],
#             "Type_of_order": [request.form['Type_of_order']],
#             "Type_of_vehicle": [request.form['Type_of_vehicle']],
#             "Festival": [request.form['Festival']],
#             "City": [request.form['City']],
#             "Time_taken(min)": ["(min) 25"]  
#         })

#         # Xử lý dữ liệu đầu vào và dự đoán
#         process_2 = ProcessingData(data_test)
#         dt = process_2.one_data_frame()

#         # Dự đoán
#         model = joblib.load('model_delivery_time.pth')
#         pred = model.predict(dt[["Delivery_person_Age", "Delivery_person_Ratings", "Weatherconditions", 
#                                    "Road_traffic_density", "Vehicle_condition", "Type_of_order", 
#                                    "Type_of_vehicle", "Festival", "City", "Time_of_Day", "distance"]])
        
#         pred_time_taken = label_scaler["Time_taken(min)"].inverse_transform(pred.reshape(-1, 1))

#         # return render_template('index.html', prediction=pred_time_taken[0][0])

#     return render_template('index.html')

# if __name__ == '__main__':
#     df = pd.read_csv("data.csv")
#     process_1 = ProcessingData(df)
#     process_1.all_data_frame() 
#     app.run(debug=True)
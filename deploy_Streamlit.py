import joblib
import pandas as pd
import numpy as np
import math
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

#Load lại các giá trị
label_encoders = joblib.load('checkpoints/label_encoders.pkl')
label_scaler = joblib.load('checkpoints/label_scalers.pkl')

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
        if 1 <= hour < 11:
            return 'morning'
        elif hour < 13:
            return 'noon'
        elif hour < 19:
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





st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🚚",
    layout="wide",
)

st.title("Food Delivery Time Prediction")
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

col1, col2 =st.columns([1, 3])

with col1:
    st.session_state.delivery_person_age = st.number_input('Delivery Person Age', min_value=0, value=25, key='age')
    st.session_state.delivery_person_ratings = st.number_input('Delivery Person Ratings', min_value=0.0, max_value=5.0, value=4.6)
    st.session_state.Restaurant_latitude = st.number_input('Restaurant_latitude', value=13.066762)
    st.session_state.Restaurant_longitude = st.number_input('Delivery Location Longitude', value=80.251865)
    st.session_state.delivery_location_latitude = st.number_input('Delivery Location Latitude', value=13.146762)
    st.session_state.delivery_location_longitude = st.number_input('Delivery Location Longitude', value=80.331865)
    st.session_state.time_ordered = st.text_input('Time Ordered (HH:MM:SS)', '19:50:00')
    st.session_state.weather_conditions = st.selectbox('Weather Conditions', ['conditions Sunny', 'conditions Stormy', 'conditions Sandstorms', 'conditions Cloudy','conditions Fog', 'conditions Windy'],index=2)
    st.session_state.road_traffic_density = st.selectbox('Road Traffic Density', ['High', 'Jam', 'Low', 'Medium'],index=1)
    st.session_state.vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=2, value=2)
    st.session_state.type_of_order = st.selectbox('Type of Order', ['Snack', 'Drinks', 'Buffet', 'Meal'],index=1)
    st.session_state.type_of_vehicle = st.selectbox('Type of Vehicle', ['motorcycle', 'scooter', 'electric_scooter'], index=1)
    st.session_state.festival = st.selectbox('Festival', ['No', 'Yes'], index=0)
    st.session_state.city = st.selectbox('City', ['Urban', 'Metropolitian', 'Semi-Urban'], index=1)


with col2:
    st.markdown("<div style='margin: 30px;'></div>", unsafe_allow_html=True)
    data_test = pd.read_csv('datasets/data_test_deploy.csv')
    
    # Bảng AgGrid
    gb = GridOptionsBuilder.from_dataframe(data_test)
    gb.configure_selection('single', use_checkbox=False)
    grid_response = AgGrid(data_test, gridOptions= gb.build(), height=600, enable_events=True)
    selected_rows = grid_response.get('selected_rows', None)
    # Sự kiện
    if selected_rows is not None and not selected_rows.empty:
        row = selected_rows.iloc[0]
        
        st.session_state.delivery_person_age = row['Delivery_person_Age']
        st.session_state.delivery_person_ratings = row['Delivery_person_Ratings']
        st.session_state.Restaurant_latitude = row['Restaurant_latitude']
        st.session_state.Restaurant_longitude = row['Restaurant_longitude']
        st.session_state.delivery_location_latitude = row['Delivery_location_latitude']
        st.session_state.delivery_location_longitude = row['Delivery_location_longitude']
        st.session_state.time_ordered = row['Time_Orderd']
        st.session_state.weather_conditions = row['Weatherconditions']
        st.session_state.road_traffic_density = row['Road_traffic_density']
        st.session_state.vehicle_condition = row['Vehicle_condition']
        st.session_state.type_of_order = row['Type_of_order']
        st.session_state.type_of_vehicle = row['Type_of_vehicle']
        st.session_state.festival = row['Festival']
        st.session_state.city = row['City']



    st.markdown("<div style='margin: 46px;'></div>", unsafe_allow_html=True)
    if st.button('Predict'):
        data_test = pd.DataFrame({
            "Delivery_person_Age": [st.session_state.delivery_person_age],
            "Delivery_person_Ratings": [st.session_state.delivery_person_ratings],
            "Restaurant_latitude": [st.session_state.Restaurant_latitude], 
            "Restaurant_longitude": [st.session_state.Restaurant_longitude], 
            "Delivery_location_latitude": [st.session_state.delivery_location_latitude],
            "Delivery_location_longitude": [st.session_state.delivery_location_longitude],
            "Time_Orderd": [st.session_state.time_ordered],
            "Weatherconditions": [st.session_state.weather_conditions],
            "Road_traffic_density": [st.session_state.road_traffic_density],
            "Vehicle_condition": [st.session_state.vehicle_condition],
            "Type_of_order": [st.session_state.type_of_order],
            "Type_of_vehicle": [st.session_state.type_of_vehicle],
            "Festival": [st.session_state.festival],
            "City": [st.session_state.city],
            "Time_taken(min)": ["(min) 25"]
        })

        # Xử lý dữ liệu
        process_data = ProcessingData(data_test)
        dt = process_data.one_data_frame()
        model = joblib.load('checkpoints/model_XGBRegress.pkl')
        # Dự đoán
        pred = model.predict(dt[["Delivery_person_Age", "Delivery_person_Ratings", "Weatherconditions", 
                                "Road_traffic_density", "Vehicle_condition", "Type_of_order", 
                                "Type_of_vehicle", "Festival", "City", "Time_of_Day", "distance"]])
        
        pred_time_taken = label_scaler["Time_taken(min)"].inverse_transform(pred.reshape(-1, 1))
        print(pred)
        print(pred_time_taken)
        st.markdown("<div style='margin: 21px;'></div>", unsafe_allow_html=True)
        st.success(f'Predicted Time Taken: {pred_time_taken[0][0]} minutes')
# streamlit run deploy_Streamlit.py
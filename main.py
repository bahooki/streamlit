import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk


#Retrieving Data
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


st.title("Uber Pickups in NYC")

data_load_state = st.text('Loading data...')
data = load_data(100000)
data_load_state.text('Data Loaded')


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)


hour_to_filter = st.slider(label="Pick-up Hour", min_value=0, max_value=24)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
#st.map(filtered_data)

expander = st.expander(label='Map Modifiers')


with expander:
    col1, col2 = st.columns(2)
    with col1:
        coverage_Slider = st.slider(label='Column Radius Display Modifier', min_value=0.0, max_value=5.0, value=1.0)
        radius_Slider = st.slider(label='Column Bin Radius (in meters)', min_value=0, max_value=500, value=100)
        elevation_Slider = st.slider(label='Elevation Scale', min_value=0, max_value=20, value=5)
    with col2:
        pitch_Slider = st.slider(label='Map View Angle', min_value=0, max_value=60, value=50)
        percent_Slider = st.slider(label='Lower Percentile to Remove from Map', min_value=0, max_value=100, value=0)

pDeck = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
                 initial_view_state=pdk.ViewState(
                     latitude=40.71, 
                     longitude=-73.96, 
                     zoom=11,
                    pitch=pitch_Slider),
                 tooltip={
                     'html': '<b>Count: </b>{elevationValue}'
                 },
                 layers=[
                     pdk.Layer(
                        'HexagonLayer',
                        data=filtered_data,
                        get_position='[lon, lat]',
                        radius=radius_Slider,
                        coverage=coverage_Slider,
                        elevation_scale=elevation_Slider,
                        elevation_range=[0, 1000],
                        extruded=True,
                        pickable=True,
                        elevationLowerPercentile=percent_Slider
                     )
                 ]
                )
st.pydeck_chart(pDeck)

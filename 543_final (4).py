#Core Pkgs
import streamlit as st
from PIL import Image
#Visualization Pkgs
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
import pydeck as pdk
import sqlite3


#App name
st.markdown(
    """
    <style>
        container {display: flex}.logo-text {font-weight:500 !important;font-size:60px !important;color: purple;padding-left:100px !important}
    </style>
    <div class="container">
        <p class="logo-text"> Data Visualization </p>
    </div>
    """, unsafe_allow_html=True
)
#Sidebar - presentation agenda
original = Image.open("uw.jpg")
new_img = original.resize((300, 200), Image.ADAPTIVE)
st.sidebar.image(new_img)
st.sidebar.title("MSIS 543 - Purple 8")
st.sidebar.subheader("")
st.sidebar.header("Agenda")
Xiang = st.sidebar.checkbox("Introduction - Xiang")
Jerry = st.sidebar.checkbox("Databases Collection - Jerry")
Kristen = st.sidebar.checkbox("Plotly Library - Kristen")
Leah = st.sidebar.checkbox("Altair Library - Leah")
Nash = st.sidebar.checkbox("PyDeck Library- Nash")
Stian = st.sidebar.checkbox("Matplotlib - Stian")
Q_A = st.sidebar.checkbox("Q & A")

#Mainpage
if Kristen:
    st.title("Time Series Dataset Example")
    st.header("")
    # Load file
    timeSeries = pd.read_csv("kristen_dataset.csv", parse_dates=['Mon-Yr'])
    # Create a filter
    all_columns = timeSeries.columns.tolist()
    all_columns.remove("Mon-Yr")
    choices = st.multiselect("Filter", all_columns)

    if not choices:
        title = "CA vs San Francisco"
        line_y = ["CA", "San Francisco"]
    else:
        s = ''
        for i in choices:
            s = s + ' vs ' + i
        title = s[4:]
        line_y = choices
    # Plot
    fig = px.line(timeSeries, x="Mon-Yr", y=line_y, title=title,color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_xaxes(rangeslider_visible=True, title='Date')
    fig.update_yaxes(dtick=0.2, tickformat='%', zeroline=True, zerolinecolor='#acaeae',title='YTY% Chg. in Sales')
    fig.update_layout(paper_bgcolor='#ffffff', plot_bgcolor="#f6f8f9", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    #Explanation
    with st.expander("Explanation"):
        #dataset
        st.dataframe(timeSeries,800,200)
        #codes
        st.write("Codes")
        st.code('''	
        #Import Packages
        import Streamlit as st
        import pandas as pd
        import plotly.express as px
        
        # Load file
        timeSeries = pd.read_csv("kristen_dataset.csv", parse_dates=['Mon-Yr'])
    
        #Filter
        all_columns = timeSeries.columns.tolist()
        all_columns.remove("Mon-Yr")
        choices = st.multiselect("Filter", all_columns)
        
        #Plot
        fig = px.line(timeSeries, x="Mon-Yr", y=line_y, title=title,color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_xaxes(rangeslider_visible=True, title='Date')
        fig.update_yaxes(dtick=0.2, tickformat='%', zeroline=True, zerolinecolor='#acaeae',title='YTY% Chg. in Sales')
        fig.update_layout(paper_bgcolor='#ffffff', plot_bgcolor="#f6f8f9", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)''',language = 'python')

if Xiang:
    st.title("Introduction")
    st.subheader("*The fastest way to build and share data apps*")
    st.write("")
    st.write("")
    st.write("By using Streamlit, machine learning developers can quickly develop a user interaction tool.")
    st.write("")
    st.write("")
    st.header("Advantages")
    st.write("")
    st.subheader("The API is Simple and Clear, Easy to Use")
    st.write("Streamlit's open-source app framework is a breeze to get started with. It’s just a matter of:")
    st.code("$ pip install streamlit")
    st.code("$ streamlit hello")
    st.write("And you’re done!")
    st.write("")
    st.write("")
    st.subheader("All in Python")
    st.write("Streamlit is compatible with the following Python libraries or frameworks:")
    st.write("- The data processing : Numpy, Pandas")
    st.write("- Machine learning framework or library : Scilit - Learn, TensorFlow, Keras, Pytorch")
    st.write("- Data visualization tool : matplotlib, seaborn, poltly, boken, Altair, GL")
    st.write("- Text processing : Markdown, LaTeX")
    st.write("")
    st.write("")
    st.subheader("No Front-End Experience Required")
    st.write("Before, data scientists need developers to help them put the data model into business application.")
    st.write("However, with Streamlit, the role of developers is greatly reduced, data scientists could build the apps and iterate models by their own within a very short time. ")
    st.write("")
    st.header("*Use Streamlit If You:*")
    st.write("- Not familiar with front-end design, or no front-end art cell")
    st.write("- Also don't want to realize too complicated webpage structure")
    st.write("- Just want to quickly generate a web-based GUI for my python program in a very short time")


if Jerry:
    # import/connect DB
    conn = sqlite3.connect('world.sqlite')
    c = conn.cursor()


    # SQL execution function
    def sqlexec(sqlcode):
        c.execute(sqlcode)
        data = c.fetchall()
        return data


    # data entity
    city = ['ID,', 'Name,', 'CountryCode,', 'District,', 'Population']
    country = ['Code,', 'Name,', 'Continent,', 'Region,', 'SurfaceArea', 'IndepYear', 'Population', 'LifeExpectancy',
               'GNP', 'GNPOld', 'LocalName', 'GovernmentForm', 'HeadOfState', 'Capital', 'Code2']
    countrylanguage = ['CountryCode,', 'Language,', 'IsOfficial,', 'Percentage,']

    st.title("Streamlit in Database");
    # layout
    colleft, colright = st.columns(2)

    # SQL section
    with colleft:
        with st.form(key='sqlquary_form'):
            sqlcode = st.text_area('Your SQL code:')
            execbutt = st.form_submit_button('Run')

        # tabel descriptions
        with st.expander('Table information'):
            table_info = {'city': city, 'country': country, 'countrylanguage': countrylanguage}
            st.json(table_info)
    # Outcome
    with colright:
        if execbutt:
            st.info('Query was successfully executed')
            st.code(sqlcode)

            # result of execution
            QueryOutcome = sqlexec(sqlcode)
            with st.expander('JSON result'):
                st.write(QueryOutcome)

            with st.expander('Result'):
                query_dataFrame = pd.DataFrame(QueryOutcome)
                st.dataframe(QueryOutcome)


if Leah:
    st.title("Altair Library")
    st.write("""
    Altair is a declarative statistical visualization library for Python, 
    based on Vega and Vega-Lite, and the source is available on https://altair-viz.github.io/
    """)
    st.write("""Pros: Simple Grammar Syntax. Interactive. Flexible""")
    st.write("""Cons: No 3D plotting. Not recommend datasets with above 5000 samples.""")
    st.subheader("Penguin Dataset")
    df = pd.read_json("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json")
    selection = alt.selection_single()
    c = alt.Chart(df).mark_point(size=130, filled=True).encode(
    alt.X("Body Mass (g)", scale=alt.Scale(zero=False)),
    alt.Y("Beak Length (mm)", scale=alt.Scale(zero=False)),
    tooltip=['Island', 'Sex', 'Body Mass (g)'],
    color=alt.condition(selection, 'Species', alt.value('grey'))
    ).properties(title='Body Mass vs Beak Length').add_selection(selection).interactive()

    st.altair_chart(c, use_container_width=True)

    with st.expander("Raw Data"):
        st.write(df)

# # Set subtitle
    st.subheader("Let's Explore the code 🐧")
    code = '''
    df=pd.read_json("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json")
    selection = alt.selection_single()
    c = alt.Chart(df).mark_point(size=130,filled=True).encode(
            alt.X("Body Mass (g)", scale=alt.Scale(zero=False)),
            alt.Y("Beak Length (mm)", scale=alt.Scale(zero=False)),
            tooltip = ['Island','Sex','Body Mass (g)'],
            color=alt.condition(selection, 'Species', alt.value('grey'))
        ).properties(title='Body Mass vs Beak Length')
        .add_selection(selection).interactive()
    
    st.altair_chart(c, use_container_width=True)
        '''
    st.code(code, language='python')
    st.subheader("Syntax")
    st.text('selection_single(): click single point to highlight.')
    st.text('make_point(): show data as points.')
    st.text('chart.encode(): mapping of visual properties to data.')
    st.text('alt.X: X axis.')
    st.text('alt.Y: Y axis.')
    st.text('scale: specify the scale.')
    st.text('tooltip: showing selected column values when you hover over points.')
    st.text('alt.condition():create a conditional color encoding.')
    st.text('properties:create title.')
    st.text('interactive():make chart axes scales interactive.')
    st.text('st.altair_chart(): import your python chart to streamlit.')

if Nash:
    st.title("write codes here")
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

if Stian:
	st.title("Matplotlib integration in streamlit")
	# st.text("We can use matplotlib within streamlit to show insightful visualizations")
	st.subheader("What country/ team won the most gold medals in tokyo?")
	df = pd.read_csv("Medals.csv")
	mask = df['Gold'] > 10
	fig, ax = plt.subplots()
	ax.bar(df[mask]['Team/NOC'], df[mask]['Gold'], color = "GOLD")
	# fig = plt.bar(df[mask]['Team/NOC'], df[mask]['Gold'])
	st.pyplot(fig)
	st.text("We can see that contrary to intuition, the most populous countries are not\nnecessarily the winningest ones. Notably missing from the top 6 would be India\nwhy is that?")
	# st.pyplot(fig = None, clear_figure = True)

	fig, ax = plt.subplots()
	st.subheader("Well then who won the most total medals (Gold/ Silver/ Bronze")
	mask = df['Total'] > 40
	ax.bar(df[mask]['Team/NOC'], df[mask]['Total'], color = 'PURPLE')
	st.pyplot(fig)
	st.text("We still don't see some of the larger countries represented in the\nsegment of teams that won 40+ total medals")
	
	st.subheader("New hypothesis: maybe these countries just had more competitors in the olympics")
	df = pd.read_csv("Teams.csv")
	fig, ax = plt.subplots()
	mask = df['NOC'] == "People's Republic of China"
	ax.hist(df[mask]['NOC'])
	mask = df['NOC'] == "United States of America"
	ax.hist(df[mask]['NOC'])
	mask = df['NOC'] == "Japan"
	ax.hist(df[mask]['NOC'])
	mask = df['NOC'] == "ROC"
	ax.hist(df[mask]['NOC'])
	mask = df['NOC'] == "India"
	ax.hist(df[mask]['NOC'])
	st.pyplot(fig)
	st.text("This shows that India competed in the least amount of sports compared to the winningest\nand other populous countries")

if Q_A:
    st.title("")
    QA_img = Image.open("Q_A.jpg")
    st.image(QA_img)

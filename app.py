import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
from streamlit_option_menu import option_menu # Import the option_menu

def load_data(data):
    df = pd.read_csv(data)
    return df

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="diabete_predictions.csv">Download CSV File</a>'
    return href

st.sidebar.image('images/photo_2025-05-23_18-50-58.jpg')

def main():
    st.markdown("<h1 style='text-align: center; color:brown;'>Diabete Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Diabete Study in Cameroun</h3>", unsafe_allow_html=True)

    # Replace st.sidebar.selectbox with option_menu
    with st.sidebar: # Use a 'with' block for the sidebar to ensure the menu is placed there
        selected = option_menu(
            menu_title=None,  # No title for the menu
            options=["Home", "Analysis", "Data Visualisation", "Machine Learning", "About"], # Your menu options
            icons=["house", "clipboard-data", "bar-chart", "robot", "info-circle"], # Optional: icons for each option
            menu_icon="cast", # Optional: icon for the menu itself
            default_index=0, # Default selected option
            styles={
                "container": {"padding": "5px!important", "background-color": "#fafafa"},
                "icon": {"color": "brown", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "brown"},
            }
        )
    
    data = load_data("diabetes.csv")

    if selected == "Home": # Use 'selected' instead of 'choice'
        left, middle, right = st.columns((2,3,2))
        with middle:
            st.image("images/photo_2025-05-23_18-49-29.jpg", width=400)
        st.write('This is an app that will analyse diabetes Datas with some python tools that can optimize decisions')
        st.subheader('Diabetis Information')
        st.write('In Cameroon, the prevalence of diabetes in adults in urban areas is currently estimated at 6 – 8%, with as much as 80% of people living with diabetes who are currently undiagnosed in the population. Further, according to data from Cameroon in 2002, only about a quarter of people with known diabetes actually had adequate control of their blood glucose levels. The burden of diabetes in Cameroon is not only high but is also rising rapidly. Data in Cameroonian adults based on three cross-sectional surveys over a 10-year period (1994–2004) showed an almost 10-fold increase in diabetes prevalence.')

    elif selected == "Analysis":
        st.subheader("Diabetes Analysis")
        st.dataframe(data.head())

        if st.checkbox("Summary"):
            st.write(data.describe())

        if st.checkbox("Correlation"):
            fig = plt.figure(figsize=(15,15))
            sns.heatmap(data.corr(), annot=True) # Removed st.write around sns.heatmap
            st.pyplot(fig)

        if st.checkbox("Column Names"):
            st.write(data.columns)
        st.markdown(filedownload(data), unsafe_allow_html=True)

    elif selected == "Data Visualisation": # Corrected to "Data Visualisation" as in the options
        st.subheader("Data Visualisation")
        if st.checkbox("Countplot"):
            fig = plt.figure(figsize=(15,15))
            sns.countplot(x=data['Age']) # Removed st.write around sns.countplot
            st.pyplot(fig)

        if st.checkbox("Caterplot"):
            fig = plt.figure(figsize=(15,15))
            sns.scatterplot(x='Glucose', y='Age', data=data, hue='Outcome') # Removed st.write around sns.scatterplot
            st.pyplot(fig)
        
    elif selected == "Machine Learning":
        st.subheader("Machine Learning")
        tab1,tab2,tab3=st.tabs([":clipboard: data",":bar_chart: Visualisation",":mask: Prediction"])
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

        if uploaded_file:
            df = load_data(uploaded_file)
            with tab1:
                st.subheader("Loaded Data")
                st.write(df)
            with tab2:
                st.subheader("Histogram glucose")
                fig = plt.figure(figsize=(8,8))
                sns.histplot(data=df, x='Glucose')
                st.pyplot(fig)
            with tab3:
                model = pickle.load(open('model_dump.pkl','rb'))
                prediction = model.predict(df) 
                st.subheader("Prediction")
                #transformation de l'arrey predict en datafram
                pp = pd.DataFrame(prediction, columns=['prediction'])
                #concatenation avec le df de depart

                ndf = pd.concat([df,pp],axis=1)
                #ndf.Prediction = ndf.prediction.map({0:'NO diabete',1:'diabete'})
                ndf.prediction.replace(0,'NO diabete Risk', inplace=True)
                ndf.prediction.replace(1,'diabete Risk', inplace=True)
                st.write(ndf)

                button = st.button("Download")
                if button:
                    st.markdown(filedownload(ndf), unsafe_allow_html=True)
                    st.write("Downloading..")
    elif selected == "About": # Added the "About" section
        st.subheader("About This Application")
        st.write("This application was developed to analyze diabetes data, provide insights through visualizations, and predict diabetes risk using machine learning models.")
        st.write("It serves as a tool to help understand and manage diabetes prevalence in regions like Cameroon.")


if __name__ == '__main__':
    main()
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Rain In Australia',layout='wide')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 0rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('🌦️ Rain Prediction in Australia')

@st.cache_data
def get_data():
    df = pd.read_csv(r'E:\weather in Australia\dashboard(streamlit)\Rain_AUS.csv')
    return df

df = get_data()


tab1, tab2, tab3 = st.tabs(['Wind & Rain Patterns', 'Rain & Weather Correlations', 'Location-based Analysis'])

with tab1:
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.subheader('Likelihood of Rain by Wind Direction')
        wind_counts = df.groupby(['WindGustDir', 'RainTomorrow']).size().unstack()
        wind_probabilities = wind_counts.div(wind_counts.sum(axis=1), axis=0)
        wind_probabilities = wind_probabilities.reset_index()
        wind_probabilities.columns = ['WindGustDir', 'No Rain', 'Rain']
        wind_probabilities['WindGustDir'] = wind_probabilities['WindGustDir'].astype('category')
        wind_probabilities['WindGustDir'] = wind_probabilities['WindGustDir'].cat.reorder_categories(
            ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'],
            ordered=True
        )
        fig = px.line_polar(
            wind_probabilities,
            r='Rain',
            theta='WindGustDir',
            line_close=True,
            title='Likelihood of Rain by Wind Direction',
            color_discrete_sequence=['blue']
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',polar=dict(
        bgcolor='rgba(135, 206, 250, 0.3)',radialaxis=dict(showticklabels=False) ))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('insights'):
            st.write('Directions carries the highest chance of rain, the answer would be North-West, West-North-West, and North-North-West')

    
    with col2:
        st.subheader('Rain Tomorrow based on Today\'s Rain')
        fig = px.sunburst(df, path=["RainToday", "RainTomorrow"], title="Rain tomorrow based on whether it rains today?")
        fig.update_traces(textinfo="label+percent parent")
        # Remove background
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('insights'):
            st.write("""
  - 78% of the time, it did not rain today.
  - Of those, 85% of the time, it also does not rain tomorrow.
  - However, 15% of the time, it rains tomorrow even if it didn’t rain today. 
  - 22% of the time, it rained today.
  - Of those, **54%** of the time, it does not rain tomorrow.
  - But in **46%** of the cases, it rains tomorrow again.
""")

with tab2:
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.subheader('Relation Between Pressure and Rain Tomorrow')
        df['RainTomorrow_en'] = [1 if x == 'Yes' else 0 for x in df['RainTomorrow']]
        pressure_bins = [990, 1000, 1010, 1020, 1030, 1040]
        pressure_labels = ['990-1000', '1000-1010', '1010-1020', '1020-1030', '1030-1040']
        df['Pressure_bin'] = pd.cut(df['Pressure9am'], bins=pressure_bins, labels=pressure_labels, right=False)
        fig_Pressure_grouped = px.bar(
            df.groupby(['Pressure_bin', 'RainTomorrow_en'], observed=True).size().reset_index(name='Count'),
            x='Pressure_bin',
            y='Count',
            color='RainTomorrow_en',
            barmode='group',
            title='Pressure and Rain Tomorrow',
            color_continuous_scale=['#00008B', '#FFD700']
        )
        fig_Pressure_grouped.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_Pressure_grouped, use_container_width=True)
        with st.expander('insights'):
         st.write("""The group 1000-1010 is the most likely to experience rain. This can be determined by the relatively larger proportion of the yellow section indicating rain compared to the total number of occurrences in this pressure group.""")
         
    st.markdown('### Cloud Distribution with Rain Tomorrow')
    df['RainTomorrow_en'] = [1 if x == 'Yes' else 0 for x in df['RainTomorrow']]
    fig_Cloud = px.histogram(df, x='Cloud9am', color='RainTomorrow_en', barmode='overlay', nbins=100, color_discrete_map={0: 'blue', 1: 'orange'})
    fig_Cloud.update_layout(title='Distribution of Cloud with RainTomorrow', xaxis_title='Cloud', yaxis_title='Count', plot_bgcolor='white')
    fig_Cloud.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_Cloud)
    with st.expander('insights'):
        st.write('High cloud cover (7 to 8 oktas) increases the probability of rain tomorrow')

    with col2:
        st.subheader('Relation Between Humidity and Rain Tomorrow')
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax = sns.scatterplot(
            data=df,
            x='Humidity9am',
            y='Humidity3pm',
            hue='RainTomorrow',
            palette={'Yes': 'blue', 'No': 'lightblue'},
            edgecolor=None
        )
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        legend = ax.legend(title='Rain Tomorrow', fontsize='small', title_fontsize='small')
        legend.get_frame().set_alpha(0.5)
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        with st.expander('Insights'):
            st.write('When the humidity at 3 PM exceeds 60% and the humidity at 9 AM falls between 60% and 90%, there is an increased likelihood of rain the following day.')


with tab3:
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader('Total Rainfall by Location')
        rain_by_location = df[df['RainTomorrow'] == 'Yes'].groupby('Location').size().reset_index(name='RainFall')
        fig = px.bar(rain_by_location, x='Location', y='RainFall', title='Total Rainfall by Location', color_discrete_sequence=['Blue'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('insights'):
         st.write('Portland, located on the southern coast of Australia, is the most likely location to experience rainfall.')
    
    with col2:
        st.subheader('Rain Likelihood by Location')
        cross_tab = pd.crosstab(df['Location'], df['RainTomorrow']).reset_index()
        cross_tab_melted = cross_tab.melt(id_vars='Location', value_vars=['No', 'Yes'], var_name='RainTomorrow', value_name='Count')
        fig = px.bar(cross_tab_melted, x='Location', y='Count', color='RainTomorrow', title='Rain Tomorrow by Location', color_discrete_map={'No':'blue', 'Yes':'#FFD700'})
        fig.update_layout(barmode='stack', xaxis_title='Location', yaxis_title='Count',)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('insights'):
            st.write("""When the wind direction is west, rainfall is high, humidity levels are elevated, and there is significant cloud cover, these conditions lead to a high probability of rain tomorrow in Portland.""")

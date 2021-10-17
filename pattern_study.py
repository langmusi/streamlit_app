from os import replace
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Pattern Study in Dataset")

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

@st.cache
def load_csv(col):
    csv = pd.read_csv(uploaded_file, sep=";")
    csv.drop(columns = col, inplace = True)
    return csv

df = load_csv(col = 'ID')

st.write('The whole datasets:', df.shape)
st.write('Variables in the dataset', df.columns.values)


cat_list = list(df.columns[df.dtypes == object].tolist())
cat_list.append("City")
col_selected =  st.sidebar.selectbox("Group-variable Selection:", cat_list)

fig = sns.pairplot(df, hue=col_selected)
st.pyplot(fig)

cat_list_1 = df.columns.tolist()
col_selected_x =  st.sidebar.selectbox("Variable x Selection to plot further:", cat_list_1)
col_selected_y =  st.sidebar.selectbox("Variable y Selection to plot further:", cat_list_1)

# fig_1 = plt.figure()
# ax = fig_1.add_subplot(111)
# plt.scatter(df, x = col_selected_1, y = col_selected_2)
# st.pyplot(fig_1)

# plot the value
fig_1 = px.scatter(df,
                x=col_selected_x,
                y=col_selected_y)

st.plotly_chart(fig_1)
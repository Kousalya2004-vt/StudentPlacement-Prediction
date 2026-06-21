import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import pandas as pd

st.markdown("""
<style>
.stApp {
    background-color: #EDE7F6;
}
</style>
""", unsafe_allow_html=True)

data = pd.read_csv("Placement_data.csv")


data = pd.read_csv("Placement_data.csv")
data = data.dropna()

for c in data.columns:
    if data[c].dtype == "object":
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c])

x = data[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]]
y = data["status"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

st.title("Student Placement Prediction")

ssc_p = st.number_input("SSC Percentage", 0, 100)
hsc_p = st.number_input("HSC Percentage", 0, 100)
degree_p = st.number_input("Degree Percentage", 0, 100)
etest_p = st.number_input("E-Test Percentage", 0, 100)
mba_p = st.number_input("MBA Percentage", 0, 100)

if st.button("Predict"):

    avg = (ssc_p + hsc_p + degree_p + etest_p + mba_p) / 5

    if avg >= 60:
        st.success("Placed")
    else:
        st.error("Not Placed")

    chance = min(100, int(avg))

    st.write(f"Placement Chance: {chance}%")

chart = pd.DataFrame({
    "Status": ["Chance", "Remaining"],
    "Value": [chance, 100 - chance]
})

fig = px.pie(
    chart,
    names="Status",
    values="Value",
    title="Placement Chance"
)

st.plotly_chart(fig)
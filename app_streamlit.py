import streamlit as st
import pandas as pd
import joblib

model_tersimpan = joblib.load("model_mlbb.joblib")

st.title("MLBB Clasification")
kill = st.slider("Jumlah Kill",0,20)
assist = st.slider("Jumlah Assist",0,20)
death = st.slider("Jumlah Death",0,20)
turret = st.slider("Jumlah Turret",0,20)

if st.button("predict"):
	data_button = pd.DataFrame([[kill,assist,death,turret]],columns=["kill","assist","death","turret"])
	st.success(f"Hasil Prediksi : {model.predict([data_baru])[0]}")
	st.ballons()
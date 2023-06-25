#нужные библиотеки
import streamlit as st
from fastai.vision.all import *
import plotly.express as px
from PIL import Image
import pathlib


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.title("Добро пожаловать 👋 💯")
st.header('Это приложение распознавать :green[любые изображений] с помощи :red[искусственный интеллект]🤓 🦾')

#загружаем изображении

file_upload = st.file_uploader('Загрузите изображение  👉', type= ['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file_upload:
    st.image(file_upload)

    img = PILImage.create(file_upload)



result = st.button('Распознавать изображения  👈')
if result:
    model = load_learner('Sort_any_images_class.pkl')
    pred, pred_id, probs = model.predict(img)
    if probs [pred_id] * 100 > 85:
        st.success(f'Предсказания: 👍  {pred}')
        st.success(f'Вероятность: {probs [pred_id]*100/1}%')
    else:
        st.info('Вы загружали изображении с ошибкой!!!!  👎   Пожалуйста, попробуйте с другим изображением!!!')


     #plotly
    fig = px.bar(x=probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)
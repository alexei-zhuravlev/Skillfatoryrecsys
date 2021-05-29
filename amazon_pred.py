# В этом файле я попробую завернуть рекомендательную систему в вэб-приложение

######################
# Import libraries
######################

import pandas as pd
import streamlit as st
import nmslib
import pickle

######################
# Download data
######################
with open('embeddings.pickle', 'rb') as f:
    item_embeddings = pickle.load(f)

titles = pd.read_csv('titles_new.csv')

# Создадим граф из товаров
nms_idx = nmslib.init(method='hnsw', space='cosinesimil')

#Начинаем добавлять наши товары в граф
nms_idx.addDataPointBatch(item_embeddings)
nms_idx.createIndex(print_progress=True)

#Вспомогательная функция для поиска по графу
def nearest_item_nms(itemid, index, n=5):
    nn = index.knnQuery(item_embeddings[itemid], k=n)
    return nn

######################
# Page Title
######################

st.write("""
# Рекомендательная система для проекта Skillfactory
это макет рекомендательной системы для учебного проекта
***
""")


######################
# Input Text Box
######################

#st.sidebar.header('Enter DNA sequence')
st.header('Введите itemid')

itemid_input = "37138"

#sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
itemid_input = st.text_area(f"Введите itemid в диапазоне от 0 до {len(titles)}", itemid_input)
# sequence = sequence.splitlines()
# sequence = sequence[1:] # Skips the sequence name (first line)
# sequence = ''.join(sequence) # Concatenates list to string
itemid = int(itemid_input)

st.write("""
***
""")
if itemid <= len(titles) and itemid >= 0 :
    st.header ( 'Выбранный товар' )
    titles.title.loc[itemid]
    # Формируем список рекомендаций
    nbm = nearest_item_nms (itemid, nms_idx )[0]
    st.subheader ( 'С этим товаром обычно покупают' )

    st.write ( '1  ' + titles.title.loc[nbm[0]] )
    st.write ( '2  ' + titles.title.loc[nbm[1]] )
    st.write ( '3  ' + titles.title.loc[nbm[2]] )
    st.write ( '4  ' + titles.title.loc[nbm[3]] )
    st.write ( '5  ' + titles.title.loc[nbm[4]] )
else:
    st.header('Выбранный товар')
    st.write ( 'itemid должен быть в диапзоне от 0 до ' + str(len(titles)) )


st.write("""
***
""")

# # Формируем список рекомендаций
# nbm = nearest_item_nms(itemid,nms_idx)[0]
#
# ## DNA nucleotide count
# st.subheader('С этим товаром обычно покупают')
#
# st.write('1  ' + titles.title.loc[nbm[0]])
# st.write('2  ' + titles.title.loc[nbm[1]])
# st.write('3  ' + titles.title.loc[nbm[2]])
# st.write('4  ' + titles.title.loc[nbm[3]])
# st.write('5  ' + titles.title.loc[nbm[4]])


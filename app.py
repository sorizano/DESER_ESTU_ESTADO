# Importacion de librerias
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pd.DataFrame.iteritems = pd.DataFrame.items
scaler = StandardScaler()


# Títulos y carga de archivo
st.title("Visualización y Clusterización automática de Data de Estudiantes")
st.write("Cargue el archivo PKL para visualizar el análisis de su contenido.")
uploaded_file = st.file_uploader("Cargar archivo: ", type='pkl')

if uploaded_file is not None:
    
    df = pd.read_pickle(uploaded_file)
    
    # Eliminación de datos inválidos
    df_050 = df.dropna(axis=0) 
    df_050.index = df_050.DNI

    # Seleccion de categorias
    st.write(df_050.shape)
    MAX_CAT = st.slider('Maximo numero de categorias: ', 10, 30, 30)
    
    # Depuración de columnas sólo para aquellas que contribuyen al clustering
    col_selec = []
    for col in df_050.columns:
        u_col = df_050[col].unique()
        if len(u_col) < MAX_CAT:
            col_selec.append(col)
            
    st.header('Lista de variables que será usada para la clusterización')
    st.write(' '.join(col_selec))
    
    # Conversion a dummies
    df_100 = df_050[col_selec]
    df_110 = pd.get_dummies(df_100)
    
    # Calcular línea base
    df_linbase = df_110.mean() / df_110.max() * 100
    df_linbase2 = pd.DataFrame(df_linbase)
    df_linbase2['col_cats'] = df_linbase2.index
    df_linbase2.columns=['valor', 'col_cats']
    csv_10 = df_linbase2.to_csv(encoding='iso-8859-1')

    st.download_button(
        label="Descargar CSV",
        data=csv_10,
        file_name='línea_base.csv',
        mime='text/csv'
    )


    st.header('Matriz de correlación de todas las categorías')
    
    corr_matrix = df_110.corr()
    plt.figure(figsize=(21, 21))  # Adjust the figure size as needed
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={'size': 5})
    plt.title('Mapa de Calor de la Correlation de Variables')
    st.pyplot(plt)

    st.header('Clustering usando PCA')
    
    X_sc = scaler.fit_transform(df_110)
    st.write('La forma de la data es: ', X_sc.shape)
    has_nan = np.isnan(X_sc).sum()
    pca = PCA(n_components=2)
    pca.fit(X_sc)
    X_pca = pca.transform(X_sc)
    data_200 = df_100
    data_200['pca_1'] = X_pca[:, 0]
    data_200['pca_2'] = X_pca[:, 1]
    
    st.write(data_200.columns)
    
    #st.write(data_200['COD_DEPARTAMENTO'].unique())
    #st.write(data_200['ESTADO_ESTUDIANTE'].unique())
    #VIRTU = st.selectbox('Virtual: ', ['UVIR', 'PCGT'])
    
    # Diagramacion de Scatter con resultado PCA
    INGRE = st.selectbox('Estado: ', ['Abandono', 'Activo'])
    data_210 = data_200[data_200['ESTADO_INGRESANTE']==INGRE]
    fig2 = px.scatter(data_210, x='pca_1', y='pca_2', title='Distribución PCA', width=800, height=800)
    st.plotly_chart(fig2)

    st.header('Diagrama de densidades')
    GRIDSIZEX = st.slider('Seleccione la densidad de la grilla de hexágonos: ', 0, 100, 35)
    
    plt.figure(figsize=(10, 8))
    plt_extracto = plt.hexbin(data_210.pca_1, data_210.pca_2, gridsize=GRIDSIZEX, cmap='inferno')
    plt.colorbar()
    plt.title('Hexbin Plot of PCA-Transformed Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)

    st.header('Histograma de Densidades')
    
    plt.figure(figsize=(7, 4))
    densidades = pd.DataFrame(plt_extracto.get_array())
    densidades.hist(bins=50, log=True)
    plt.ylabel('Cantidad de Ocurrencias')
    plt.xlabel('Densidad Estudiantes por Area')
    
    plt.title('Histograma de Densidades')
    st.pyplot(plt)
    
    offsets = plt_extracto.get_offsets()
    offsets_df = pd.DataFrame(offsets)
    st.write(offsets_df.shape)
    offsets_df['densidad'] = densidades[0]
    offsets_df.columns = ['col1', 'col2', 'densidad']
    offset_selec = offsets_df.sort_values(by='densidad', ascending=False)
    patrones_df = pd.DataFrame(index = [0,1,2,3,4,5,6,7,8,9], data=offset_selec.values[0:10], columns=offset_selec.columns)

    st.header('Tabla de Densidades')

    st.write(patrones_df)
    
    NUM_CASOS = st.slider("¿Qué rango de valores elige explorar?", 1, 10, value=(3,7))
    st.write('Usted ha elegido ', NUM_CASOS, 'casos.')
    CASES_LIST = [i for i in range(NUM_CASOS[0], NUM_CASOS[1] + 1)]
    
    radiohex = (data_210.pca_1.max() - data_210.pca_1.min())/GRIDSIZEX/2

    st.header('Visualización de Caso Particular')
    
    CASOX = st.selectbox('Elija el caso: ', CASES_LIST)
    
    a, b = patrones_df.col1[CASOX], patrones_df.col2[CASOX]
    enfoqueX = data_210[
        (data_210.pca_1 > a - radiohex) & 
        (data_210.pca_1 < a + radiohex) & 
        (data_210.pca_2 > b - radiohex) & 
        (data_210.pca_2 < b + radiohex)
    ]
    
    st.write(enfoqueX.shape)

    st.subheader('Diagrama de Coordenadas Paralelas')
    
    LISTA_SELEC = st.multiselect('Escoja la variable de color: ', list(enfoqueX.columns))
    st.write(LISTA_SELEC)
        
    fig2 = px.parallel_categories(data_frame=enfoqueX[list(LISTA_SELEC)])
    st.plotly_chart(fig2)

    st.subheader('Poblaciones por Hexágonos Elegidos')
    
    for c in CASES_LIST:
        a, b = patrones_df.col1[c], patrones_df.col2[c]
        enfoqueX = data_210[
            (data_210.pca_1 > a - radiohex) & 
            (data_210.pca_1 < a + radiohex) & 
            (data_210.pca_2 > b - radiohex) & 
            (data_210.pca_2 < b + radiohex)  
            ]
        st.write(f'Tamaño {c}', len(enfoqueX))
    
    st.header('Descarga de Items de Hexagonos Densos Elegidos')

    enfoques = pd.DataFrame()
    for c in CASES_LIST:
        a, b = patrones_df.col1[c], patrones_df.col2[c]
        enfoqueX = data_210[
            (data_210.pca_1 > a - radiohex) & 
            (data_210.pca_1 < a + radiohex) & 
            (data_210.pca_2 > b - radiohex) & 
            (data_210.pca_2 < b + radiohex)
        ]
        enfoqueX['HexDens'] = 'Hex_'+str(c)
        enfoques = pd.concat([enfoques, enfoqueX])
    
    st.write(enfoques.columns)
    
    enfoques2 = enfoques.drop(columns=['pca_1', 'pca_2', 'HexDens', 'ESTADO_INGRESANTE'])
    csv = enfoques2.to_csv(encoding='iso-8859-1')

    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name='hexagonos_densos.csv',
        mime='text/csv'
    )
    
    df = enfoques2
    cat_col = df.select_dtypes(include=['object']).columns.tolist()
    df_dummies = pd.get_dummies(df[cat_col])
    percentage_presence = df_dummies.mean()*100
    dfx = df.drop(cat_col, axis=1)
    mean_values = dfx.mean()/dfx.max()*100
    result = pd.concat([percentage_presence, mean_values])
    df2 = pd.DataFrame()
    df2['valor_hex'] = result
    df2['col_cats'] = result.index
    df2 = df2.sort_values(by='valor_hex', ascending=False)
    df3 = df2.head(25)

    st.subheader('Radar Porcentajes Categorias')
    
    fig3 = px.line_polar(df3, r='valor_hex', theta='col_cats')
    st.plotly_chart(fig3)

    df_result = pd.merge(df3, df_linbase2, on='col_cats', how='left')
    df_result = df_result[['col_cats', 'valor_hex', 'valor']]
    df_result['diff_linbase'] = df_result.valor_hex - df_result.valor

    st.subheader('Radar Diferencia con Linea Base')
    fig4 = px.line_polar(df_result, r='diff_linbase', theta='col_cats')
    st.plotly_chart(fig4)
    
    csv2 = df_result.to_csv(encoding='iso-8859-1')

    st.download_button(
        label="Descargar CSV",
        data=csv2,
        file_name='frecuencias_experimento.csv',
        mime='text/csv'
    )
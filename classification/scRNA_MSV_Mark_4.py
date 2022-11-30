# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:21:35 2022

@author: IngBi
"""

# =============================================================================
# Librerias utilizadas
# =============================================================================
import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
#from umap import UMAP
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from collections import Counter
import numpy as np
import scanpy as sc
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import *
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# =============================================================================
# Extracción de características (Intersección de genes AML & FA)
# Num. de características: 22312
# Este proceso se tiene que utilizar de nuevo, si es que agregas un individuo
# de otra base diferente. 
# =============================================================================
# =============================================================================

url = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'
path_anno = url + '\patient1.csv'
df_FA = pd.read_csv(path_anno, sep='\t')

df_FA.shape
df_FA = df_FA.rename({'Unnamed: 0': 'Gene'}, axis=1)
gene_names_FA = df_FA['Gene']
#df_FA.head()

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_419 = pd.read_csv(ge_files +'\GSM3587950_AML419A-D0.dem.txt.gz',
                 sep='\t')

# Lista de características utilizadas (Genes)
df_inner_join = pd.merge(left=df_FA,right=df_419, left_on='Gene', right_on='Gene')

gene_names_inner = df_inner_join['Gene']

columns_names = df_inner_join.columns.values
columns_names_list = list(columns_names)
columns_names_list = list(columns_names[1:,])

m_col_inner = [i for i,x in enumerate(df_inner_join.columns)]
df_inner_join = df_inner_join.iloc[:, m_col_inner]
df_inner_join = df_inner_join.drop(['Gene'], axis=1)
df_inner_join.shape

barcodes_inner = [x for x in df_inner_join.columns if 'Gene' not in x]

anndata_inner = sc.AnnData(X=df_inner_join.T.to_numpy(), obs=barcodes_inner, var=gene_names_inner.values)

anndata_inner.var_names = gene_names_inner.values

anndata_inner.obs_names = barcodes_inner

anndata_inner.raw = anndata_inner
sc.pl.highest_expr_genes(anndata_inner, n_top=20, )

anndata_inner.shape

# =============================================================================
# Lectura de archivosde anotaciones AML
# Utilizar la ruta de tu computadora, donde tienes los archivos de anotación
# =============================================================================
dir_anno_val = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

anno210Ad0 = pd.read_csv(dir_anno_val + '\GSM3587926_AML210A-D0.anno.txt.gz',
                         sep='\t')

anno328d0 = pd.read_csv(dir_anno_val + '\GSM3587932_AML328-D0.anno.txt.gz',
                         sep='\t')

anno329d0 = pd.read_csv(dir_anno_val + '\GSM3587941_AML329-D0.anno.txt.gz',
                         sep='\t')

anno371d0 = pd.read_csv(dir_anno_val + '\GSM3587947_AML371-D0.anno.txt.gz',
                         sep='\t')

anno419d0 = pd.read_csv(dir_anno_val + '\GSM3587951_AML419A-D0.anno.txt.gz',
                         sep='\t')

anno420Bd0 = pd.read_csv(dir_anno_val + '\GSM3587954_AML420B-D0.anno.txt.gz',
                         sep='\t')

anno475d0 = pd.read_csv(dir_anno_val + '\GSM3587960_AML475-D0.anno.txt.gz',
                         sep='\t')

anno556d0 = pd.read_csv(dir_anno_val + '\GSM3587964_AML556-D0.anno.txt.gz',
                         sep='\t')

anno707Bd0 = pd.read_csv(dir_anno_val + '\GSM3587970_AML707B-D0.anno.txt.gz',
                         sep='\t')

#anno722Bd0 = pd.read_csv(dir_anno_val + '\GSM3587981_AML722B-D0.anno.txt.gz',
                         #sep='\t')

anno870d0 = pd.read_csv(dir_anno_val + '\GSM3587985_AML870-D0.anno.txt.gz',
                         sep='\t')

anno916d0 = pd.read_csv(dir_anno_val + '\GSM3587989_AML916-D0.anno.txt.gz',
                         sep='\t')

anno921Ad0 = pd.read_csv(dir_anno_val + '\GSM3587991_AML921A-D0.anno.txt.gz',
                         sep='\t')

anno1012d0 = pd.read_csv(dir_anno_val + '\GSM3587924_AML1012-D0.anno.txt.gz',
                         sep='\t')

annoBM1 = pd.read_csv(dir_anno_val + '\GSM3587996_BM1.anno.txt.gz', 
                        sep='\t')

annoBM2 = pd.read_csv(dir_anno_val + '\GSM3587997_BM2.anno.txt.gz', 
                        sep='\t')

annoBM3 = pd.read_csv(dir_anno_val + '\GSM3587999_BM3.anno.txt.gz', 
                        sep='\t')

annoBM4 = pd.read_csv(dir_anno_val + '\GSM3588001_BM4.anno.txt.gz', 
                        sep='\t')

annoBM5 = pd.read_csv(dir_anno_val + '\GSM3588002_BM5-34p.anno.txt.gz', 
                        sep='\t')

annohealthy1 = pd.read_csv(dir_anno_val + '\\annohealthy_1.txt.gz', 
                        sep='\t')  

annohealthy2 = pd.read_csv(dir_anno_val + '\\annohealthy_2.txt.gz', 
                        sep='\t')      

annohealthy3 = pd.read_csv(dir_anno_val + '\\annohealthy_3.txt.gz', 
                        sep='\t')

annopatient1 = pd.read_csv(dir_anno_val + '\\annopatient1.txt.gz', 
                        sep='\t')    
  
annopatient2 = pd.read_csv(dir_anno_val + '\\annopatient2.txt.gz', 
                        sep='\t')  

annopatient3 = pd.read_csv(dir_anno_val + '\\annopatient3.txt.gz', 
                        sep='\t')

annopatient4 = pd.read_csv(dir_anno_val + '\\annopatient4.txt.gz', 
                        sep='\t')  

#annopatient5 = pd.read_csv(dir_anno_val + '\\annopatient5.txt.gz', 
                        #sep='\t')  

annopatient6 = pd.read_csv(dir_anno_val + '\\annopatient6.txt.gz', 
                        sep='\t')    

# =============================================================================
# Información de las celulas malignas y benignas de los pacientes
# Esta parte del código no es forzosa, se utiliza si quierer corroborar el número
# de células con las que estas entrenado y probando. 
# =============================================================================

anno210Ad0.columns
print(Counter(anno210Ad0.PredictionRefined))
print(Counter(anno210Ad0.CellType))

anno328d0.columns
print(Counter(anno328d0.PredictionRefined))
print(Counter(anno328d0.CellType))

anno329d0.columns
print(Counter(anno329d0.PredictionRefined))
print(Counter(anno329d0.CellType))

anno371d0.columns
print(Counter(anno371d0.PredictionRF2))
print(Counter(anno371d0.CellType))

anno419d0.columns
print(Counter(anno419d0.PredictionRefined))
print(Counter(anno419d0.CellType))

anno420Bd0.columns
print(Counter(anno420Bd0.PredictionRefined))
print(Counter(anno420Bd0.CellType))

anno475d0.columns
print(Counter(anno475d0.PredictionRefined))
print(Counter(anno475d0.CellType))

anno556d0.columns
print(Counter(anno556d0.PredictionRefined))
print(Counter(anno556d0.CellType))

anno707Bd0.columns
print(Counter(anno707Bd0.PredictionRefined))
print(Counter(anno707Bd0.CellType))

anno870d0.columns
print(Counter(anno870d0.PredictionRefined))
print(Counter(anno870d0.CellType))

anno916d0.columns
print(Counter(anno916d0.PredictionRefined))
print(Counter(anno916d0.CellType))

anno921Ad0.columns
print(Counter(anno921Ad0.PredictionRefined))
print(Counter(anno921Ad0.CellType))

anno1012d0.columns
print(Counter(anno1012d0.PredictionRefined))
print(Counter(anno1012d0.CellType))

annoBM1.columns
print(Counter(annoBM1.PredictionRefined))
print(Counter(annoBM1.CellType))

annoBM2.columns
print(Counter(annoBM2.PredictionRefined))
print(Counter(annoBM2.CellType))

annoBM3.columns
print(Counter(annoBM3.PredictionRefined))
print(Counter(annoBM3.CellType))

annoBM4.columns
print(Counter(annoBM4.PredictionRefined))
print(Counter(annoBM4.CellType))

annoBM5.columns
print(Counter(annoBM5.PredictionRefined))
print(Counter(annoBM5.CellType))

annohealthy1.columns
print(Counter(annohealthy1.PredictionRefined))
#print(Counter(annohealthy1.CellType))

annohealthy2.columns
print(Counter(annohealthy2.PredictionRefined))
#print(Counter(annohealthy1.CellType))

annohealthy3.columns
print(Counter(annohealthy3.PredictionRefined))
#print(Counter(annohealthy1.CellType))

annopatient1.columns
print(Counter(annopatient1.PredictionRefined))
#print(Counter(annopatient2.CellType))

annopatient2.columns
print(Counter(annopatient2.PredictionRefined))
#print(Counter(annopatient2.CellType))

annopatient3.columns
print(Counter(annopatient3.PredictionRefined))
#print(Counter(annopatient2.CellType))

annopatient4.columns
print(Counter(annopatient4.PredictionRefined))
#print(Counter(annopatient2.CellType))

#annopatient5.columns
#print(Counter(annopatient5.PredictionRefined))
#print(Counter(annopatient2.CellType))

annopatient6.columns
print(Counter(annopatient6.PredictionRefined))
#print(Counter(annopatient2.CellType))

# =============================================================================
# Revisar nuestros datos
# =============================================================================
#anno1012d0['PredictionRefined'].value_counts().tolist()
#anno1012d0['PredictionRefined'].value_counts().keys().tolist()

norm_210A = anno210Ad0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_210A = anno210Ad0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_328 = anno328d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_328 = anno328d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_329 = anno329d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_329 = anno329d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_371 = anno371d0['PredictionRF2'].str.contains('normal').value_counts()[True]
malig_371 = anno371d0['PredictionRF2'].str.contains('malignant').value_counts()[True]

norm_419 = anno419d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_419 = anno419d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_420 = anno420Bd0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_420 = anno420Bd0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_475 = anno475d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_475 = anno475d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_556 = anno556d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_556 = anno556d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_707 = anno707Bd0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_707 = anno707Bd0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_870 = anno870d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_870 = anno870d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_916 = anno916d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_916 = anno916d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_921 = anno921Ad0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_921 = anno921Ad0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_1012 = anno1012d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_1012 = anno1012d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

BM1_norm = annoBM1['PredictionRefined'].str.contains('normal').value_counts()[True]

BM2_norm = annoBM2['PredictionRefined'].str.contains('normal').value_counts()[True]

BM3_norm = annoBM3['PredictionRefined'].str.contains('normal').value_counts()[True]

BM4_norm = annoBM4['PredictionRefined'].str.contains('normal').value_counts()[True]

BM5_norm = annoBM5['PredictionRefined'].str.contains('normal').value_counts()[True]

healthy1_norm = annohealthy1['PredictionRefined'].str.contains('normal').value_counts()[True]

healthy2_norm = annohealthy2['PredictionRefined'].str.contains('normal').value_counts()[True]

healthy3_norm = annohealthy3['PredictionRefined'].str.contains('normal').value_counts()[True]

patient1 = annopatient1['PredictionRefined'].str.contains('malignant').value_counts()[True]

patient2 = annopatient2['PredictionRefined'].str.contains('malignant').value_counts()[True]

patient3 = annopatient3['PredictionRefined'].str.contains('malignant').value_counts()[True]

patient4 = annopatient4['PredictionRefined'].str.contains('malignant').value_counts()[True]

#patient5 = annopatient5['PredictionRefined'].str.contains('malignant').value_counts()[True]

patient6 = annopatient6['PredictionRefined'].str.contains('malignant').value_counts()[True]

total_cell_benign = norm_210A + norm_328 + norm_329 + norm_371 + norm_419 + norm_420 + norm_475 + norm_556 + norm_707 + norm_870 + norm_916 + norm_921 + norm_1012 + BM1_norm + BM2_norm + BM3_norm + BM4_norm + BM5_norm + healthy1_norm + healthy2_norm + healthy3_norm
total_cell_malig = malig_210A + malig_328 + malig_329 + malig_371 + malig_419 + malig_420 + malig_475 + malig_556 + malig_707 + malig_870 + malig_916 + malig_921 + malig_1012 + patient1 + patient2 + patient3 + patient4 + patient6

# =============================================================================
# Myeloid cells
# =============================================================================

# Myeloid cells
#mcells = ['HSC', 'Prog', 'GMP', 'Promono', 'Mono', 'cDC', 'pDC'] # Revisar si agregamos los otros grupos
#Seleccionar sólo células mieloides (normales o tumorales)

ind_myeloid_malign_210A = [i for i, x in enumerate(anno210Ad0.CellType) if 'like' in x]
ind_myeloid_benign_210A = [i for i, x in enumerate(anno210Ad0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_328 = [i for i, x in enumerate(anno328d0.CellType) if 'like' in x]
ind_myeloid_benign_328 = [i for i, x in enumerate(anno328d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_329 = [i for i, x in enumerate(anno329d0.CellType) if 'like' in x]
ind_myeloid_benign_329 = [i for i, x in enumerate(anno329d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_371 = [i for i, x in enumerate(anno371d0.PredictionRF2) if 'malignant' in x]
ind_myeloid_benign_371 = [i for i, x in enumerate(anno371d0.PredictionRF2) if 'normal' in x]

ind_myeloid_malign_419 = [i for i, x in enumerate(anno419d0.CellType) if 'like' in x]
ind_myeloid_benign_419 = [i for i, x in enumerate(anno419d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_420B = [i for i, x in enumerate(anno420Bd0.CellType) if 'like' in x]
ind_myeloid_benign_420B = [i for i, x in enumerate(anno420Bd0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_475 = [i for i, x in enumerate(anno475d0.CellType) if 'like' in x]
ind_myeloid_benign_475 = [i for i, x in enumerate(anno475d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_556 = [i for i, x in enumerate(anno556d0.PredictionRF2) if 'malignant' in x]
ind_myeloid_benign_556 = [i for i, x in enumerate(anno556d0.PredictionRF2) if 'normal' in x]

ind_myeloid_malign_707B = [i for i, x in enumerate(anno707Bd0.CellType) if 'like' in x]
ind_myeloid_benign_707B = [i for i, x in enumerate(anno707Bd0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_870 = [i for i, x in enumerate(anno870d0.CellType) if 'like' in x]
ind_myeloid_benign_870 = [i for i, x in enumerate(anno870d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_916 = [i for i, x in enumerate(anno916d0.CellType) if 'like' in x]
ind_myeloid_benign_916 = [i for i, x in enumerate(anno916d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_921A = [i for i, x in enumerate(anno921Ad0.CellType) if 'like' in x]
ind_myeloid_benign_921A = [i for i, x in enumerate(anno921Ad0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_1012 = [i for i, x in enumerate(anno1012d0.CellType) if 'like' in x]
ind_myeloid_benign_1012 = [i for i, x in enumerate(anno1012d0.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_BM1 = [i for i, x in enumerate(annoBM1.CellType) if 'like' in x]
ind_myeloid_benign_BM1 = [i for i, x in enumerate(annoBM1.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_BM2 = [i for i, x in enumerate(annoBM2.CellType) if 'like' in x]
ind_myeloid_benign_BM2 = [i for i, x in enumerate(annoBM2.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_BM3 = [i for i, x in enumerate(annoBM3.CellType) if 'like' in x]
ind_myeloid_benign_BM3 = [i for i, x in enumerate(annoBM3.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_BM4 = [i for i, x in enumerate(annoBM4.CellType) if 'like' in x]
ind_myeloid_benign_BM4 = [i for i, x in enumerate(annoBM4.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_BM5 = [i for i, x in enumerate(annoBM5.CellType) if 'like' in x]
ind_myeloid_benign_BM5 = [i for i, x in enumerate(annoBM5.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_healthy1 = [i for i, x in enumerate(annohealthy1.CellType) if 'like' in x]
ind_myeloid_benign_healthy1 = [i for i, x in enumerate(annohealthy1.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_healthy2 = [i for i, x in enumerate(annohealthy2.CellType) if 'like' in x]
ind_myeloid_benign_healthy2 = [i for i, x in enumerate(annohealthy2.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_healthy3 = [i for i, x in enumerate(annohealthy3.CellType) if 'like' in x]
ind_myeloid_benign_healthy3 = [i for i, x in enumerate(annohealthy3.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_patient1 = [i for i, x in enumerate(annopatient1.PredictionRefined) if 'malignant' in x]
ind_myeloid_benign_patient1 = [i for i, x in enumerate(annopatient1.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_patient2 = [i for i, x in enumerate(annopatient2.PredictionRefined) if 'malignant' in x]
ind_myeloid_benign_patient2 = [i for i, x in enumerate(annopatient2.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_patient3 = [i for i, x in enumerate(annopatient3.PredictionRefined) if 'malignant' in x]
ind_myeloid_benign_patient3 = [i for i, x in enumerate(annopatient3.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_patient4 = [i for i, x in enumerate(annopatient4.PredictionRefined) if 'malignant' in x]
ind_myeloid_benign_patient4 = [i for i, x in enumerate(annopatient4.PredictionRefined) if 'normal' in x]

#ind_myeloid_malign_patient5 = [i for i, x in enumerate(annopatient5.PredictionRefined) if 'malignant' in x]
#ind_myeloid_benign_patient5 = [i for i, x in enumerate(annopatient5.PredictionRefined) if 'normal' in x]

ind_myeloid_malign_patient6 = [i for i, x in enumerate(annopatient6.PredictionRefined) if 'malignant' in x]
ind_myeloid_benign_patient6 = [i for i, x in enumerate(annopatient6.PredictionRefined) if 'normal' in x]


#total_cell_malig_new = len(ind_myeloid_malign_419) + len(ind_myeloid_malign_420B) + len(ind_myeloid_malign_707B) + len(ind_myeloid_malign_916) + len(ind_myeloid_malign_1012) + len(ind_myeloid_malign_BM1) + len(ind_myeloid_malign_BM2) + len(ind_myeloid_malign_BM4) + len(ind_myeloid_malign_BM5) + len(ind_myeloid_malign_FA)
#total_cell_benign_new = len(ind_myeloid_benign_419) + len(ind_myeloid_benign_420B) + len(ind_myeloid_benign_707B) + len(ind_myeloid_benign_916) + len(ind_myeloid_benign_1012) + len(ind_myeloid_benign_BM1) + len(ind_myeloid_benign_BM2) + len(ind_myeloid_benign_BM3) + len(ind_myeloid_benign_BM4) + len(ind_myeloid_benign_BM5) + len(ind_myeloid_benign_healthy1)

# =============================================================================
# =============================================================================
# # Cargar el archivo de conteo de los sujetos muestra
# =============================================================================
# =============================================================================

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_210A = pd.read_csv(ge_files +'\GSM3587925_AML210A-D0.dem.txt.gz',
                 sep='\t')

df_328 = pd.read_csv(ge_files +'\GSM3587931_AML328-D0.dem.txt.gz',
                 sep='\t')

df_329 = pd.read_csv(ge_files +'\GSM3587940_AML329-D0.dem.txt.gz',
                 sep='\t')

df_371 = pd.read_csv(ge_files +'\GSM3587946_AML371-D0.dem.txt.gz',
                 sep='\t')

df_419 = pd.read_csv(ge_files +'\GSM3587950_AML419A-D0.dem.txt.gz',
                 sep='\t')

df_420B = pd.read_csv(ge_files +'\GSM3587953_AML420B-D0.dem.txt.gz',
                 sep='\t')

df_475 = pd.read_csv(ge_files +'\GSM3587959_AML475-D0.dem.txt.gz',
                 sep='\t')

df_556 = pd.read_csv(ge_files +'\GSM3587963_AML556-D0.dem.txt.gz',
                 sep='\t')

df_707B = pd.read_csv(ge_files +'\GSM3587969_AML707B-D0.dem.txt.gz',
                 sep='\t')

df_870 = pd.read_csv(ge_files +'\GSM3587984_AML870-D0.dem.txt.gz',
                 sep='\t')

df_916 = pd.read_csv(ge_files +'\GSM3587988_AML916-D0.dem.txt.gz',
                 sep='\t')

df_921A = pd.read_csv(ge_files +'\GSM3587990_AML921A-D0.dem.txt.gz',
                 sep='\t')

df_1012 = pd.read_csv(ge_files +'\GSM3587923_AML1012-D0.dem.txt.gz',
                 sep='\t')

df_BM1 = pd.read_csv(ge_files +'\GSM3587996_BM1.dem.txt.gz',
                 sep='\t')

df_BM2 = pd.read_csv(ge_files +'\GSM3587997_BM2.dem.txt.gz',
                 sep='\t')

df_BM3 = pd.read_csv(ge_files +'\GSM3587998_BM3.dem.txt.gz',
                 sep='\t')

df_BM4 = pd.read_csv(ge_files +'\GSM3588000_BM4.dem.txt.gz',
                 sep='\t')

df_BM5 = pd.read_csv(ge_files +'\GSM3588002_BM5-34p.dem.txt.gz',
                 sep='\t')

df_healthy1 = pd.read_csv(ge_files +'\healthy1.csv',
                 sep='\t')

df_healthy2 = pd.read_csv(ge_files +'\healthy2.csv',
                 sep='\t')

df_healthy3 = pd.read_csv(ge_files +'\healthy3.csv',
                 sep='\t')

df_patient1 = pd.read_csv(ge_files +'\patient1.csv',
                 sep='\t')

df_patient2 = pd.read_csv(ge_files +'\patient2.csv',
                 sep='\t')

df_patient3 = pd.read_csv(ge_files +'\patient3.csv',
                 sep='\t')

df_patient4= pd.read_csv(ge_files +'\patient4.csv',
                 sep='\t')

#df_patient5 = pd.read_csv(ge_files +'\patient5.csv',
                 #sep='\t')

df_patient6 = pd.read_csv(ge_files +'\patient7.csv',
                 sep='\t')

# =============================================================================
# Ajustar a cada individuo a las características obtenidas en la intersección
# =============================================================================

###  Indices de los genes
iGenes_210A = [i for i, x in enumerate(df_210A['Gene']) if x in anndata_inner.var_names]
iGenes_328 = [i for i, x in enumerate(df_328['Gene']) if x in anndata_inner.var_names]
iGenes_329 = [i for i, x in enumerate(df_329['Gene']) if x in anndata_inner.var_names]
iGenes_371 = [i for i, x in enumerate(df_371['Gene']) if x in anndata_inner.var_names]
iGenes_419 = [i for i, x in enumerate(df_419['Gene']) if x in anndata_inner.var_names]
iGenes_420B = [i for i, x in enumerate(df_420B['Gene']) if x in anndata_inner.var_names]
iGenes_475 = [i for i, x in enumerate(df_475['Gene']) if x in anndata_inner.var_names]
iGenes_556 = [i for i, x in enumerate(df_556['Gene']) if x in anndata_inner.var_names]
iGenes_707B = [i for i, x in enumerate(df_707B['Gene']) if x in anndata_inner.var_names]
iGenes_870 = [i for i, x in enumerate(df_870['Gene']) if x in anndata_inner.var_names]
iGenes_916 = [i for i, x in enumerate(df_916['Gene']) if x in anndata_inner.var_names]
iGenes_921A = [i for i, x in enumerate(df_921A['Gene']) if x in anndata_inner.var_names]
iGenes_1012 = [i for i, x in enumerate(df_1012['Gene']) if x in anndata_inner.var_names]
iGenes_BM1 = [i for i, x in enumerate(df_BM1['Gene']) if x in anndata_inner.var_names]
iGenes_BM2 = [i for i, x in enumerate(df_BM2['Gene']) if x in anndata_inner.var_names]
iGenes_BM3 = [i for i, x in enumerate(df_BM3['Gene']) if x in anndata_inner.var_names]
iGenes_BM4 = [i for i, x in enumerate(df_BM4['Gene']) if x in anndata_inner.var_names]
iGenes_BM5 = [i for i, x in enumerate(df_BM5['Gene']) if x in anndata_inner.var_names]
iGenes_healthy1 = [i for i, x in enumerate(df_healthy1['Gene']) if x in anndata_inner.var_names]
iGenes_healthy2 = [i for i, x in enumerate(df_healthy2['Gene']) if x in anndata_inner.var_names]
iGenes_healthy3 = [i for i, x in enumerate(df_healthy3['Gene']) if x in anndata_inner.var_names]
iGenes_patient1 = [i for i, x in enumerate(df_patient1['Gene']) if x in anndata_inner.var_names]
iGenes_patient2 = [i for i, x in enumerate(df_patient2['Gene']) if x in anndata_inner.var_names]
iGenes_patient3 = [i for i, x in enumerate(df_patient3['Gene']) if x in anndata_inner.var_names]
iGenes_patient4 = [i for i, x in enumerate(df_patient4['Gene']) if x in anndata_inner.var_names]
#iGenes_patient5 = [i for i, x in enumerate(df_patient5['Gene']) if x in anndata_inner.var_names]
iGenes_patient6 = [i for i, x in enumerate(df_patient6['Gene']) if x in anndata_inner.var_names]



# =============================================================================
# Enlistar el numero de genes de cada sujeto muestra
# =============================================================================

### Nombres de los genes
gene_names_210A = df_210A['Gene']
gene_names_328 = df_328['Gene']
gene_names_329 = df_329['Gene']
gene_names_371 = df_371['Gene']
gene_names_419 = df_419['Gene']
gene_names_420B = df_420B['Gene']
gene_names_475 = df_475['Gene']
gene_names_556 = df_556['Gene']
gene_names_707B = df_707B['Gene']
gene_names_870 = df_870['Gene']
gene_names_916 = df_916['Gene']
gene_names_921A = df_921A['Gene']
gene_names_1012 = df_1012['Gene']
gene_names_BM1 = df_BM1['Gene']
gene_names_BM2 = df_BM2['Gene']
gene_names_BM3 = df_BM3['Gene']
gene_names_BM4 = df_BM4['Gene']
gene_names_BM5 = df_BM5['Gene']
gene_names_healthy1 = df_healthy1['Gene']
gene_names_healthy2 = df_healthy2['Gene']
gene_names_healthy3 = df_healthy3['Gene']
gene_names_patient1 = df_patient1['Gene']
gene_names_patient1 = df_patient2['Gene']
gene_names_patient1 = df_patient3['Gene']
gene_names_patient1 = df_patient4['Gene']
#gene_names_patient1 = df_patient5['Gene']
gene_names_patient1 = df_patient6['Gene']


# =============================================================================
# Seleccion de celulas malignas y benignas
# =============================================================================
barcodes_myeloid_210A = list(anno210Ad0.Cell[ind_myeloid_malign_210A]) + list(anno210Ad0.Cell[ind_myeloid_benign_210A])
barcodes_myeloid_328 = list(anno328d0.Cell[ind_myeloid_malign_328]) + list(anno328d0.Cell[ind_myeloid_benign_328])
barcodes_myeloid_329 = list(anno329d0.Cell[ind_myeloid_malign_329]) + list(anno329d0.Cell[ind_myeloid_benign_329])
barcodes_myeloid_371 = list(anno371d0.Cell[ind_myeloid_malign_371]) + list(anno371d0.Cell[ind_myeloid_benign_371])
barcodes_myeloid_419 = list(anno419d0.Cell[ind_myeloid_malign_419]) + list(anno419d0.Cell[ind_myeloid_benign_419])
barcodes_myeloid_420B = list(anno420Bd0.Cell[ind_myeloid_malign_420B]) + list(anno420Bd0.Cell[ind_myeloid_benign_420B])
barcodes_myeloid_475 = list(anno475d0.Cell[ind_myeloid_malign_475]) + list(anno475d0.Cell[ind_myeloid_benign_475])
barcodes_myeloid_556 = list(anno556d0.Cell[ind_myeloid_malign_556]) + list(anno556d0.Cell[ind_myeloid_benign_556])
barcodes_myeloid_707B = list(anno707Bd0.Cell[ind_myeloid_malign_707B]) + list(anno707Bd0.Cell[ind_myeloid_benign_707B])
barcodes_myeloid_870 = list(anno870d0.Cell[ind_myeloid_malign_870]) + list(anno870d0.Cell[ind_myeloid_benign_870])
barcodes_myeloid_916 = list(anno916d0.Cell[ind_myeloid_malign_916]) + list(anno916d0.Cell[ind_myeloid_benign_916])
barcodes_myeloid_921A = list(anno921Ad0.Cell[ind_myeloid_malign_921A]) + list(anno921Ad0.Cell[ind_myeloid_benign_921A])
barcodes_myeloid_1012 = list(anno1012d0.Cell[ind_myeloid_malign_1012]) + list(anno1012d0.Cell[ind_myeloid_benign_1012])
barcodes_myeloid_BM1 = list(annoBM1.Cell[ind_myeloid_malign_BM1]) + list(annoBM1.Cell[ind_myeloid_benign_BM1])
barcodes_myeloid_BM2 = list(annoBM2.Cell[ind_myeloid_malign_BM2]) + list(annoBM2.Cell[ind_myeloid_benign_BM2])
barcodes_myeloid_BM3 = list(annoBM3.Cell[ind_myeloid_malign_BM3]) + list(annoBM3.Cell[ind_myeloid_benign_BM3])
barcodes_myeloid_BM4 = list(annoBM4.Cell[ind_myeloid_malign_BM4]) + list(annoBM4.Cell[ind_myeloid_benign_BM4])
barcodes_myeloid_BM5 = list(annoBM5.Cell[ind_myeloid_malign_BM5]) + list(annoBM5.Cell[ind_myeloid_benign_BM5])
barcodes_myeloid_healthy1 = list(annohealthy1.Cell[ind_myeloid_malign_healthy1]) + list(annohealthy1.Cell[ind_myeloid_benign_healthy1])
barcodes_myeloid_healthy2 = list(annohealthy2.Cell[ind_myeloid_malign_healthy2]) + list(annohealthy2.Cell[ind_myeloid_benign_healthy2])
barcodes_myeloid_healthy3 = list(annohealthy3.Cell[ind_myeloid_malign_healthy3]) + list(annohealthy3.Cell[ind_myeloid_benign_healthy3])
barcodes_myeloid_patient1 = list(annopatient1.Cell[ind_myeloid_malign_patient1]) + list(annopatient1.Cell[ind_myeloid_benign_patient1])
barcodes_myeloid_patient2 = list(annopatient2.Cell[ind_myeloid_malign_patient2]) + list(annopatient2.Cell[ind_myeloid_benign_patient2])
barcodes_myeloid_patient3 = list(annopatient3.Cell[ind_myeloid_malign_patient3]) + list(annopatient3.Cell[ind_myeloid_benign_patient3])
barcodes_myeloid_patient4 = list(annopatient4.Cell[ind_myeloid_malign_patient4]) + list(annopatient4.Cell[ind_myeloid_benign_patient4])
#barcodes_myeloid_patient5 = list(annopatient5.Cell[ind_myeloid_malign_patient5]) + list(annopatient5.Cell[ind_myeloid_benign_patient5])
barcodes_myeloid_patient6 = list(annopatient6.Cell[ind_myeloid_malign_patient6]) + list(annopatient6.Cell[ind_myeloid_benign_patient6])



# Obtener los indices de cada gen 
m_col_210A = [i for i, x in enumerate(df_210A.columns) if x in barcodes_myeloid_210A]
m_col_328 = [i for i, x in enumerate(df_328.columns) if x in barcodes_myeloid_328]
m_col_329 = [i for i, x in enumerate(df_329.columns) if x in barcodes_myeloid_329]
m_col_371 = [i for i, x in enumerate(df_371.columns) if x in barcodes_myeloid_371]
m_col_419 = [i for i, x in enumerate(df_419.columns) if x in barcodes_myeloid_419]
m_col_420B = [i for i, x in enumerate(df_420B.columns) if x in barcodes_myeloid_420B]
m_col_475 = [i for i, x in enumerate(df_475.columns) if x in barcodes_myeloid_475]
m_col_556 = [i for i, x in enumerate(df_556.columns) if x in barcodes_myeloid_556]
m_col_707B = [i for i, x in enumerate(df_707B.columns) if x in barcodes_myeloid_707B]
m_col_870 = [i for i, x in enumerate(df_870.columns) if x in barcodes_myeloid_870]
m_col_916 = [i for i, x in enumerate(df_916.columns) if x in barcodes_myeloid_916]
m_col_921A = [i for i, x in enumerate(df_921A.columns) if x in barcodes_myeloid_921A]
m_col_1012 = [i for i, x in enumerate(df_1012.columns) if x in barcodes_myeloid_1012]
m_col_BM1 = [i for i, x in enumerate(df_BM1.columns) if x in barcodes_myeloid_BM1]
m_col_BM2 = [i for i, x in enumerate(df_BM2.columns) if x in barcodes_myeloid_BM2]
m_col_BM3 = [i for i, x in enumerate(df_BM3.columns) if x in barcodes_myeloid_BM3]
m_col_BM4 = [i for i, x in enumerate(df_BM4.columns) if x in barcodes_myeloid_BM4]
m_col_BM5 = [i for i, x in enumerate(df_BM5.columns) if x in barcodes_myeloid_BM5]
m_col_healthy1 = [i for i, x in enumerate(df_healthy1.columns) if x in barcodes_myeloid_healthy1]
m_col_healthy2 = [i for i, x in enumerate(df_healthy2.columns) if x in barcodes_myeloid_healthy2]
m_col_healthy3 = [i for i, x in enumerate(df_healthy3.columns) if x in barcodes_myeloid_healthy3]
m_col_patient1 = [i for i, x in enumerate(df_patient1.columns) if x in barcodes_myeloid_patient1]
m_col_patient2 = [i for i, x in enumerate(df_patient2.columns) if x in barcodes_myeloid_patient2]
m_col_patient3 = [i for i, x in enumerate(df_patient3.columns) if x in barcodes_myeloid_patient3]
m_col_patient4 = [i for i, x in enumerate(df_patient4.columns) if x in barcodes_myeloid_patient4]
#m_col_patient5 = [i for i, x in enumerate(df_patient5.columns) if x in barcodes_myeloid_patient5]
m_col_patient6 = [i for i, x in enumerate(df_patient6.columns) if x in barcodes_myeloid_patient6]


# =============================================================================
# Generación de archivos andata
# =============================================================================
df_210A = df_210A.iloc[iGenes_210A,m_col_210A]
df_328 = df_328.iloc[iGenes_328, m_col_328]
df_329 = df_329.iloc[iGenes_329, m_col_329]
df_371 = df_371.iloc[iGenes_371, m_col_371]
df_419 = df_419.iloc[iGenes_419, m_col_419]
df_420B = df_420B.iloc[iGenes_420B, m_col_420B]
df_475 = df_475.iloc[iGenes_475, m_col_475]
df_556 = df_556.iloc[iGenes_556, m_col_556]
df_707B = df_707B.iloc[iGenes_707B, m_col_707B]
df_870 = df_870.iloc[iGenes_870, m_col_870]
df_916 = df_916.iloc[iGenes_916, m_col_916]
df_921A = df_921A.iloc[iGenes_921A, m_col_921A]
df_1012 = df_1012.iloc[iGenes_1012, m_col_1012]
df_BM1 = df_BM1.iloc[iGenes_BM1, m_col_BM1]
df_BM2 = df_BM2.iloc[iGenes_BM2, m_col_BM2]
df_BM3 = df_BM3.iloc[iGenes_BM3, m_col_BM3]
df_BM4 = df_BM4.iloc[iGenes_BM4, m_col_BM4]
df_BM5 = df_BM5.iloc[iGenes_BM5, m_col_BM5]
df_healthy1 = df_healthy1.iloc[iGenes_healthy1, m_col_healthy1]
df_healthy2 = df_healthy2.iloc[iGenes_healthy2, m_col_healthy2]
df_healthy3 = df_healthy3.iloc[iGenes_healthy3, m_col_healthy3]
df_patient1 = df_patient1.iloc[iGenes_patient1, m_col_patient1]
df_patient2 = df_patient2.iloc[iGenes_patient2, m_col_patient2]
df_patient3 = df_patient3.iloc[iGenes_patient3, m_col_patient3]
df_patient4 = df_patient4.iloc[iGenes_patient4, m_col_patient4]
#df_patient5 = df_patient5.iloc[iGenes_patient5, m_col_patient5]
df_patient6 = df_patient6.iloc[iGenes_patient6, m_col_patient6]


# =============================================================================
# index = pd.Index(range(0,22312))
# df_healthy1 = df_healthy1.set_index(index)
# =============================================================================

df_healthy1.reset_index(inplace=True,drop=True)
df_healthy2.reset_index(inplace=True,drop=True)
df_healthy3.reset_index(inplace=True,drop=True)
df_patient1.reset_index(inplace=True,drop=True)
df_patient2.reset_index(inplace=True,drop=True)
df_patient3.reset_index(inplace=True,drop=True)
df_patient4.reset_index(inplace=True,drop=True)
#df_patient5.reset_index(inplace=True,drop=True)
df_patient6.reset_index(inplace=True,drop=True)

# =============================================================================
# Etiquetaqs de las células seleccionadas (malignas y benignas)
# =============================================================================

#df_d0 = df_d0.iloc[:, m_col_d0]
#df_d37 = df_d37.iloc[:, m_col_d37]

barcodes_210A = [x for x in df_210A.columns if 'Gene' not in x]
barcodes_328 = [x for x in df_328.columns if 'Gene' not in x]
barcodes_329 = [x for x in df_329.columns if 'Gene' not in x]
barcodes_371 = [x for x in df_371.columns if 'Gene' not in x]
barcodes_419 = [x for x in df_419.columns if 'Gene' not in x]
barcodes_420B = [x for x in df_420B.columns if 'Gene' not in x]
barcodes_475 = [x for x in df_475.columns if 'Gene' not in x]
barcodes_556 = [x for x in df_556.columns if 'Gene' not in x]
barcodes_707B = [x for x in df_707B.columns if 'Gene' not in x]
barcodes_870 = [x for x in df_870.columns if 'Gene' not in x]
barcodes_916 = [x for x in df_916.columns if 'Gene' not in x]
barcodes_921A = [x for x in df_921A.columns if 'Gene' not in x]
barcodes_1012 = [x for x in df_1012.columns if 'Gene' not in x]
barcodes_BM1 = [x for x in df_BM1.columns if 'Gene' not in x]
barcodes_BM2 = [x for x in df_BM2.columns if 'Gene' not in x]
barcodes_BM3 = [x for x in df_BM3.columns if 'Gene' not in x]
barcodes_BM4 = [x for x in df_BM4.columns if 'Gene' not in x]
barcodes_BM5 = [x for x in df_BM5.columns if 'Gene' not in x]
barcodes_healthy1 = [x for x in df_healthy1.columns if 'Gene' not in x]
barcodes_healthy2 = [x for x in df_healthy2.columns if 'Gene' not in x]
barcodes_healthy3 = [x for x in df_healthy3.columns if 'Gene' not in x]
barcodes_patient1 = [x for x in df_patient1.columns if 'Gene' not in x]
barcodes_patient2 = [x for x in df_patient2.columns if 'Gene' not in x]
barcodes_patient3 = [x for x in df_patient3.columns if 'Gene' not in x]
barcodes_patient4 = [x for x in df_patient4.columns if 'Gene' not in x]
#barcodes_patient5 = [x for x in df_patient5.columns if 'Gene' not in x]
barcodes_patient6 = [x for x in df_patient6.columns if 'Gene' not in x]



barcodes_train = barcodes_210A + barcodes_328 + barcodes_329 + barcodes_371 + barcodes_419 + barcodes_420B + barcodes_475 + barcodes_556 + barcodes_707B + barcodes_870 + barcodes_916 + barcodes_921A + barcodes_1012 + barcodes_BM1 + barcodes_BM2 + barcodes_BM3 + barcodes_BM4 + barcodes_BM5 + barcodes_healthy1 + barcodes_healthy2 + barcodes_healthy3 + barcodes_patient1 + barcodes_patient2 + barcodes_patient3 + barcodes_patient4 + barcodes_patient6
# barcodes_patient5

df_train = np.concatenate((df_210A, df_328, df_329, df_371, df_419, df_420B, df_475, 
                           df_556, df_707B, df_870, df_916, df_921A, df_1012, df_BM1, 
                           df_BM2, df_BM3, df_BM4, df_BM5, df_healthy1, df_healthy2, 
                           df_healthy3, df_patient1, df_patient2, df_patient3, df_patient4,
                           df_patient6),axis=1) # df_patient5

#df_train=pd.DataFrame(df_train,columns=barcodes_train)

df_train=pd.DataFrame(df_train)

## Transformar a dataframe para poderlo usar en el archivo andata
a = anndata_inner.var_names
a = pd.DataFrame(anndata_inner.var_names)

# =============================================================================
# =============================================================================
# Cargar los datos seleccionados en el archivo andata para normalizarlos
# =============================================================================
# =============================================================================

andata_train = sc.AnnData(X=df_train.T.to_numpy(), obs=barcodes_train, var = a)

andata_train.var_names = anndata_inner.var_names # genes

andata_train.obs_names = barcodes_train # células

#sc.pp.filter_genes(anndata_val, min_cells=50)

andata_train.shape

andata_train.raw = andata_train # trabajamos con el archivo .raw de las cuenats
sc.pl.highest_expr_genes(andata_train, n_top=20)

andata_train.shape

# =============================================================================
# # Normalizamos y filtramos 
# =============================================================================

#sc.pp.filter_genes(andata_t, min_cells=50)

#andata_t.shape

sc.pp.normalize_total(andata_train, target_sum=1e4)
sc.pp.log1p(andata_train)

np.max(andata_train.raw.X)
np.max(andata_train.X)

sc.pl.highest_expr_genes(andata_train, n_top=50)

# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================

# =============================================================================
# from kneed import KneeLocator as kl
# from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import DistanceMetric
# from sklearn.decomposition import PCA
# import louvain
# large_root_PCA = r"C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\AML_OCI"
# 
# sc.pp.highly_variable_genes(andata_train, min_mean=0.0125, max_mean=4, min_disp=0.25)
# andata_train.var
# 
# sc.pl.highly_variable_genes(andata_train)
# 
# sc.pp.scale(andata_train, max_value=10)
# 
# 
# ################################################################
# ############### Principal component analysis####################
# 
# # se reduce la dimensionalidad de los datos ejecutando un 
# # análisis de componentes principales (PCA), que revela los 
# # principales ejes de variación y elimina el ruido de los datos.
# sc.tl.pca(andata_train, svd_solver='arpack')
# 
# # Diagrama de dispersión en las coordenadas PCA
# sc.pl.pca(andata_train, color='CST3')
# 
# # Se inspecciona la contribución de los PCs individuales a la 
# # varianza total en los datos
# sc.pl.pca_variance_ratio(andata_train, log=True)
# 
# # Trazar las cargas, o la contribución de cada gen a las PCs
# sc.pl.pca_loadings(andata_train)
# 
# 
# # dispersión en las coordenadas PCA de todos los genes
# #sc.pp.scale(norm_data, max_value = 0)
# #sc.tl.pca(norm_data, svd_solver = 'arpack')
# #sc.pl.pca(norm_data, colors = 'CST3')
# #sc.pl pca_loadings(norm_data)
# 
# # visualizar diagrama 
# #sc.pl.pca_variance_ratio(bdata, log = True)
# 
# # Guardar resultados 
# #bdata.write(results) # Guardar los resultados
# 
# # =============================================================================
# # Kmeans
# # Utiliza los genes altamente variables 
# # =============================================================================
# ## Regresando a Scanpy  
# ####### Computing the neighborhood graph ########
# 
# sc.pp.neighbors(andata_train, n_neighbors=10, n_pcs=10)
# 
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score
# 
# kmeans = KMeans(n_clusters=4, random_state=0).fit(andata_train.obsm["X_pca"])
# andata_train.obs['kmeans2'] = kmeans.labels_.astype(str)
# 
# fig, axs = plt.subplots(2,2, figsize=(7,5), constrained_layout=True)
# sc.pl.pca_scatter(andata_train, color="kmeans2", ax=axs[0,0], show=False, size=10, use_raw=False, legend_loc="on data")
# sc.pl.pca_scatter(andata_train, color="kmeans2", components="1,3", ax=axs[0,1], show=False, size=10, use_raw=False, legend_loc="on data")
# sc.pl.pca_scatter(andata_train, color="kmeans2", components="2,3", ax=axs[1,0], show=False, size=10, use_raw=False, legend_loc="on data")
# sc.pl.pca_scatter(andata_train, color="kmeans2", components="1,4", ax=axs[1,1], show=False, size=10, use_raw=False, legend_loc="on data")
# 
# =============================================================================
# =============================================================================
# =============================================================================
# # Machine Learning
# =============================================================================
# =============================================================================

# annopatient5.Cell
# annopatient5.PredictionRefined
# Agregar la lista de celulas y etiquetas a una variable, obtenida del archivo anno
barcodes2class = dict(zip(pd.concat((anno210Ad0.Cell, anno328d0.Cell, anno329d0.Cell, anno371d0.Cell, anno419d0.Cell, anno420Bd0.Cell, 
                                     anno475d0.Cell, anno556d0.Cell, anno707Bd0.Cell, anno870d0.Cell, anno916d0.Cell, anno921Ad0.Cell, anno1012d0.Cell, 
                                     annoBM1.Cell, annoBM2.Cell, annoBM3.Cell,
                                     annoBM4.Cell, annoBM5.Cell, annohealthy1.Cell, annohealthy2.Cell, annohealthy3.Cell, 
                                     annopatient1.Cell, annopatient2.Cell, annopatient3.Cell, annopatient4.Cell, annopatient6.Cell)),
                          pd.concat((anno210Ad0.PredictionRefined, anno328d0.PredictionRefined, anno329d0.PredictionRefined, anno371d0.PredictionRefined, anno419d0.PredictionRefined, 
                                     anno420Bd0.PredictionRefined, anno475d0.PredictionRefined, anno556d0.PredictionRF2,
                                     anno707Bd0.PredictionRefined, anno870d0.PredictionRefined,
                                     anno916d0.PredictionRefined, anno921Ad0.PredictionRefined, anno1012d0.PredictionRefined, 
                                     annoBM1.PredictionRefined, annoBM2.PredictionRefined, annoBM3.PredictionRefined, 
                                     annoBM4.PredictionRefined, annoBM5.PredictionRefined,
                                     annohealthy1.PredictionRefined, annohealthy2.PredictionRefined, annohealthy3.PredictionRefined, 
                                     annopatient1.PredictionRefined, annopatient2.PredictionRefined, annopatient3.PredictionRefined,
                                     annopatient4.PredictionRefined, annopatient6.PredictionRefined))))




# Etiquetar cada celula con malig y benig
y_true = [barcodes2class[x] for x in andata_train.obs_names]
X = andata_train.X
X_train = X
classdict = dict(normal = 0, unclear = 0, malignant=1)
y_true_num = [classdict[x] for x in y_true] #Agregamos etiqueta 0 y 1 
y_train = np.array(y_true_num)


# =============================================================================
# =============================================================================
# PCA Visualizar los datos
# =============================================================================
# =============================================================================

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(X_train)
my_components_train=pca.transform(X_train)

#plt.figure(figsize=(10,7))
#plt.scatter(my_components_train[:,0],my_components_train[:,1],c=y_train,s=5,alpha=0.75)
#plt.xlabel("PC1")
#plt.ylabel("PC2")
#plt.title("Análsis de componentes principales para las características de las células")
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
clases=['Células normales','Células malignas']
for clase in np.unique(y_train):
    ax.scatter(my_components_train[np.where(y_train==clase),0],
               my_components_train[np.where(y_train==clase),1],
               my_components_train[np.where(y_train==clase),2],
               label=clases[clase],s=10,alpha=0.75)
ax.legend(loc='best',fontsize=20)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()

plt.figure()
pca1=my_components_train[:,0]
idxsc1=np.where(y_train==0)
idxsc2=np.where(y_train==1)
g_normales=pca1[idxsc1]
g_malignos=pca1[idxsc2]
plt.hist(g_normales,bins=500,alpha=0.75, label="Normales")
plt.hist(g_malignos,bins=500,alpha=0.75, label="Malignas")
plt.legend(loc="best",fontsize=20)
plt.show()

# =============================================================================
# =============================================================================
# MACHINE LEARNING
# =============================================================================
# =============================================================================
Xtrain,Xtest,ytrain,ytest=train_test_split(X_train,y_train,test_size=0.2,shuffle=True)


clf_svm = SVC()
scores_svm = cross_validate(clf_svm, Xtrain, 
                        ytrain, cv=5,
                        scoring={'accuracy_score':make_scorer(accuracy_score),
                                 'precision_score':make_scorer(precision_score),
                                 'prc':make_scorer(average_precision_score),
                                 'mcc':make_scorer(matthews_corrcoef)})

# Entrenamiento en todo el conjunto de entrenamiento
clf_svm.fit(Xtest, ytest)

scores_svm
 
mean_cv = [x+':'+str(np.mean(scores_svm[x])) for x in scores_svm.keys() ]
std_cv = [x+':'+str(np.std(scores_svm[x])) for x in scores_svm.keys() ]

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svm['test_accuracy_score'].mean(), scores_svm['test_accuracy_score'].std() * 2))
print("Precision: %0.2f (+/- %0.2f)" % (scores_svm['test_precision_score'].mean(), scores_svm['test_precision_score'].std() * 2))
print("MCC: %0.2f (+/- %0.2f)" % (scores_svm['test_mcc'].mean(), scores_svm['test_mcc'].std() * 2))

scores_svm.keys()


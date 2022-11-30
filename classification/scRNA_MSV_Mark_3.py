# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:39:21 2022

@author: IngBi
"""

import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from umap import UMAP
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# =============================================================================
# Estructura e importe de datos
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

# Normalization and filtering
#sc.pp.filter_genes(anndata_FA, min_cells=50)
#sc.pp.filter_cells(anndata_FA, min_genes=200)

sc.pp.normalize_total(anndata_inner, target_sum=1e4)
sc.pp.log1p(anndata_inner)

sc.pl.highest_expr_genes(anndata_inner, n_top=50,)


# =============================================================================
# Lectura de archivos anno
# =============================================================================
dir_anno_val = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

#anno371d0 = pd.read_csv(dir_anno_val + '\GSM3587947_AML371-D0.anno.txt.gz',
                         #sep='\t')

anno419d0 = pd.read_csv(dir_anno_val + '\GSM3587951_AML419A-D0.anno.txt.gz',
                         sep='\t')

anno420Bd0 = pd.read_csv(dir_anno_val + '\GSM3587954_AML420B-D0.anno.txt.gz',
                         sep='\t')

#anno556d0 = pd.read_csv(dir_anno_val + '\GSM3587964_AML556-D0.anno.txt.gz',
                         #sep='\t')

anno707Bd0 = pd.read_csv(dir_anno_val + '\GSM3587970_AML707B-D0.anno.txt.gz',
                         sep='\t')

anno916d0 = pd.read_csv(dir_anno_val + '\GSM3587989_AML916-D0.anno.txt.gz',
                         sep='\t')

#anno921Ad0 = pd.read_csv(dir_anno_val + '\GSM3587991_AML921A-D0.anno.txt.gz',
                         #sep='\t')

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

#annoBM5 = pd.read_csv(dir_anno_val + '\GSM3588002_BM5-34p.anno.txt.gz', 
                        #sep='\t')

# =============================================================================
# Información de las celulas malignas y benignas de los pacientes
# =============================================================================

#anno371d0.columns
#print(Counter(anno371d0.PredictionRF2))
#print(Counter(anno371d0.CellType))

anno419d0.columns
print(Counter(anno419d0.PredictionRefined))
print(Counter(anno419d0.CellType))

anno420Bd0.columns
print(Counter(anno420Bd0.PredictionRefined))
print(Counter(anno420Bd0.CellType))

#anno556d0.columns
#print(Counter(anno556d0.PredictionRefined))
#print(Counter(anno556d0.CellType))

anno707Bd0.columns
print(Counter(anno707Bd0.PredictionRefined))
print(Counter(anno707Bd0.CellType))

anno916d0.columns
print(Counter(anno916d0.PredictionRefined))
print(Counter(anno916d0.CellType))

#anno921Ad0.columns
#print(Counter(anno921Ad0.PredictionRefined))
#print(Counter(anno921Ad0.CellType))

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

#annoBM5.columns
#print(Counter(annoBM5.PredictionRefined))
#print(Counter(annoBM5.CellType))


# =============================================================================
# Revisar nuestros datos
# =============================================================================
#anno1012d0['PredictionRefined'].value_counts().tolist()
#anno1012d0['PredictionRefined'].value_counts().keys().tolist()

#norm_371 = anno371d0['PredictionRF2'].str.contains('normal').value_counts()[True]
#malig_371 = anno371d0['PredictionRF2'].str.contains('malignant').value_counts()[True]

norm_419 = anno419d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_419 = anno419d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_420 = anno420Bd0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_420 = anno420Bd0['PredictionRefined'].str.contains('malignant').value_counts()[True]

#norm_556 = anno556d0['PredictionRefined'].str.contains('normal').value_counts()[True]
#malig_556 = anno556d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_707 = anno707Bd0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_707 = anno707Bd0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_916 = anno916d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_916 = anno916d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

norm_1012 = anno1012d0['PredictionRefined'].str.contains('normal').value_counts()[True]
malig_1012 = anno1012d0['PredictionRefined'].str.contains('malignant').value_counts()[True]

BM1_norm = annoBM1['PredictionRefined'].str.contains('normal').value_counts()[True]

BM2_norm = annoBM2['PredictionRefined'].str.contains('normal').value_counts()[True]

BM3_norm = annoBM3['PredictionRefined'].str.contains('normal').value_counts()[True]

BM4_norm = annoBM4['PredictionRefined'].str.contains('normal').value_counts()[True]

#BM5_norm = annoBM5['PredictionRefined'].str.contains('normal').value_counts()[True]

total_cell_benign = norm_419 + norm_420 + norm_707 + norm_916 + norm_1012 + BM1_norm + BM2_norm + BM3_norm + BM4_norm
total_cell_malig = malig_419 + malig_420 + malig_707 + malig_916 + malig_1012

# =============================================================================
# Myeloid cells
# =============================================================================

# Myeloid cells
mcells = ['HSC', 'Prog', 'GMP', 'Promono', 'Mono', 'cDC', 'pDC'] # Revisar si agregamos los otros grupos

# Seleccionar sólo células mieloides (normales o tumorales)
#ind_myeloid_malign_371 = [i for i, x in enumerate(anno371d0.CellType) if 'like' in x]
#ind_myeloid_benign_371 = [i for i, x in enumerate(anno371d0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_419 = [i for i, x in enumerate(anno419d0.CellType) if 'like' in x]
ind_myeloid_benign_419 = [i for i, x in enumerate(anno419d0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_420B = [i for i, x in enumerate(anno420Bd0.CellType) if 'like' in x]
ind_myeloid_benign_420B = [i for i, x in enumerate(anno420Bd0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
#ind_myeloid_malign_556 = [i for i, x in enumerate(anno556d0.CellType) if 'like' in x]
#ind_myeloid_benign_556 = [i for i, x in enumerate(anno556d0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_707B = [i for i, x in enumerate(anno707Bd0.CellType) if 'like' in x]
ind_myeloid_benign_707B = [i for i, x in enumerate(anno707Bd0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_916 = [i for i, x in enumerate(anno916d0.CellType) if 'like' in x]
ind_myeloid_benign_916 = [i for i, x in enumerate(anno916d0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
#ind_myeloid_malign_921A = [i for i, x in enumerate(anno921Ad0.CellType) if 'like' in x]
#ind_myeloid_benign_921A = [i for i, x in enumerate(anno921Ad0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_1012 = [i for i, x in enumerate(anno1012d0.CellType) if 'like' in x]
ind_myeloid_benign_1012 = [i for i, x in enumerate(anno1012d0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_BM1 = [i for i, x in enumerate(annoBM1.CellType) if 'like' in x]
ind_myeloid_benign_BM1 = [i for i, x in enumerate(annoBM1.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_BM2 = [i for i, x in enumerate(annoBM2.CellType) if 'like' in x]
ind_myeloid_benign_BM2 = [i for i, x in enumerate(annoBM2.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_BM3 = [i for i, x in enumerate(annoBM3.CellType) if 'like' in x]
ind_myeloid_benign_BM3 = [i for i, x in enumerate(annoBM3.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_BM4 = [i for i, x in enumerate(annoBM4.CellType) if 'like' in x]
ind_myeloid_benign_BM4 = [i for i, x in enumerate(annoBM4.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
#ind_myeloid_malign_BM5 = [i for i, x in enumerate(annoBM5.CellType) if 'like' in x]
#ind_myeloid_benign_BM5 = [i for i, x in enumerate(annoBM5.CellType) if x in mcells]

total_cell_malig_new = len(ind_myeloid_malign_419) + len(ind_myeloid_malign_420B) + len(ind_myeloid_malign_707B) + len(ind_myeloid_malign_916) + len(ind_myeloid_malign_1012) + len(ind_myeloid_malign_BM1) + len(ind_myeloid_malign_BM2) + len(ind_myeloid_malign_BM4)
total_cell_benign_new = len(ind_myeloid_benign_419) + len(ind_myeloid_benign_420B) + len(ind_myeloid_benign_707B) + len(ind_myeloid_benign_916) + len(ind_myeloid_benign_1012) + len(ind_myeloid_benign_BM1) + len(ind_myeloid_benign_BM2) + len(ind_myeloid_benign_BM3) + len(ind_myeloid_benign_BM4)
# =============================================================================
# =============================================================================
# # Cargar el archivo de conteo de los sujetos muestra
# =============================================================================
# =============================================================================

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_419 = pd.read_csv(ge_files +'\GSM3587950_AML419A-D0.dem.txt.gz',
                 sep='\t')

df_420B = pd.read_csv(ge_files +'\GSM3587953_AML420B-D0.dem.txt.gz',
                 sep='\t')

#df_556 = pd.read_csv(ge_files +'\GSM3587963_AML556-D0.dem.txt.gz',
                 #sep='\t')

df_707B = pd.read_csv(ge_files +'\GSM3587969_AML707B-D0.dem.txt.gz',
                 sep='\t')

df_916 = pd.read_csv(ge_files +'\GSM3587988_AML916-D0.dem.txt.gz',
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

#df_BM5 = pd.read_csv(ge_files +'\GSM3588002_BM5-34p.dem.txt.gz',
                 #sep='\t')

###  Indices de los genes 
iGenes_419 = [i for i, x in enumerate(df_419['Gene']) if x in anndata_inner.var_names]
iGenes_420B = [i for i, x in enumerate(df_420B['Gene']) if x in anndata_inner.var_names]
#iGenes_556 = [i for i, x in enumerate(df_556['Gene']) if x in andata.var_names]
iGenes_707B = [i for i, x in enumerate(df_707B['Gene']) if x in anndata_inner.var_names]
iGenes_916 = [i for i, x in enumerate(df_916['Gene']) if x in anndata_inner.var_names]
iGenes_1012 = [i for i, x in enumerate(df_1012['Gene']) if x in anndata_inner.var_names]
iGenes_BM1 = [i for i, x in enumerate(df_BM1['Gene']) if x in anndata_inner.var_names]
iGenes_BM2 = [i for i, x in enumerate(df_BM2['Gene']) if x in anndata_inner.var_names]
iGenes_BM3 = [i for i, x in enumerate(df_BM3['Gene']) if x in anndata_inner.var_names]
iGenes_BM4 = [i for i, x in enumerate(df_BM4['Gene']) if x in anndata_inner.var_names]
#iGenes_BM5 = [i for i, x in enumerate(df_BM5['Gene']) if x in andata.var_names]

### Nombres de los genes
gene_names_419 = df_419['Gene']
gene_names_420B = df_420B['Gene']
#gene_names_556 = df_556['Gene']
gene_names_707B = df_707B['Gene']
gene_names_916 = df_916['Gene']
gene_names_1012 = df_1012['Gene']
gene_names_BM1 = df_BM1['Gene']
gene_names_BM2 = df_BM2['Gene']
gene_names_BM3 = df_BM3['Gene']
gene_names_BM4 = df_BM4['Gene']
#gene_names_BM5 = df_BM5['Gene']


#genes_t = pd.concat((gene_names_d0, gene_names_d37),axis=1, ignore_index=True)

### Seleccion de celulas malignas y benignas
barcodes_myeloid_419 = list(anno419d0.Cell[ind_myeloid_malign_419]) + list(anno419d0.Cell[ind_myeloid_benign_419])
barcodes_myeloid_420B = list(anno420Bd0.Cell[ind_myeloid_malign_420B]) + list(anno420Bd0.Cell[ind_myeloid_benign_420B])
#barcodes_myeloid_556 = list(anno556d0.Cell[ind_myeloid_malign_556]) + list(anno556d0.Cell[ind_myeloid_benign_556])
barcodes_myeloid_707B = list(anno707Bd0.Cell[ind_myeloid_malign_707B]) + list(anno707Bd0.Cell[ind_myeloid_benign_707B])
barcodes_myeloid_916 = list(anno916d0.Cell[ind_myeloid_malign_916]) + list(anno916d0.Cell[ind_myeloid_benign_916])
barcodes_myeloid_1012 = list(anno1012d0.Cell[ind_myeloid_malign_1012]) + list(anno1012d0.Cell[ind_myeloid_benign_1012])
barcodes_myeloid_BM1 = list(annoBM1.Cell[ind_myeloid_malign_BM1]) + list(annoBM1.Cell[ind_myeloid_benign_BM1])
barcodes_myeloid_BM2 = list(annoBM2.Cell[ind_myeloid_malign_BM2]) + list(annoBM2.Cell[ind_myeloid_benign_BM2])
barcodes_myeloid_BM3 = list(annoBM3.Cell[ind_myeloid_malign_BM3]) + list(annoBM3.Cell[ind_myeloid_benign_BM3])
barcodes_myeloid_BM4 = list(annoBM4.Cell[ind_myeloid_malign_BM4]) + list(annoBM4.Cell[ind_myeloid_benign_BM4])
#barcodes_myeloid_BM5 = list(annoBM5.Cell[ind_myeloid_malign_BM5]) + list(annoBM5.Cell[ind_myeloid_benign_BM5])


m_col_419 = [i for i, x in enumerate(df_419.columns) if x in barcodes_myeloid_419]
m_col_420B = [i for i, x in enumerate(df_420B.columns) if x in barcodes_myeloid_420B]
#m_col_556 = [i for i, x in enumerate(df_556.columns) if x in barcodes_myeloid_556]
m_col_707B = [i for i, x in enumerate(df_707B.columns) if x in barcodes_myeloid_707B]
m_col_916 = [i for i, x in enumerate(df_916.columns) if x in barcodes_myeloid_916]
m_col_1012 = [i for i, x in enumerate(df_1012.columns) if x in barcodes_myeloid_1012]
m_col_BM1 = [i for i, x in enumerate(df_BM1.columns) if x in barcodes_myeloid_BM1]
m_col_BM2 = [i for i, x in enumerate(df_BM2.columns) if x in barcodes_myeloid_BM2]
m_col_BM3 = [i for i, x in enumerate(df_BM3.columns) if x in barcodes_myeloid_BM3]
m_col_BM4 = [i for i, x in enumerate(df_BM4.columns) if x in barcodes_myeloid_BM4]
#m_col_BM5 = [i for i, x in enumerate(df_BM5.columns) if x in barcodes_myeloid_BM5]

# =============================================================================
# Indices de las células seleccionadas (malignas y benignas)
# =============================================================================

df_419 = df_419.iloc[iGenes_419, m_col_419]
df_420B = df_420B.iloc[iGenes_420B, m_col_420B]
df_707B = df_707B.iloc[iGenes_707B, m_col_707B]
df_916 = df_916.iloc[iGenes_916, m_col_916]
df_1012 = df_1012.iloc[iGenes_1012, m_col_1012]
df_BM1 = df_BM1.iloc[iGenes_BM1, m_col_BM1]
df_BM2 = df_BM2.iloc[iGenes_BM2, m_col_BM2]
df_BM3 = df_BM3.iloc[iGenes_BM3, m_col_BM3]
df_BM4 = df_BM4.iloc[iGenes_BM4, m_col_BM4]
#df_BM5 = df_BM5.iloc[iGenes_BM5, m_col_BM5]

# =============================================================================
# Etiquetaqs de las células seleccionadas (malignas y benignas)
# =============================================================================

#df_d0 = df_d0.iloc[:, m_col_d0]
#df_d37 = df_d37.iloc[:, m_col_d37]

barcodes_419 = [x for x in df_419.columns if 'Gene' not in x]
barcodes_420B = [x for x in df_420B.columns if 'Gene' not in x]
barcodes_707B = [x for x in df_707B.columns if 'Gene' not in x]
barcodes_916 = [x for x in df_916.columns if 'Gene' not in x]
barcodes_1012 = [x for x in df_1012.columns if 'Gene' not in x]
barcodes_BM1 = [x for x in df_BM1.columns if 'Gene' not in x]
barcodes_BM2 = [x for x in df_BM2.columns if 'Gene' not in x]
barcodes_BM3 = [x for x in df_BM3.columns if 'Gene' not in x]
barcodes_BM4 = [x for x in df_BM4.columns if 'Gene' not in x]
#barcodes_BM5 = [x for x in df_BM5.columns if 'Gene' not in x]


df_train = pd.concat((df_419,df_420B, df_707B, df_916, df_1012, 
                      df_BM1, df_BM2, df_BM3, df_BM4),axis=1, 
                     ignore_index=True)

barcodes_train = barcodes_419 + barcodes_420B + barcodes_707B + barcodes_916 + barcodes_1012 + barcodes_BM1 + barcodes_BM2 + barcodes_BM3 + barcodes_BM4

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
# # Machine Learning
# =============================================================================
# =============================================================================

# Agregar la lista de celulas y etiquetas a una variable, obtenida del archivo anno
barcodes2class = dict(zip(pd.concat((anno419d0.Cell, anno420Bd0.Cell, 
                                     anno707Bd0.Cell, anno916d0.Cell, anno1012d0.Cell, 
                                     annoBM1.Cell, annoBM2.Cell, annoBM3.Cell,
                                     annoBM4.Cell)),
                          pd.concat((anno419d0.PredictionRefined, anno420Bd0.PredictionRefined,
                                     anno707Bd0.PredictionRefined, 
                                     anno916d0.PredictionRefined, anno1012d0.PredictionRefined, 
                                     annoBM1.PredictionRefined, annoBM2.PredictionRefined, annoBM3.PredictionRefined, 
                                     annoBM4.PredictionRefined))))

# Etiquetar cada celula con malig y benig
y_true = [barcodes2class[x] for x in andata_train.obs_names]
X = andata_train.X
X_train = X
classdict = dict(normal=0, malignant=1)
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
ax.legend(loc='best',fontsize=20, edgecolor="k")
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
plt.hist(g_normales,bins=500,alpha=0.75)
plt.hist(g_malignos,bins=500,alpha=0.75)
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


# =============================================================================
# =============================================================================
# # Predicción en un conjunto de datos de validación
# =============================================================================
# =============================================================================

dir_anno_val = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

anno328d0 = pd.read_csv(dir_anno_val + '\GSM3587932_AML328-D0.anno.txt.gz', 
                        sep='\t')

anno329d37 = pd.read_csv(dir_anno_val + '\GSM3587945_AML329-D37.anno.txt.gz',
                         sep='\t')

# =============================================================================
# Información de los pacientes
# =============================================================================
anno328d0.columns
print(Counter(anno328d0.PredictionRefined))
print(Counter(anno328d0.CellType))

anno329d37.columns
print(Counter(anno329d37.PredictionRefined))
print(Counter(anno329d37.CellType))


# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_d0 = [i for i, x in enumerate(anno328d0.CellType) if 'like' in x]
ind_myeloid_benign_d0 = [i for i, x in enumerate(anno328d0.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_d37 = [i for i, x in enumerate(anno329d37.CellType) if 'like' in x]
ind_myeloid_benign_d37 = [i for i, x in enumerate(anno329d37.CellType) if x in mcells]

# =============================================================================
# =============================================================================
# # Cargar el archivo de conteo de los sujetos muestra
# =============================================================================
# =============================================================================

ge_files_val = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_d0 = pd.read_csv(ge_files_val +'\GSM3587931_AML328-D0.dem.txt.gz',
                 sep='\t')

df_d37 = pd.read_csv(ge_files_val +'\GSM3587944_AML329-D37.dem.txt.gz',
                 sep='\t')

iGenes_d0 = [i for i, x in enumerate(df_d0['Gene']) if x in andata_t.var_names]
iGenes_d37 = [i for i, x in enumerate(df_d37['Gene']) if x in andata_t.var_names]

gene_names_d0 = df_d0['Gene']
gene_names_d37 = df_d37['Gene']

#genes_t = pd.concat((gene_names_d0, gene_names_d37),axis=1, ignore_index=True)

# Selecting only Myeloid
barcodes_myeloid_d0 = list(anno328d0.Cell[ind_myeloid_malign_d0]) + list(anno328d0.Cell[ind_myeloid_benign_d0])
barcodes_myeloid_d37 = list(anno329d37.Cell[ind_myeloid_malign_d37]) + list(anno329d37.Cell[ind_myeloid_benign_d37])

m_col_d0 = [i for i, x in enumerate(df_d0.columns) if x in barcodes_myeloid_d0]
m_col_d37 = [i for i, x in enumerate(df_d37.columns) if x in barcodes_myeloid_d37]

# =============================================================================
df_d0 = df_d0.iloc[iGenes_d0, m_col_d0]
df_d37 = df_d37.iloc[iGenes_d37, m_col_d37]
# =============================================================================

#df_d0 = df_d0.iloc[:, m_col_d0]
#df_d37 = df_d37.iloc[:, m_col_d37]

barcodes_d0 = [x for x in df_d0.columns if 'Gene' not in x]
barcodes_d37 = [x for x in df_d37.columns if 'Gene' not in x]

df_val = pd.concat((df_d0, df_d37),axis=1, ignore_index=True)

barcodes_val = barcodes_d0 + barcodes_d37

a = andata_t.var_names
a = pd.DataFrame(andata_t.var_names)

andata_val = sc.AnnData(X=df_val.T.to_numpy(), obs=barcodes_val, var = a)

andata_val.var_names = andata_t.var_names

andata_val.obs_names = barcodes_val

#sc.pp.filter_genes(anndata_val, min_cells=50)

andata_val.shape

# Normalization per cell
sc.pp.normalize_total(andata_val)
sc.pp.log1p(andata_val)

barcodes2class_1 = dict(zip(pd.concat((anno328d0.Cell, anno329d37.Cell)),
                          pd.concat((anno328d0.PredictionRefined, anno329d37.PredictionRefined))))



y_true_val = [barcodes2class_1[x] for x in andata_val.obs_names]

X_val = andata_val.X
y_true_val_num = [classdict[x] for x in y_true_val]
y_val = np.array(y_true_val_num)

vald = Counter(y_val)
vald

ypred_val_svm = clf_svm.predict(X_val)

acc_val_svm = accuracy_score(y_val, ypred_val_svm)

prec_val_svm = precision_score(y_val, ypred_val_svm)  # Precision decrease a lot! (form 86% to 56%)

rec_val_svm = recall_score(y_val, ypred_val_svm)

mcc_val_svm = matthews_corrcoef(y_val, ypred_val_svm)

tn_val_svm, fp_val_svm, fn_val_svm, tp_val_svm = confusion_matrix(y_val, ypred_val_svm).ravel()

print('Accuracy on validation:'+str(acc_val_svm))
print('Precision on validation:'+str(prec_val_svm))
print('Recall on validation:'+str(rec_val_svm))
print('MCC on validation:'+str(mcc_val_svm))
spec_svm = tn_val_svm / (tn_val_svm + fp_val_svm)
print('Specificity on validation:'+str(spec_svm))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_val, ypred_val_svm)
plt.show()
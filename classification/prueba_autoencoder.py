# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 02:50:59 2022

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

# =============================================================================
# =============================================================================
# Estructura e importe de datos
# =============================================================================
# =============================================================================

# Lectura de archivos anno
dir_anno = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'
path_anno = dir_anno + '\GSM3587943_AML329-D20.anno.txt.gz'
anno = pd.read_csv(path_anno, sep='\t')

anno.columns # selecciona las columnas del archivo de anotaciones
print(Counter(anno.PredictionRefined)) # Numero de celulas malignas y sanas
print(Counter(anno.CellType)) # Numero de cada tipo de celula

# Myeloid cells
mcells = ['HSC', 'Prog', 'GMP', 'Promono', 'Mono', 'cDC', 'pDC']

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign = [i for i,x in enumerate(anno.CellType) if 'like' in x]
ind_myeloid_benign = [i for i,x in enumerate(anno.CellType) if x in mcells]

# Tamaño de cada variable beninga y maligna
len(ind_myeloid_benign)
len(ind_myeloid_malign)

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'
df_train = pd.read_csv(ge_files+'\GSM3587942_AML329-D20.dem.txt.gz',
                 sep='\t')

df_train.shape

#df.head()
gene_names = df_train['Gene'] # seleccion de genes AML329-D0

# Seleccionar sólo mieloides
barcodes_myeloid = list(anno.Cell[ind_myeloid_malign])+list(anno.Cell[ind_myeloid_benign])
m_col = [i for i,x in enumerate(df_train.columns) if x in barcodes_myeloid]
df_train = df_train.iloc[:, m_col] # Me quedo con la lista de malignas y benignas 714
df_train.shape # tamaño del archivo

# Agregar los genes a mis cuentas
barcodes = [x for x in df_train.columns if 'Gene' not in x] 

# Creamos el archivo anndata con el sistema mieloide seleccionado
andata = sc.AnnData(X=df_train.T.to_numpy(), obs=barcodes, var=gene_names.values)

andata.var_names = gene_names.values # Agregamos la lista de genes

andata.obs_names = barcodes # Agregamos la lista de células

andata.raw = andata # trabajamos con el archivo .raw de las cuenats
sc.pl.highest_expr_genes(andata, n_top=20)

andata.shape

# =============================================================================
# # Normalizamos y filtramos 
# =============================================================================

sc.pp.filter_genes(andata, min_cells=80)

andata.shape

sc.pp.normalize_total(andata, target_sum=1e4)
sc.pp.log1p(andata)

np.max(andata.raw.X)
np.max(andata.X)

sc.pl.highest_expr_genes(andata, n_top=50)


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

#annoBM3 = pd.read_csv(dir_anno_val + '\GSM3587999_BM3.anno.txt.gz', 
                        #sep='\t')

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

#annoBM3.columns
#print(Counter(annoBM3.PredictionRefined))
#print(Counter(annoBM3.CellType))

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

#BM3_norm = annoBM3['PredictionRefined'].str.contains('normal').value_counts()[True]

BM4_norm = annoBM4['PredictionRefined'].str.contains('normal').value_counts()[True]

#BM5_norm = annoBM5['PredictionRefined'].str.contains('normal').value_counts()[True]

total_cell_benign = norm_419 + norm_420 + norm_707 + norm_916 + norm_1012 + BM1_norm + BM2_norm + BM4_norm
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
#ind_myeloid_malign_BM3 = [i for i, x in enumerate(annoBM3.CellType) if 'like' in x]
#ind_myeloid_benign_BM3 = [i for i, x in enumerate(annoBM3.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
ind_myeloid_malign_BM4 = [i for i, x in enumerate(annoBM4.CellType) if 'like' in x]
ind_myeloid_benign_BM4 = [i for i, x in enumerate(annoBM4.CellType) if x in mcells]

# Seleccionar sólo células mieloides (normales o tumorales)
#ind_myeloid_malign_BM5 = [i for i, x in enumerate(annoBM5.CellType) if 'like' in x]
#ind_myeloid_benign_BM5 = [i for i, x in enumerate(annoBM5.CellType) if x in mcells]

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

#df_BM3 = pd.read_csv(ge_files +'\GSM3587998_BM3.dem.txt.gz',
                 #sep='\t')

df_BM4 = pd.read_csv(ge_files +'\GSM3588000_BM4.dem.txt.gz',
                 sep='\t')

#df_BM5 = pd.read_csv(ge_files +'\GSM3588002_BM5-34p.dem.txt.gz',
                 #sep='\t')


iGenes_419 = [i for i, x in enumerate(df_419['Gene']) if x in andata.var_names]
iGenes_420B = [i for i, x in enumerate(df_420B['Gene']) if x in andata.var_names]
#iGenes_556 = [i for i, x in enumerate(df_556['Gene']) if x in andata.var_names]
iGenes_707B = [i for i, x in enumerate(df_707B['Gene']) if x in andata.var_names]
iGenes_916 = [i for i, x in enumerate(df_916['Gene']) if x in andata.var_names]
iGenes_1012 = [i for i, x in enumerate(df_1012['Gene']) if x in andata.var_names]
iGenes_BM1 = [i for i, x in enumerate(df_BM1['Gene']) if x in andata.var_names]
iGenes_BM2 = [i for i, x in enumerate(df_BM2['Gene']) if x in andata.var_names]
#iGenes_BM3 = [i for i, x in enumerate(df_BM3['Gene']) if x in andata.var_names]
iGenes_BM4 = [i for i, x in enumerate(df_BM4['Gene']) if x in andata.var_names]
#iGenes_BM5 = [i for i, x in enumerate(df_BM5['Gene']) if x in andata.var_names]

gene_names_419 = df_419['Gene']
gene_names_420B = df_420B['Gene']
#gene_names_556 = df_556['Gene']
gene_names_707B = df_707B['Gene']
gene_names_916 = df_916['Gene']
gene_names_1012 = df_1012['Gene']
gene_names_BM1 = df_BM1['Gene']
gene_names_BM2 = df_BM2['Gene']
#gene_names_BM3 = df_BM3['Gene']
gene_names_BM4 = df_BM4['Gene']
#gene_names_BM5 = df_BM5['Gene']


#genes_t = pd.concat((gene_names_d0, gene_names_d37),axis=1, ignore_index=True)

# Selecting only Myeloid
barcodes_myeloid_419 = list(anno419d0.Cell[ind_myeloid_malign_419]) + list(anno419d0.Cell[ind_myeloid_benign_419])
barcodes_myeloid_420B = list(anno420Bd0.Cell[ind_myeloid_malign_420B]) + list(anno420Bd0.Cell[ind_myeloid_benign_420B])
#barcodes_myeloid_556 = list(anno556d0.Cell[ind_myeloid_malign_556]) + list(anno556d0.Cell[ind_myeloid_benign_556])
barcodes_myeloid_707B = list(anno707Bd0.Cell[ind_myeloid_malign_707B]) + list(anno707Bd0.Cell[ind_myeloid_benign_707B])
barcodes_myeloid_916 = list(anno916d0.Cell[ind_myeloid_malign_916]) + list(anno916d0.Cell[ind_myeloid_benign_916])
barcodes_myeloid_1012 = list(anno1012d0.Cell[ind_myeloid_malign_1012]) + list(anno1012d0.Cell[ind_myeloid_benign_1012])
barcodes_myeloid_BM1 = list(annoBM1.Cell[ind_myeloid_malign_BM1]) + list(annoBM1.Cell[ind_myeloid_benign_BM1])
barcodes_myeloid_BM2 = list(annoBM2.Cell[ind_myeloid_malign_BM2]) + list(annoBM2.Cell[ind_myeloid_benign_BM2])
#barcodes_myeloid_BM3 = list(annoBM3.Cell[ind_myeloid_malign_BM3]) + list(annoBM3.Cell[ind_myeloid_benign_BM3])
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
#m_col_BM3 = [i for i, x in enumerate(df_BM3.columns) if x in barcodes_myeloid_BM3]
m_col_BM4 = [i for i, x in enumerate(df_BM4.columns) if x in barcodes_myeloid_BM4]
#m_col_BM5 = [i for i, x in enumerate(df_BM5.columns) if x in barcodes_myeloid_BM5]

# =============================================================================
# 
# =============================================================================

df_419 = df_419.iloc[iGenes_419, m_col_419]
df_420B = df_420B.iloc[iGenes_420B, m_col_420B]
df_707B = df_707B.iloc[iGenes_707B, m_col_707B]
df_916 = df_916.iloc[iGenes_916, m_col_916]
df_1012 = df_1012.iloc[iGenes_1012, m_col_1012]
df_BM1 = df_BM1.iloc[iGenes_BM1, m_col_BM1]
df_BM2 = df_BM2.iloc[iGenes_BM2, m_col_BM2]
#df_BM3 = df_BM3.iloc[iGenes_BM3, m_col_BM3]
df_BM4 = df_BM4.iloc[iGenes_BM4, m_col_BM4]
#df_BM5 = df_BM5.iloc[iGenes_BM5, m_col_BM5]

# =============================================================================
# 
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
#barcodes_BM3 = [x for x in df_BM3.columns if 'Gene' not in x]
barcodes_BM4 = [x for x in df_BM4.columns if 'Gene' not in x]
#barcodes_BM5 = [x for x in df_BM5.columns if 'Gene' not in x]


df_train = pd.concat((df_419,df_420B, df_707B, df_916, df_1012, 
                      df_BM1, df_BM2, df_BM4),axis=1, 
                     ignore_index=True)

barcodes_train = barcodes_419 + barcodes_420B + barcodes_707B + barcodes_916 + barcodes_1012 + barcodes_BM1 + barcodes_BM2 + barcodes_BM4 

a = andata.var_names
a = pd.DataFrame(andata.var_names)

# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================

andata_train = sc.AnnData(X=df_train.T.to_numpy(), obs=barcodes_train, var = a)

andata_train.var_names = andata.var_names

andata_train.obs_names = barcodes_train

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
                                     annoBM1.Cell, annoBM2.Cell, 
                                     annoBM4.Cell)),
                          pd.concat((anno419d0.PredictionRefined, anno420Bd0.PredictionRefined,
                                     anno707Bd0.PredictionRefined, 
                                     anno916d0.PredictionRefined, anno1012d0.PredictionRefined, 
                                     annoBM1.PredictionRefined, annoBM2.PredictionRefined, 
                                     annoBM4.PredictionRefined))))

# Etiquetar cada celula con malig y benig
y_true = [barcodes2class[x] for x in andata_train.obs_names]
X = andata_train.X
X_train = X
classdict = dict(normal=0, malignant=1)
y_true_num = [classdict[x] for x in y_true] #Agregamos etiqueta 0 y 1 
y_train = np.array(y_true_num)


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


pca1=my_components_train[:,0]
idxsc1=np.where(y_train==0)
idxsc2=np.where(y_train==1)
g_normales=pca1[idxsc1]
g_malignos=pca1[idxsc2]
plt.hist(g_normales,bins=500,alpha=0.75)
plt.hist(g_malignos,bins=500,alpha=0.75)

Xtrain,Xtest,ytrain,ytest=train_test_split(X_train,y_train,test_size=0.2)
idxsc1=np.where(ytrain==0)
idxsc2=np.where(ytrain==1)
g_normales=Xtrain[idxsc1]
g_malignos=Xtrain[idxsc2]

# REDUCE DIMENSIONS WITH PRINCIPAL COMPONENT ANALYSIS (PCA)
n_input = 3252
#x_train = PCA(n_components = n_input).fit_transform(X); y_train = Y
#plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = 'tab20', s = 10)
#plt.title('Principal Component Analysis (PCA)')
#plt.xlabel("PC1")
#plt.ylabel("PC2")
n_dim=100
# REDUCE DIMENSIONS WITH AUTOENCODER
model = Sequential()
model.add(Dense(1000,       activation='elu', input_shape=(n_input,)))
model.add(Dense(800,       activation='elu'))
model.add(Dense(500,       activation='elu'))
model.add(Dense(n_dim,    activation='linear', name="bottleneck"))
model.add(Dense(500,       activation='elu'))
model.add(Dense(800,       activation='elu'))
model.add(Dense(1000,       activation='elu'))
model.add(Dense(n_input,  activation='sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = Adam())
model.fit(g_normales, g_normales, batch_size = 128, epochs = 500, verbose = 1)
encoder = Model(model.input, model.get_layer('bottleneck').output)
bottleneck_representation = encoder.predict(g_malignos)



model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, random_state = 123, 
                       perplexity = 90, n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1], c = y_train, cmap = 'tab20', s = 15,alpha=0.75)
plt.title('tSNE on Autoencoder: 8 Layers')
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")

modelUMAP = UMAP(n_neighbors = 30, min_dist = 0.3, n_components = 2)
modelUMAP.fit(bottleneck_representation)
umap=modelUMAP.transform(bottleneck_representation)
umap_coords = pd.DataFrame({'UMAP1':umap[:, 0], 'UMAP2':umap[:, 1]})
#umap_coords.to_csv('umap_coords_10X_1.3M_MouseBrain.txt', sep='\t')
plt.scatter(umap[:, 0], umap[:, 1], c = y_train, cmap = 'tab20', s = 15)
plt.title('UMAP')
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

rec_norm=np.sum(np.sqrt((Xtrain[idxsc1]-model.predict(Xtrain[idxsc1]))**2),axis=1)/X_train.shape[1]
rec_anorm=np.sum(np.sqrt((Xtrain[idxsc2]-model.predict(Xtrain[idxsc2]))**2),axis=1)/X_train.shape[1]
plt.hist(rec_norm,bins=250,alpha=0.75)
plt.hist(rec_anorm,bins=250,alpha=0.75)
plt.axvline(np.mean(rec_norm)+2*np.std(rec_norm))


idxsc1=np.where(ytest==0)
idxsc2=np.where(ytest==1)
rec_norm=np.sum(np.sqrt((Xtest[idxsc1]-model.predict(Xtest[idxsc1]))**2),axis=1)/Xtest.shape[1]
rec_anorm=np.sum(np.sqrt((Xtest[idxsc2]-model.predict(Xtest[idxsc2]))**2),axis=1)/Xtest.shape[1]
plt.figure()
plt.hist(rec_norm,bins=250,alpha=0.75)
plt.hist(rec_anorm,bins=250,alpha=0.75)

threshold=np.mean(rec_norm)+2*np.std(rec_norm)
predictions_test=model.predict(Xtest)
rec_errors=np.sum(np.sqrt((Xtest-predictions_test)**2),axis=1)/Xtest.shape[1]

rec_errors[np.where(rec_errors>threshold)]=1
rec_errors[np.where(rec_errors<threshold)]=0

confusion_matrix(rec_errors,ytest)
# =============================================================================

# =============================================================================


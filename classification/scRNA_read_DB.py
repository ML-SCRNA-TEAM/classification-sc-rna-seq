# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:30:56 2022

@author: IngBi
"""

# =============================================================================
# Importar librerias 
# =============================================================================
import pandas as pd
import pandas as pd
from collections import Counter
import numpy as np
import scanpy as sc
from matplotlib import pyplot
import matplotlib.pyplot as plt
import csv

# =============================================================================
# Lectura de anotaciones AML 
# =============================================================================

dir_anno = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

anno210Ad0 = pd.read_csv(dir_anno + '\GSM3587926_AML210A-D0.anno.txt.gz',
                         sep='\t')

# =============================================================================
# Lectura de archivos de cuentas AML 
# =============================================================================

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_419 = pd.read_csv(ge_files +'\GSM3587950_AML419A-D0.dem.txt.gz',
                 sep='\t')

# =============================================================================
# Lectura de anotaciones FA 
# =============================================================================

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_healthy1 = pd.read_csv(ge_files +'\healthy1.csv',
                 sep='\t')

# =============================================================================
# Creación de anotaciones FA (Este paso se hace por si no tienes archivos de anotaciones)
# =============================================================================

ge_files = r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion'

df_healthy1 = pd.read_csv(ge_files +'\patient1.csv',
                 sep='\t')

df_healthy1 = df_healthy1.rename({'Unnamed: 0': 'Gene'}, axis=1)

healthy1_columns = df_healthy1.columns[1:,].values

annohealthy1 = pd.DataFrame(healthy1_columns)

# Agregar nombre a columna con número de titulo
annohealthy1 = annohealthy1.rename({0: 'Cell'},
             axis=1)

annohealthy1 = pd.DataFrame(annohealthy1,columns=['Cell','PredictionRefined', 'CellType'])

annohealthy1['PredictionRefined'] = "malignant"

annohealthy1['CellType'] = 'like'

annohealthy1.to_csv(r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion\annopatient6.txt', sep="\t")

# =============================================================================
# Lectura de anotaciones FA 
# =============================================================================

annopatient1 = pd.read_csv(r'C:\Users\IngBi\OneDrive\Documentos\Animus\programacion\Spyder\Bioinformatica\data_van_galen\clasificacion\annopatient_1.txt.gz', sep='\t')





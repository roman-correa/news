import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load(name):
    return pickle.load(open(Path('artifacts') / name, 'rb'))


cols_arr = load('best_features_arr.pkl')
pp_arr = load('preprocessor_arr.pkl')
arr = pd.read_csv('arr_mede_final.csv')

NUM = ['baños', 'parqueaderos', 'espacios', 'pppz', 'garaje_bin', 'ppmc', 'axe',
       'axh', 'axa', 'new_index', 'parq2', 'pppp', 'pppp/pppz', 'pppp/ppmc', 'pppz/ppmc']
CAT = ['tipo']

X = pd.DataFrame(pp_arr.transform(
    arr[NUM+CAT]), columns=pp_arr.get_feature_names_out())

print('=== Selected features ===')
print(cols_arr)
print()
print('=== arr_mede_final shape ===', arr.shape)
print()
print('=== precio distribution ===')
print(arr['precio'].describe().apply(lambda x: f'{x:,.0f}'))
print()
print('=== tipo counts ===')
print(arr['tipo'].value_counts())
print()
print('=== nombre (barrio) unique count ===', arr['nombre'].nunique())
print()
print('=== nulls in feature cols ===')
print(arr[NUM].isnull().sum()[arr[NUM].isnull().sum() > 0])

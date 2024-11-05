import os
import pandas as pd
import numpy as np
import time  # Importar o módulo time para medir o tempo de execução
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Iniciar o timer
start_time = time.time()

# Lista de símbolos das ações
acoes = ['PETR3', 'PETR4', 'PRIO3', 'BRAV3', 'RRRP3', 'CSAN3', 'VBBR3', 'UGPA3']

# Definir o caminho relativo baseado no diretório atual
caminho_relativo = os.path.join(os.getcwd(), 'Planilhas')

# Obter a data atual para o nome do arquivo
data_atual = datetime.today().strftime('%Y-%m-%d')
nome_arquivo_excel = f'{caminho_relativo}/historico_acoes_petroleo_{data_atual}.xlsx'

# Lista para armazenar os DataFrames de cada ação
dfs = []

# Carregar dados de todas as ações e adicionar à lista dfs
for acao in acoes:
    df = pd.read_excel(nome_arquivo_excel, sheet_name=acao, decimal=',')
    df['Acao'] = acao  # Adicionar coluna para identificar a ação
    dfs.append(df)

# Combinar todos os DataFrames em um único DataFrame
data = pd.concat(dfs)

# Converter a coluna 'Data' para datetime
data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y')

# Ordenar os dados por data
data.sort_values('Data', inplace=True)

# Pivotar os dados para que cada ação tenha suas colunas próprias
data_pivot = data.pivot_table(
    index='Data',
    columns='Acao',
    values=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
)

# Preencher valores ausentes usando o método forward-fill e backward-fill
data_pivot.fillna(method='ffill', inplace=True)
data_pivot.fillna(method='bfill', inplace=True)

# Remover quaisquer linhas que ainda contenham valores NaN
data_pivot.dropna(inplace=True)

# Criar features baseadas em lags para a PETR3
def create_lag_features(data, lags, target_col):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Escolher o número de lags
num_lags = 5  # Você pode ajustar este valor

# Criar as features de lag para PETR3
target_col = ('Close', 'PETR3')
data_features = create_lag_features(data_pivot, num_lags, target_col)

# Remover valores NaN resultantes do shift
data_features.dropna(inplace=True)

# Preparar os dados para o modelo
# As features serão todas as colunas, exceto o alvo futuro
feature_cols = [col for col in data_features.columns if col != target_col]

X = data_features[feature_cols]
y = data_features[target_col]

# Dividir os dados em treino e teste (80% treino, 20% teste)
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Construir o modelo XGBoost
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5, random_state=42)

# Treinar o modelo e medir o tempo de treinamento
training_start_time = time.time()
xgb_model.fit(X_train, y_train)
training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f'Tempo de treinamento: {training_time:.2f} segundos')

# Fazer previsões no conjunto de teste
predictions = xgb_model.predict(X_test)
y_test_actual = y_test.values

# Obter as datas correspondentes ao conjunto de teste
datas_test = X_test.index

# Criar um DataFrame com os resultados do teste
resultados_teste = pd.DataFrame({
    'Data': datas_test,
    'Close_Real': y_test_actual,
    'Close_Previsto': predictions
})

# Fazer previsões futuras
dias_futuros = 5  # Manter apenas 5 dias futuros
previsoes_futuras = []

# Preparar a sequência inicial para as previsões futuras
last_data = data_features.iloc[-1].copy()

for _ in range(dias_futuros):
    # Criar um DataFrame com a última linha
    input_data = last_data.to_frame().T

    # Remover a coluna alvo
    input_data = input_data.drop(columns=target_col)

    # Fazer a previsão
    pred = xgb_model.predict(input_data)[0]

    # Armazenar a previsão
    previsoes_futuras.append(pred)

    # Atualizar last_data para a próxima previsão
    # Shift as lag features
    for lag in range(num_lags, 1, -1):
        last_data[f'{target_col}_lag_{lag}'] = last_data[f'{target_col}_lag_{lag - 1}']
    last_data[f'{target_col}_lag_1'] = pred

    # Atualizar a data
    last_data.name = last_data.name + timedelta(days=1)

# Criar um DataFrame com as previsões futuras
datas_futuras = [last_data.name + timedelta(days=i) for i in range(dias_futuros)]
resultados_futuros = pd.DataFrame({
    'Data': datas_futuras,
    'Close_Previsto': previsoes_futuras
})

# Combinar os dados reais e as previsões em um único DataFrame para plotagem
# Vamos usar os dados dos últimos 2 anos
data_inicio_plot = data_pivot.index[-(252*2)]  # Aproximadamente 252 dias úteis por ano

# Filtrar os dados reais de fechamento da PETR3
dados_reais = data_pivot[target_col].loc[data_inicio_plot:]

# Concatenar as previsões do conjunto de teste
dados_previsoes_teste = resultados_teste.set_index('Data')['Close_Previsto']

# Concatenar as previsões futuras
dados_previsoes_futuras = resultados_futuros.set_index('Data')['Close_Previsto']

# Plotar os dados
plt.figure(figsize=(14,7))
plt.plot(dados_reais.index, dados_reais.values, label='Preço Real PETR3', color='blue')
plt.plot(dados_previsoes_teste.index, dados_previsoes_teste.values, label='Previsão Modelo (Teste)', color='orange')
plt.plot(dados_previsoes_futuras.index, dados_previsoes_futuras.values, label='Previsão Futura', color='red', marker='o')

plt.xlabel('Data')
plt.ylabel('Preço de Fechamento PETR3')
plt.title('Preço Real vs Previsões do Modelo para PETR3')
plt.legend()
plt.show()

# Parar o timer e calcular o tempo total de execução
end_time = time.time()
execution_time = end_time - start_time
print(f'Tempo total de execução do script: {execution_time:.2f} segundos')

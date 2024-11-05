import os
import pandas as pd
import numpy as np
import time  # Importar o módulo time para medir o tempo de execução
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense  # Importar LSTM em vez de SimpleRNN
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

# Preparar os dados para o modelo
# Usar todas as colunas de todas as ações como features
features = data_pivot.values

# O alvo é o próximo valor de fechamento da PETR3
# Precisamos deslocar a série temporal para que o modelo aprenda a prever o próximo valor
target = data_pivot['Close', 'PETR3'].shift(-1).values[:-1]  # Remover o último valor que será NaN após o shift
features = features[:-1]  # Remover o último registro das features para alinhar com o target

# Normalizar as features e o target
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)

scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

# Definir o número de timesteps para a LSTM (aumentado significativamente)
timesteps = 200  # Você pode ajustar este valor conforme o tamanho do seu conjunto de dados

# Criar sequências para a LSTM
X = []
y = []

for i in range(timesteps, len(features_scaled)):
    X.append(features_scaled[i-timesteps:i])
    y.append(target_scaled[i])

X, y = np.array(X), np.array(y)

# Dividir os dados em treino e teste (80% treino, 20% teste)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construir o modelo LSTM com um número significativamente maior de neurônios
lstm_model = Sequential()
lstm_model.add(LSTM(units=500, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(LSTM(units=500, activation='relu'))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo e medir o tempo de treinamento
training_start_time = time.time()
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32)
training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f'Tempo de treinamento: {training_time:.2f} segundos')

# Fazer previsões no conjunto de teste
predictions_scaled = lstm_model.predict(X_test)

# Inverter a normalização para obter os valores reais
predictions = scaler_target.inverse_transform(predictions_scaled)
y_test_actual = scaler_target.inverse_transform(y_test)

# Obter as datas correspondentes ao conjunto de teste
datas_test = data_pivot.index[-len(y_test_actual):]

# Criar um DataFrame com os resultados do teste
resultados_teste = pd.DataFrame({
    'Data': datas_test,
    'Close_Real': y_test_actual.flatten(),
    'Close_Previsto': predictions.flatten()
})

# Fazer previsões futuras
dias_futuros = 5  # Manter apenas 5 dias futuros
previsoes_futuras = []

# Usar os últimos 'timesteps' dias para fazer a primeira previsão
input_seq = features_scaled[-timesteps:]

for _ in range(dias_futuros):
    # Redimensionar a sequência de entrada
    input_seq_reshaped = input_seq.reshape(1, timesteps, features_scaled.shape[1])

    # Fazer a previsão
    pred_scaled = lstm_model.predict(input_seq_reshaped)

    # Inverter a normalização para obter o valor real
    pred = scaler_target.inverse_transform(pred_scaled)

    # Armazenar a previsão
    previsoes_futuras.append(pred[0][0])

    # Atualizar a sequência de entrada com a nova previsão
    # Aqui, usamos a previsão para PETR3 e mantemos as outras features constantes
    new_feature = input_seq[-1].copy()
    index_close_petr3 = data_pivot.columns.get_loc(('Close', 'PETR3'))
    new_feature[index_close_petr3] = pred_scaled[0][0]
    input_seq = np.append(input_seq[1:], [new_feature], axis=0)

# Criar um DataFrame com as previsões futuras
datas_futuras = [data_pivot.index[-1] + timedelta(days=i+1) for i in range(dias_futuros)]
resultados_futuros = pd.DataFrame({
    'Data': datas_futuras,
    'Close_Previsto': previsoes_futuras
})

# Combinar os dados reais e as previsões em um único DataFrame para plotagem
# Vamos usar os dados dos últimos 2 anos
data_inicio_plot = data_pivot.index[-(252*2)]  # Aproximadamente 252 dias úteis por ano

# Filtrar os dados reais de fechamento da PETR3
dados_reais = data_pivot['Close', 'PETR3'].loc[data_inicio_plot:]

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

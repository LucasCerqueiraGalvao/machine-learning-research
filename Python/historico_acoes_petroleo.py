import os
import yfinance as yf
import pandas as pd

# Lista de símbolos das empresas no formato B3 (adicionando '.SA' para as ações brasileiras)
ticker_symbols = ['PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'BRAV3.SA', 'RRRP3.SA', 'CSAN3.SA', 'VBBR3.SA', 'UGPA3.SA']

# Definir o caminho relativo baseado no diretório atual
caminho_relativo = os.path.join(os.getcwd(), 'Planilhas')

# Criar o diretório, se não existir (corrigido o parâmetro para exist_ok)
os.makedirs(caminho_relativo, exist_ok=True)

# Obter a data atual para o nome do arquivo
data_atual = pd.Timestamp.today().strftime("%Y-%m-%d")

# Nome da planilha Excel incluindo a data no nome do arquivo
nome_arquivo_excel = os.path.join(caminho_relativo, f'historico_acoes_petroleo_{data_atual}.xlsx')

# Criar um ExcelWriter para salvar todas as abas no mesmo arquivo
with pd.ExcelWriter(nome_arquivo_excel) as writer:
    for ticker_symbol in ticker_symbols:
        # Obter o histórico de preços usando yfinance
        ticker = yf.Ticker(ticker_symbol)
        history_data = ticker.history(period='max').reset_index()

        # Verificar se há dados históricos disponíveis
        if not history_data.empty:
            # Remover fuso horário da coluna 'Date' e formatar a data no estilo DD/MM/YYYY
            history_data['Date'] = history_data['Date'].dt.tz_localize(None)
            history_data['Date'] = history_data['Date'].dt.strftime('%d/%m/%Y')

            # Renomear a coluna 'Date' para 'Data'
            history_data.rename(columns={'Date': 'Data'}, inplace=True)

            # Converter colunas numéricas para float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            history_data[numeric_columns] = history_data[numeric_columns].astype(float)

            # Nome da aba com o ticker (remover ".SA" para a aba)
            nome_aba = ticker_symbol.split('.')[0]

            # Salvar os dados no Excel, cada ticker em uma aba diferente
            history_data.to_excel(writer, sheet_name=nome_aba, index=False)

print(f'Dados salvos com sucesso no arquivo {nome_arquivo_excel}')

"""Ponto de entrada que mantém o encadeamento original do notebook."""
import pandas as pd
import joblib
from pathlib import Path

from preprocessing import preparar_dados
from modeling import treinar_modelo

modelo_job = Path("model/modelo_qualidade_vinhos.joblib")
dados_modelo = Path("data/WineQT.csv")



def main() -> None:
    # Verifica se o modelo já foi treinado
    if not modelo_job.exists():
        print("Modelo não encontrado. Iniciando treinamento...")

        # Carregamento dos dados
        df = pd.read_csv(dados_modelo)

        # Pré-processamento
        X_train, X_test, y_train, y_test = preparar_dados(df)

        # Treinamento
        modelo = treinar_modelo(
            X_train, X_test, y_train, y_test
        )

        # Salvar modelo
        joblib.dump(modelo, modelo_job)

        print("Modelo treinado e salvo com sucesso!")
    else:

        print("Modelo encontrado. Carregando...")

        modelo = joblib.load(modelo_job)

        print("Por favor, Informe as características do novo vinho")

        # Aqui eu capturo as colunas do modelo que são usados - Validar pois parece mais fácil que fazer um a um
        colunas = modelo.feature_names_in_

        dados = {}

        for coluna in colunas:
            while True:
                try:
                    valor = float(
                        input(f"{coluna}: ").replace(",", ".")
                    )
                    dados[coluna] = [valor]
                    break

                except ValueError:
                    print("Valor inválido. Digite um número.")

        # Criar DataFrame com os novos dados
        novo_vinho = pd.DataFrame(dados)

        # Realizar classificação
        previsao = modelo.predict(novo_vinho)[0]

        probabilidade = modelo.predict_proba(novo_vinho)[0, 1]

        print("\nSegue abaixo o resultado")

        if previsao == 1:
            print("Vinho de ALTA qualidade, deguste com vontade!")
        else:
            print("Vinho de BAIXA/MÉDIA qualidade, espere sua dor de cabeça ahahahaha")

        print(
            f"Probabilidade estimada de alta qualidade: "
            f"{probabilidade:.2%}"
        )
    


if __name__ == "__main__":
    main()

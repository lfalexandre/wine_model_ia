"""Pré-processamento mantendo a sequência usada no notebook."""
import pandas as pd
from sklearn.model_selection import train_test_split


def preparar_dados(df: pd.DataFrame):
    df = pd.read_csv('data/WineQT.csv', sep=",")

    # Retirando da base o campo Id.
    df.drop(["Id"], axis=1, inplace=True)

    # Criação da coluna binária e retirada da base de qualidade.
    # 1 = boa qualidade (nota >= 7); 0 = baixa/média qualidade.
    df["quality_bin"] = (df["quality"] >= 7).astype(int)
    df.drop("quality", axis=1, inplace=True)

    # Separação das bases de x e y.
    x = df.drop("quality_bin", axis=1)
    y = df["quality_bin"]

    # Separação das bases de treino e teste.
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

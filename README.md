# Classificação de Qualidade de Vinhos

## Sobre o projeto

Projeto desenvolvido para o **Tech Challenge – Fase 2**, com o objetivo de classificar vinhos como de **alta qualidade (nota ≥ 7)** ou **baixa/média qualidade (nota < 7)**, utilizando características físico-químicas.

Foram avaliados quatro algoritmos de Machine Learning: KNN, Regressão Logística, Random Forest e Gaussian Naive Bayes.

O **Random Forest** apresentou o melhor equilíbrio entre precisão e recall.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

# Entregas Teóricas
Temos um arquivo PPTx com a Apresentação realizada no vídeo e o notebook com todo o estudo compilado para a criação do projeto.

```text
├── ppt/
│   └── Apresentação - ML techchallange.pptx
├── notebooks/
│   ├── wine_analysis.ipynb
```

## Estrutura do projeto

```text
├── data/
│   └── WineQT.csv
├── src/
│   ├── preprocessing.py
│   └── modeling.py
│   └── train.py
├── main.py
├── requirements.txt
└── README.md
```

## Como executar

1. Instale as dependências necessárias.
2. Execute o projeto pelo terminal:

```bash
python main.py
```

**Primeira execução:** o sistema prepara os dados, treina o Random Forest e salva o modelo em `modelo_qualidade_vinhos.joblib`.

**Próximas execuções:** o modelo salvo é carregado automaticamente e o usuário pode informar as características de um novo vinho para obter sua classificação.

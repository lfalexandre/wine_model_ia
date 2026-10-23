from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def treinar_modelo(X_train, X_test, y_train, y_test):
    #Definindo uqal o modelo
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )

    # Treinar o modelo
    modelo.fit(X_train, y_train)

    # Avaliar no conjunto de teste
    y_pred = modelo.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))

    return modelo
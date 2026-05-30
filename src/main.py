from preprocessing import preprocess
from models import *
import pandas as pd

X_train, X_val, y_train, Y_val, X_test = preprocess()

#Treina o modelo KNN, que foi escolher por ter a melhor acurácia
knn_model, knn_acc, knn_f1 = train_knn(
    X_train,
    y_train,
    X_val,
    Y_val
)

print("\n===== RESULTADOS DO MODELO =====")
print(f"Acurácia: {knn_acc:.4f}")
print(f"F1-score: {knn_f1:.4f}")


#Retorna a probabilidades de cada classe, como [0.20, 0.80]
probabilities = knn_model.predict_proba(X_test)

#Seleciona aenas a probabilidade da classe winner ser 1
winner_probability = probabilities[:, 1]

#Carrega o dataframe com original de teste, para recuperar os nomes dos países
test_df = pd.read_csv("../data/test.csv")

#Cria dataframe com o país e probabilidade de vencer
results = pd.DataFrame({
    "team": test_df["team_name"],
    "winning_probability": winner_probability
})

#Agrupa por país e calcula a média das probabilidades
results = (
    results
    .groupby("team", as_index=False)
    .mean()
)

#Ordena do maior para o menor
results = results.sort_values(by="winning_probability", ascending=False)

print(results.head(20))

print("\n===== COMPARAÇÃO DOS MODELOS =====")

print(
    f"KNN -> "
    f"Accuracy: {knn_acc:.4f} | "
    f"F1: {knn_f1:.4f}"
)
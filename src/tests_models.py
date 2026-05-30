from src.models import train_knn, train_decision_tree, train_naive_bayes
from src.preprocessing import preprocess

X_train, X_val, Y_train, Y_val, X_test = preprocess()

knn_model, knn_acc, knn_f1 = train_knn(
    X_train, Y_train, X_val, Y_val
)

tree_model, tree_acc, tree_f1 = train_decision_tree(
    X_train, Y_train, X_val, Y_val
)

nb_model, nb_acc, nb_f1 = train_naive_bayes(
    X_train, Y_train, X_val, Y_val,
)

print("\n===== RESULTADOS FINAIS DO MELHOR MODELO =====")

print(
    f"KNN -> "
    f"Accuracy: {knn_acc:.3f} | "
    f"F1: {knn_f1:.3f}"
)

print(
    f"Decision Tree -> "
    f"Accuracy: {tree_acc:.3f} | "
    f"F1: {tree_f1:.3f}"
)

print(
    f"Naive Bayes -> "
    f"Accuracy: {nb_acc:.3f} | "
    f"F1: {nb_f1:.3f}"
)

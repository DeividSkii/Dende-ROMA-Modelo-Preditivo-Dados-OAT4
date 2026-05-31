from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def train_knn(X_train, y_train, X_val, y_val):

    """
    Treina diferentes configurações de algoritmo KNN, variando o n de vizinhps.

    O melhor modelo é escolhido com base na melhor acurácia
    no conjunto da validação
    """
    best_acc = 0
    best_model = None
    best_k = 0
    best_f1 = 0

    #Lista dos Ks testados
    for k in [3, 5, 7, 9, 11, 15, 17, 19, 21, 25]:
        model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train, y_train)

        prediction = model.predict(X_val)

        accuracy = accuracy_score(y_val, prediction)

        f1 = f1_score(
            y_val,
            prediction
        )

        print(f"O k foi {k}, a acurácia foi {accuracy} e a f1 foi {f1:.3f}")
        #Salva melhor modelo encontrado
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            best_k = k
            best_f1 = f1



    return best_model, best_acc, best_f1

def train_decision_tree(X_train, y_train, X_val, y_val):

    """
    Treina diferentes configurações do algoritmo Decision Tree
    variando a profundidade máxima.

    O melhor modelo é escolhi com base na melhor acurácia
    """
    best_acc = 0
    best_model = None
    best_depth = 0
    best_f1 = 0

    #Profundidade testadas
    for depth in [3, 5, 7, 10, None]:

        model = DecisionTreeClassifier(max_depth=depth, random_state=67)

        model.fit(X_train, y_train)

        prediction = model.predict(X_val)

        accuracy = accuracy_score(y_val, prediction)

        f1 = f1_score(y_val,prediction)

        print(f"O depth foi {depth}, a acurácia foi {accuracy} e a f1 foi {f1:.3f}")

        #Atualiza melhor modelo
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            best_depth = depth
            best_f1 = f1

    return best_model, best_acc, best_f1

def train_naive_bayes(X_train, y_train, X_val, y_val):

    model = GaussianNB()

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    accuracy = accuracy_score(y_val, prediction)

    f1 = f1_score(
        y_val,
        prediction
    )

    return model, accuracy, f1
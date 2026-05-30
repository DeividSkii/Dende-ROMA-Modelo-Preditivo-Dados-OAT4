import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Função para carregar os dados
def load_data():
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    return train_data, test_data


#Validação do dataset
def validar_dataset(df):
    print("Valores nulos:")
    print(df.isnull().sum())

    print("Valores duplicados:")
    print(df.duplicated().sum())

#Separação das colunas categóricas e numéricas
def separar_colunas(df):
    colunas_categoricas = df.select_dtypes(include=["object", "string"]).columns.tolist()

    colunas_numericas = df.select_dtypes(exclude="object").columns.tolist()

    print(f"Colunas Categóricas: {colunas_categoricas}")
    print(f"Colunas Numéricas: {colunas_numericas}")

    return colunas_categoricas, colunas_numericas

#One hot encoder
def encoder_categoricas(train, test):
    """
    Foi utilizado o one hot encoding em vez do label encoder para evitar viés de ordem,
    já que não há hierarquia.
    """

    #Seleciona as colunas categóricas do datatest
    colunas_categoricas = [
        "team_name",
        "country_code",
        "confederation"
    ]

    #Aplica o one hot no treino
    train = pd.get_dummies(
        train,
        columns=colunas_categoricas,
        drop_first=True
    )

    #Aplica one hot no teste
    test = pd.get_dummies(
        test,
        columns=colunas_categoricas,
        drop_first=True
    )

    #Armazena a variável alvo temporariamente
    winner = train["winner"]

    #Remove target antes do alinhamento
    train = train.drop("winner", axis=1)

    #Garante as mesmas features no treino e no teste
    train, test = train.align(
        test,
        join="left",
        axis=1,
        fill_value=0
    )

    #Reinsere a variável alvo
    train["winner"] = winner

    return train, test

#Padronização dos dados numéricos
def standard_scaler(train, test):

    #Remove variável alvo das features
    X_train = train.drop("winner", axis=1)

    #Define o target
    y_train = train["winner"]

    #Remove o winner do teste, caso exista
    X_test = test.drop("winner", axis=1, errors="ignore")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled

def preprocess(debug=False):

    """
    Executa o pipeline de pré-processamento:

    1 - Carrega os dados
    2 - Faz encoding das categorias
    3 - Padroniza os dados
    4 - Separa X e Y
    """
    train_data, test_data = load_data()

    #Executa validações opcionais para debug
    if debug:
        validar_dataset(train_data)
        separar_colunas(train_data)

    train, test = encoder_categoricas(
        train_data,
        test_data
    )


    X_train, y_train, X_test = standard_scaler(
        train,
        test
    )

    return X_train, y_train, X_test

X_train, y_train, X_test = preprocess()

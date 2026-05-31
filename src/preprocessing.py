import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#Função para tratar dados nulos ou duplicados
def limpar_dataset(df):
    """
    Trata valores ausentes e duplicados no dataset, caso tenha
    """

    #Remove registros duplicados
    df = df.drop_duplicates()

    #Verifica percentual de nulos
    porcentagem_nulos = df.isnull().mean()

    #Remove colunas com muitos nulos
    colunas_remover = porcentagem_nulos[
        porcentagem_nulos > 0.5
    ].index

    df = df.drop(columns=colunas_remover)

    return df


#One hot encoder
def encoder_categoricas(train, test):
    """
    Foi utilizado o one hot encoding em vez do label encoder para evitar viés de ordem,
    já que não há hierarquia.
    """

    #Seleciona as colunas categóricas do datatest
    colunas_categoricas = [
        "confederation",
        "country_code",
        "team_name"
    ]

    #Aplica o one hot no treino
    train = pd.get_dummies(
        train,
        columns=colunas_categoricas
    )

    #Aplica one hot no teste
    test = pd.get_dummies(
        test,
        columns=colunas_categoricas
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

    train["winner"] = winner

    return train, test

#Padronização dos dados numéricos
def standard_scaler(X_train, X_val, X_test):

    scaler = StandardScaler()

    #aprende a média e desvio padrao do treino
    X_train_scaled = scaler.fit_transform(X_train)

    #Aplica a transformação nos demais conjuntos
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled

def preprocess(debug=False):

    """
    Executa o pipeline de pré-processamento:

    1 - Carrega os dados
    2 - Faz limpeza e encoding das categorias
    3 - Padroniza os dados
    4 - Separa X e Y
    """
    train_data, test_data = load_data()

    #Tratamento de valores nulos e duplicados
    train_data = limpar_dataset(train_data)
    test_data = limpar_dataset(test_data)

    #Executa validações opcionais para debug
    if debug:
        validar_dataset(train_data)
        separar_colunas(train_data)

    #Aplica encoding nas categorias
    train, test = encoder_categoricas(
        train_data,
        test_data
    )

    #Separa variável alvo e feature
    X = train.drop("winner", axis=1)
    y = train["winner"]

    #Remove o winner do teste caso ele exista
    X_test = test.drop("winner", axis=1, errors="ignore")


    #Divide o treino e a validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Faz a padronização dos dados
    X_train_scaled, X_val_scaled, X_test_scaled = standard_scaler(
        X_train, X_val, X_test

    )

    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled

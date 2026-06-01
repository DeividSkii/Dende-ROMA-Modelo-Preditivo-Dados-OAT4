import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

class ExploradorDados:
    """Executa a análise exploratória de dados (EDA) de forma clara e reutilizável."""

    def __init__(self, caminho_arquivo: str, coluna_alvo: str = "winner"):
        self.caminho_arquivo = Path(caminho_arquivo)
        self.coluna_alvo = coluna_alvo
        self.dataframe: pd.DataFrame | None = None
        self.colunas_categoricas: list[str] = []
        self.colunas_numericas: list[str] = []

    def carregar_dados(self) -> pd.DataFrame:
        """Carrega o dataset para memória e retorna o DataFrame."""
        if self.dataframe is None:
            self.dataframe = pd.read_csv(self.caminho_arquivo)
        return self.dataframe

    def relatorio_geral(self) -> dict:
        """Exibe informações gerais do dataset: dimensões, tipos, valores ausentes e valores únicos."""
        df = self.carregar_dados()
        relatorio = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "valores_faltantes": df.isna().sum().to_dict(),
            "valores_unicos": df.nunique().to_dict(),
        }

        print("\n=== Visão Geral do Dataset ===")
        print(f"Linhas: {relatorio['shape'][0]}")
        print(f"Colunas: {relatorio['shape'][1]}")
        print("\nTipos de cada coluna:")
        print(df.dtypes)
        print("\nValores faltantes por coluna:")
        print(df.isna().sum())

        return relatorio

    def separar_variaveis(self) -> tuple[list[str], list[str]]:
        """Separa automaticamente as colunas categóricas das colunas numéricas."""
        df = self.carregar_dados()
        self.colunas_categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.colunas_numericas = df.select_dtypes(include=["number"]).columns.tolist()

        print("\n=== Separação de Variáveis ===")
        print(f"Colunas categóricas ({len(self.colunas_categoricas)}): {self.colunas_categoricas}")
        print(f"Colunas numéricas ({len(self.colunas_numericas)}): {self.colunas_numericas}")

        return self.colunas_categoricas, self.colunas_numericas

    def criar_agrupamentos(self) -> dict:
        """Gera agrupamentos que conectam a variável alvo com colunas estratégicas."""
        df = self.carregar_dados()

        resumo_confederacao = (
            df.groupby("confederation")[self.coluna_alvo]
            .agg(total_times="count", taxa_vitoria="mean")
            .sort_values("taxa_vitoria", ascending=False)
            .reset_index()
        )

        intervalos_rank = pd.cut(df["fifa_rank"], bins=[0, 10, 20, 50, 100, 200], include_lowest=True)
        resumo_rank = (
            df.groupby(intervalos_rank)[self.coluna_alvo]
            .agg(total_times="count", taxa_vitoria="mean")
            .reset_index()
        )

        quartis_valor_mercado = pd.qcut(df["market_value_million_eur"], q=4, duplicates="drop")
        resumo_valor = (
            df.groupby(quartis_valor_mercado)[self.coluna_alvo]
            .agg(total_times="count", taxa_vitoria="mean", valor_medio="mean")
            .reset_index()
        )

        print("\n=== Agrupamento por Confederação ===")
        print(resumo_confederacao)
        print("\n=== Agrupamento por Faixa de Ranking FIFA ===")
        print(resumo_rank)
        print("\n=== Agrupamento por Quartil de Valor de Mercado ===")
        print(resumo_valor)

        return {
            "confederation": resumo_confederacao,
            "fifa_rank_bins": resumo_rank,
            "market_value_quartiles": resumo_valor,
        }

    def _top_variaveis_correlacionadas(self, top_n: int = 10) -> list[str]:
        """Retorna as top N variáveis numéricas mais correlacionadas com a coluna alvo."""
        df = self.carregar_dados()
        if not self.colunas_numericas:
            self.separar_variaveis()

        correlacoes = (
            df[self.colunas_numericas].corr()[self.coluna_alvo].abs().sort_values(ascending=False)
        )

        return correlacoes.drop(self.coluna_alvo, errors="ignore").head(top_n).index.tolist()

    def plotar_balanceamento_alvo(self) -> None:
        """Plota o balanceamento da variável alvo winner."""
        df = self.carregar_dados()
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.coluna_alvo, data=df, color="#5A9BD5")
        plt.title("Balanceamento da variável alvo: winner")
        plt.xlabel("Winner")
        plt.ylabel("Número de observações")
        plt.tight_layout()
        plt.show()

    def plotar_boxplot_valor_mercado(self) -> None:
        """Plota a distribuição do valor de mercado por resultado de vitória."""
        df = self.carregar_dados()
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.coluna_alvo, y="market_value_million_eur", data=df, color="#8CBFE9")
        plt.title("Boxplot de valor de mercado por resultado de 'winner'")
        plt.xlabel("Winner")
        plt.ylabel("Valor de Mercado (milhões EUR)")
        plt.tight_layout()
        plt.show()

    def plotar_heatmap_correlacao(self, top_n: int = 10) -> None:
        """Desenha um heatmap das variáveis numéricas mais correlacionadas com o alvo."""
        df = self.carregar_dados()
        if not self.colunas_numericas:
            self.separar_variaveis()

        top_variaveis = self._top_variaveis_correlacionadas(top_n=top_n)
        variaveis_heatmap = top_variaveis + [self.coluna_alvo]
        matriz_correlacao = df[variaveis_heatmap].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz_correlacao, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title(f"Heatmap: Top {top_n} variáveis numéricas mais correlacionadas com '{self.coluna_alvo}'")
        plt.tight_layout()
        plt.show()

    def executar_analise_completa(self) -> dict:
        """Executa todas as etapas principais de EDA e retorna um resumo final."""
        relatorio = self.relatorio_geral()
        self.separar_variaveis()
        agrupamentos = self.criar_agrupamentos()
        return {
            "relatorio": relatorio,
            "colunas_categoricas": self.colunas_categoricas,
            "colunas_numericas": self.colunas_numericas,
            "agrupamentos": agrupamentos,
        }


if __name__ == "__main__":
    explorador = ExploradorDados("data/train.csv")
    explorador.executar_analise_completa()
    explorador.plotar_balanceamento_alvo()
    explorador.plotar_boxplot_valor_mercado()
    explorador.plotar_heatmap_correlacao()

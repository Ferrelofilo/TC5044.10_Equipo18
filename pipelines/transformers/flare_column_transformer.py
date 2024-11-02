from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def get_flare_transformer():
    """
    Crea un pipeline de transformación para los datos, que incluye codificación ordinal
    para columnas categóricas específicas. Permite agregar más transformaciones en el futuro.

    Returns:
        Pipeline: Objeto de pipeline de transformación con codificación ordinal aplicada a columnas seleccionadas.
    """
    columns_to_encode = [
        "modified Zurich class",
        "largest spot size",
        "spot distribution",
    ]
    pipeline = Pipeline(
        [
            (
                "encode",
                make_column_transformer((OrdinalEncoder(), columns_to_encode), remainder="passthrough"),
            ),
            # ('scale', MinMaxScaler()), Podemos ir agregando más transformaciones de ser necesario.
        ]
    )
    return pipeline

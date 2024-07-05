import pandas as pd
import requests

from genetic import evolution_fns


def test_overall_evolution():
    test_model_params = [
        {"KMeans": ["n_clusters", "max_iter", "tol"]},
        {"KMedoids": ["n_clusters", "metric_1", "method", "max_iter"]},
        {"DBSCAN": ["eps", "min_samples", "metric_1"]},
        {"HDBSCAN": ["metric_2", "min_samples", "eps"]},
    ]
    test_param_values = {
        "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_iter": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "metric_1": [
            "euclidean",
            "cosine",
            "haversine",
            "l2",
            "cityblock",
            "l1",
            "manhattan",
        ],
        "metric_2": [
            "l2",
            "canberra",
            "manhattan",
            "euclidean",
            "braycurtis",
            "chebyshev",
            "hamming",
        ],
        "method": ["alternate", "pam"],
        "eps": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 3, 4],
        "min_samples": [3, 4, 5, 6, 7, 8, 9, 10],
    }
    URL = 'https://api.nationalgrideso.com/api/3/action/datastore_search_sql?sql=SELECT * FROM "bb44a1b5-75b1-4db2-8491-257f23385006"'
    response = requests.get(URL).json()
    test_df = pd.json_normalize(
        response["result"]["records"],
        meta=[
            "IFA_FLOW",
            "TSD",
            "VIKING_FLOW",
            "IFA2_FLOW",
            "EMBEDDED_WIND_GENERATION",
            "ND",
            "MOYLE_FLOW",
            "NEMO_FLOW",
            "ELECLINK_FLOW",
            "PUMP_STORAGE_PUMPING",
            "EMBEDDED_WIND_CAPACITY",
            "SETTLEMENT_DATE",
            "ENGLAND_WALES_DEMAND",
            "EMBEDDED_SOLAR_CAPACITY",
            "SCOTTISH_TRANSFER",
            "NON_BM_STOR",
            "_FULL_TEXT",
            "SETTLEMENT_PERIOD",
            "EAST_WEST_FLOW",
            "NSL_FLOW",
            "BRITNED_FLOW",
            "_ID",
            "EMBEDDED_SOLAR_GENERATION",
        ],
    )
    test_df_drop = test_df.drop(
        ["_full_text", "NON_BM_STOR", "_id", "SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
        axis=1,
    )
    test_df_transpose = test_df_drop.transpose()

    returns = evolution_fns.evolution(
        model_params=test_model_params,
        param_values=test_param_values,
        init_population_num=20,
        df=test_df_transpose,
        selection_param=0.5,
        crossover_repeat=5,
        mutation_repeat=5,
        cutoff_score=1,
    )

    assert returns[0][1] > 0 and returns[0][1] <= 1

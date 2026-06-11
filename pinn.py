import pandas as pd
import numpy as np


class ElectrolyteRecommendationEngine:

    def __init__(self, ranked_df):

        self.df = ranked_df.copy()

    def recommend(
        self,
        sigma,
        voltage,
        viscosity,
        thermal,
        top_k=3
    ):

        df = self.df.copy()

        # Distance from desired target

        df["Distance"] = np.sqrt(

            ((df["Conductivity"] - sigma) / sigma) ** 2

            +

            ((df["Stability"] - voltage) / voltage) ** 2

            +

            ((df["Viscosity"] - viscosity) / viscosity) ** 2

            +

            ((df["Thermal"] - thermal) / thermal) ** 2
        )

        return (
            df
            .sort_values(
                "Distance"
            )
            .head(top_k)
        )
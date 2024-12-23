import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class AIModel:
    def __init__(self, algae, debris):
        self.algae = algae.data
        self.debris = debris.data
        self.model = None

    def train_model(self):
        # 데이터 병합
        merged_data = pd.merge(self.algae, self.debris, on=["lon", "lat"], suffixes=("_algae", "_debris"))
        X = merged_data[["lon", "lat", "density_algae"]]
        y = merged_data["density_debris"]

        # 데이터 분리 및 모델 훈련
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        print("AI Model trained successfully!")

    def predict_new_data(self, new_data):
        df_new = pd.DataFrame(new_data)
        predictions = self.model.predict(df_new)
        df_new["predicted_density_debris"] = predictions
        return df_new

from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

@trigger(events=['s3'])
@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2', 'lightgbm' : '3.3.5','xgboost' : '1.7.4'})
class TaxiFarePrediction(FlowSpec):

    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):

        obviously_bad_data_filters = [
            df.fare_amount > 0,         
            df.trip_distance <= 100,    
            df.trip_distance > 0,
            df.passenger_count > 0,
            df.mta_tax > 0,
            df.tip_amount >= 0,
            df.tolls_amount >= 0,
            df.total_amount > 0,
            df.PULocationID !=df.DOLocationID,
            df.hour > 0
        ]

        for f in obviously_bad_data_filters:
            df = df[f]

        
        return df

    @step
    def start(self):

        import pandas as pd
        from sklearn.model_selection import train_test_split

        self.df = self.transform_features(pd.read_parquet(self.data_url))
 
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.model_linear_reg,self.model_elasticnet, self.model_bayesianridge,self.model_xgboost,self.model_lightgbm)
        
    @step
    def model_linear_reg(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        

        self.reg = LinearRegression()
        self.scores = cross_val_score(self.reg, self.X, self.y, cv=5,scoring='r2')
        print("scores LR", self.scores)
        self.next(self.choose_model)

    @step
    def model_elasticnet(self):
        from sklearn.linear_model import ElasticNet
        from sklearn.model_selection import cross_val_score
        
        self.reg = ElasticNet()
        self.scores = cross_val_score(self.reg, self.X, self.y, cv=5,scoring='r2')
        print("scores EN", self.scores)
        self.next(self.choose_model) 




    @step
    def model_bayesianridge(self):
        from sklearn.linear_model import BayesianRidge
        from sklearn.model_selection import cross_val_score
        
        
        self.reg = BayesianRidge()
        self.scores = cross_val_score(self.reg, self.X, self.y, cv=5,scoring='r2')
        print("scores BR", self.scores)
        self.next(self.choose_model)
    

    @step
    def model_xgboost(self):
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
        
       
        self.reg = XGBRegressor() 
        self.scores = cross_val_score(self.reg, self.X, self.y, cv=5,scoring='r2')
        print("scores XG", self.scores)
        self.next(self.choose_model)

    @step
    def model_lightgbm(self):
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import cross_val_score
        
       
        self.reg = LGBMRegressor()
        self.scores = cross_val_score(self.reg, self.X, self.y, cv=5,scoring='r2')
        print("scores LG", self.scores)
        self.next(self.choose_model)

    @card(type="corise")
    @step
    def choose_model(self, inputs):
        """
        find 'best' model
        """
        import numpy as np

        def score(inp):
            return inp.reg, np.mean(inp.scores)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        print(self.results)
        self.model = self.results[0][0]
        current.card.append(Markdown("# Taxi Fare Prediction Multiple Model Results"))
        current.card.append(Artifact(self.results[0][1],self.results[0][0]))
        current.card.append(Artifact(self.results[1][1],self.results[1][0]))
        current.card.append(Artifact(self.results[2][1],self.results[2][0]))
        current.card.append(Artifact(self.results[3][1],self.results[3][0]))
        current.card.append(Artifact(self.results[4][1],self.results[4][0]))
        self.next(self.end)

    
    @step
    def end(self):
        """
        End of flow, yo!
        """
        print("Scores:")
        print("\n".join("%s %f" % res for res in self.results))
        



if __name__ == "__main__":
    TaxiFarePrediction()

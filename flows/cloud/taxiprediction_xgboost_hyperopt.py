from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

@trigger(events=['s3'])
@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2', 'lightgbm' : '3.3.5','xgboost' : '1.7.4'})
class TaxiFarePrediction(FlowSpec):

    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):

    

        # TODO: 
            # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
            # Understand what is happening.
            # Revisit task 1 and think about what might go in this function.
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

        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
        # This is a simple/naive way to do this, and is meant to keep this example simple, to focus learning on deploying Metaflow flows.
        # In practice, you want split time series data in more sophisticated ways and run backtests. 
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.make_grid)

    @step
    def make_grid(self):
        from sklearn.model_selection import ParameterGrid
        param_values = {'n_estimators': [100, 250, 500],
                        'max_depth': [4, 5, 6],
                        'learning_rate': [0.05, 0.1, 0.25]}

        self.grid_points = list(
            ParameterGrid(param_values)
        )
        # evaluate each in cross product of ParameterGrid.
        self.next(self.model_xgboost, 
                  foreach='grid_points')
    


    @step
    def model_xgboost(self):
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
        
        # TODO: Play around with the model if you are feeling it.
        self.reg = XGBRegressor(**self.input) 
        self.scores = cross_val_score(self.reg, self.X, self.y, cv=5,scoring='r2')
        self.next(self.choose_model)

   

    @step
    def choose_model(self, inputs):
        """
        find 'best' model
        """
        import numpy as np

        def score(inp):
            return inp.reg,\
                   np.mean(inp.scores)

            
        self.results = sorted(map(score, inputs), key=lambda x: -x[1]) 
        self.model = self.results[0][0]
        
        self.next(self.end)

    @card(type="corise")
    @step
    def end(self):
        """
        End of flow!
        """
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        current.card.append(Markdown("Best Model"))
        current.card.append(Artifact(self.model))
        current.card.append(Markdown("Score of Best Model"))
        current.card.append(Artifact(self.results[0][1]))


  


if __name__ == "__main__":
    TaxiFarePrediction()

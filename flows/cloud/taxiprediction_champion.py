from metaflow import FlowSpec, step, card, conda_base, project, current, Parameter, Flow, trigger, retry, timeout,catch
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

@trigger(events=['s3'])
@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2'})
@project(name="taxi_fare_prediction")
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

    @catch(var="read_failure")
    @retry(times=4)
    @timeout(minutes=10)
    @step
    def start(self):

        import pandas as pd
        from sklearn.model_selection import train_test_split

        self.df = self.transform_features(pd.read_parquet(self.data_url))

        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.linear_model)

    @step
    def linear_model(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

        self.next(self.validate)
                
    
    def gather_sibling_flow_run_results(self):

        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow 
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅" 
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [Markdown(icon), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(score), Markdown(msg)]
                rows.append(row)
            else:
                rows.append([Markdown("✅"), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(str(self.scores.mean())), Markdown("This run...")])
        return rows
                
    
    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score
        self.scores = cross_val_score(self.model, self.X, self.y, cv=5,scoring='r2')
        current.card.append(Markdown("# Taxi Fare Prediction Champion Results"))
        current.card.append(Table(self.gather_sibling_flow_run_results(), headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"]))
        self.next(self.end)


    @step
    def end(self):
        self.model_type = "baseline"
        print("Score = %s" % self.scores.mean())


if __name__ == "__main__":
    TaxiFarePrediction()

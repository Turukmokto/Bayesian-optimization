import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import pandas as pd

def grafics_objective_function_with_steps(df_random, step_update):
    fig = make_subplots()
    fig.append_trace(go.Scatter(x=[i for i in range(0, len(step_update[0]))], y=df_random['f1'], mode='lines')
                     , row=1, col=1)
    fig['data'][0]['line']['color'] = "#00ff00"
    fig.show()


def grafics_comparing_with_random_search(df_random, df_surrogate):
    pca = PCA(n_components=1)
    random = pca.fit_transform(df_random.drop('f1', axis=1))
    random_df = pd.DataFrame(data=random, columns=['hyper_params'])
    random_df = pd.concat([random_df, df_random['f1']], axis=1)
    pca = PCA(n_components=1)
    bays = pca.fit_transform(df_surrogate.drop('f1', axis=1))
    bays_df = pd.DataFrame(data=bays, columns=['hyper_params'])
    bays_df = pd.concat([bays_df, df_surrogate['f1']], axis=1)
    fig = make_subplots()
    fig.append_trace(
        go.Scatter(x=random_df['hyper_params'], y=random_df['f1'], mode='markers'), row=1,
        col=1)
    fig.append_trace(
        go.Scatter(x=bays_df['hyper_params'], y=bays_df['f1'], mode='markers'), row=1,
        col=1)
    fig['data'][0]['line']['color'] = "#00ff00"
    fig.show()


def grafics_check_with_real_scores(primary_df_pred, primary_df_real):
    visual_df = primary_df_pred[(primary_df_pred['criterion'] == 0) & (primary_df_pred['min_samples_split'] == 2) & (
            primary_df_pred['min_samples_leaf'] == 2)]
    visual_df_real = primary_df_real[
        (primary_df_real['criterion'] == 0) & (primary_df_real['min_samples_split'] == 2) & (
                primary_df_real['min_samples_leaf'] == 2)]
    fig = make_subplots()
    fig.add_traces(go.Scatter(x=visual_df['max_depth'], y=visual_df['f1'], mode='lines'))
    fig.add_traces(go.Scatter(x=visual_df_real['max_depth'], y=visual_df_real['f1'], mode='lines'))
    fig['data'][0]['line']['color'] = "#00ff00"
    fig.show()

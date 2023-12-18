import plotly.graph_objects as go

from models.random_forest import RandomForest


class Visualisation:

    @staticmethod
    def visualise_best_models(models: list[list[RandomForest | float]], plot_title: str = ''):
        models = list(reversed(models))
        x = []
        y = []
        text = []
        for i in range(len(models)):
            x.append(f'M = {models[i][0].nb_trees}, '
                     f'L1 = {models[i][0].max_nb_dim_to_check}, '
                     f'L2 = {models[i][0].max_nb_thresholds}')
            y.append(str(models[i][1]))
            text.append(str(models[i][2]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 hovertext=text,
                                 mode='markers',
                                 name='random forest'))
        fig.update_layout(title=plot_title,
                          xaxis_title='models',
                          yaxis_title='accuracy on valid set')
        fig.update_traces(hoverinfo='all',
                          hovertemplate='Model: %{x}<br>'
                                        'Accuracy on valid set: %{y}<br>'
                                        'Accuracy on test set: %{hovertext}')
        fig.show()

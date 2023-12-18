import plotly.graph_objects as go

from models.gradient_boosting import GradientBoosting


class Visualisation:

    @staticmethod
    def visualise_best_models(models: list[list[GradientBoosting | float]], plot_title: str = ''):
        x = []
        y = []
        text = []
        for i in range(len(models)):
            x.append(f'Number of weak learners = {models[i][0].number_of_weak_learners}, '
                     f'weight of weak learners = {models[i][0].weight_of_weak_learners}')
            y.append(str(models[i][1]))
            text.append(str(models[i][2]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 hovertext=text,
                                 mode='markers',
                                 name='gradient boosting'))
        fig.update_layout(title=plot_title,
                          xaxis_title='models',
                          yaxis_title='MSE on valid set')
        fig.update_traces(hoverinfo='all',
                          hovertemplate='Model: %{x}<br>'
                                        'MSE on valid set: %{y}<br>'
                                        'MSE on test set: %{hovertext}')
        fig.show()

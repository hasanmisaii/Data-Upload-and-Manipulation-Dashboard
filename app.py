import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import re
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(
        html.H1("Data Upload and Manipulation Dashboard", style={'textAlign': 'center', 'color': 'white'}),
        style={'backgroundColor': 'blue', 'padding': '10px'}
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '90%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '75px',
            'backgroundColor': 'green',  # Add green background color
            'color': 'white'  # Add white text color for better readability
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='column-selector'),
    html.Div(id='column-stats'),
    html.Div(id='column-graph'),
    html.Div(id='column-filter'),
    html.H4("Select Plot Type"),
    dcc.Dropdown(
        id='plot-type-dropdown',
        options=[
            {'label': 'Scatter', 'value': 'scatter'},
            {'label': 'Line', 'value': 'line'},
            {'label': 'Bar', 'value': 'bar'},
            {'label': 'Pie', 'value': 'pie'}
        ],
        value='scatter'
    ),
    html.Button('Plot', id='plot-button', style={'background-color': 'red', 'color': 'white'}),
    html.Div(id='plot'),
    html.H4("Select Graph Type"),
    dcc.Dropdown(
        id='graph-type-dropdown',
        options=[
            {'label': 'Line', 'value': 'line'},
            {'label': 'Bar', 'value': 'bar'},
            {'label': 'Histogram', 'value': 'histogram'}
        ],
        value='line'
    ),
    html.Div(id='graph-list'),
    html.Div(
        "powered by Hasan MISAII",
        style={
            'position': 'fixed',
            'bottom': '10px',
            'right': '10px',
            'backgroundColor': 'blue',
            'color': 'white',
            'padding': '5px',
            'borderRadius': '5px'
        }
    )
], style={'backgroundColor': '#f0f0f0', 'border': '5px solid blue', 'paddingBottom': '170px'})  # Light gray background color with blue border

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename or 'txt' in filename:
            # Handle CSV and TXT with various delimiters
            decoded_str = decoded.decode('utf-8')
            delimiter = re.findall(r'[;, \t]', decoded_str)[0] if re.findall(r'[;, \t]', decoded_str) else ','
            df = pd.read_csv(io.StringIO(decoded_str), sep=delimiter)
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div([
                'Unsupported file type.'
            ])
        # Remove the first redundant column
        if df.columns[0] == df.index.name or df.columns[0] == 'Unnamed: 0':
            df = df.iloc[:, 1:]
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        return html.Div([
            html.H5(filename),
            html.Hr(),
            # Display the dataframe
            dcc.Graph(
                id='data-table',
                figure={
                    'data': [{
                        'type': 'table',
                        'header': {'values': list(df.columns)},
                        'cells': {'values': [df[col] for col in df.columns]}
                    }]
                }
            ),
            html.Hr(),
            html.H4("Select Column for Descriptive Statistics"),
            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=df.columns[0]
            ),
            html.H4("Select Graph Type for Descriptive Statistics"),
            dcc.Dropdown(
                id='desc-graph-type-dropdown',
                options=[
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Histogram', 'value': 'histogram'}
                ],
                value='line'
            ),
            html.H4("Filter Column"),
            dcc.Dropdown(
                id='filter-column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=df.columns[0]
            ),
            dcc.Dropdown(id='filter-value-dropdown'),
            html.Button('Apply Filter', id='filter-button', style={'background-color': 'red', 'color': 'white'}),
            html.Button('Add Filter', id='add-filter-button', n_clicks=0, style={'background-color': 'red', 'color': 'white'}),
            html.Div(id='additional-filters'),
            html.Button('Apply Filter', id='filter-button', style={'background-color': 'red', 'color': 'white'}),
            html.Div(id='column-stats'),
            html.Div(id='column-graph'),
            html.Div(id='column-filter'),
            html.H4("Select X and Y for Plot"),
            dcc.Dropdown(
                id='x-column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=df.columns[0]
            ),
            dcc.Dropdown(
                id='y-column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=df.columns[1]
            )
        ])

@app.callback(Output('filter-value-dropdown', 'options'),
              [Input('filter-column-dropdown', 'value')],
              [State('upload-data', 'contents'), State('upload-data', 'filename')])
def set_filter_values(selected_filter_column, contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if selected_filter_column in df.columns:
            unique_values = df[selected_filter_column].unique()
            return [{'label': str(val), 'value': str(val)} for val in unique_values]
    return []

@app.callback(
    Output('additional-filters', 'children'),
    [Input('add-filter-button', 'n_clicks'), Input({'type': 'remove-filter-button', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('additional-filters', 'children'), State('upload-data', 'contents'), State('upload-data', 'filename')]
)
def manage_filters(add_clicks, remove_clicks, existing_filters, contents, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return existing_filters

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if 'add-filter-button' in triggered_id:
        if add_clicks > 0 and contents is not None:
            df = parse_contents(contents, filename)
            new_filter = html.Div([
                html.H4(f"Filter Column {add_clicks}"),
                dcc.Dropdown(
                    id={'type': 'filter-column-dropdown', 'index': add_clicks},
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value=df.columns[0]
                ),
                dcc.Dropdown(id={'type': 'filter-value-dropdown', 'index': add_clicks}),
                html.Button('Apply Filter', id={'type': 'apply-filter-button', 'index': add_clicks}, style={'background-color': 'red', 'color': 'white'}),
                #html.Button('Remove Filter', id={'type': 'remove-filter-button', 'index': add_clicks}, style={'background-color': 'red', 'color': 'white'})
            ], id={'type': 'filter-div', 'index': add_clicks})
            if existing_filters is None:
                existing_filters = []
            existing_filters.append(new_filter)
            return existing_filters

    elif 'remove-filter-button' in triggered_id:
        index_to_remove = int(triggered_id.split('index":')[1].split('}')[0])
        existing_filters = [f for f in existing_filters if f['props']['id']['index'] != index_to_remove]
        return existing_filters

    return existing_filters

@app.callback(
    Output({'type': 'filter-value-dropdown', 'index': dash.dependencies.ALL}, 'options'),
    [Input({'type': 'filter-column-dropdown', 'index': dash.dependencies.ALL}, 'value')],
    [State('upload-data', 'contents'), State('upload-data', 'filename')]
)
def update_filter_values(selected_filter_columns, contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        options = []
        for col in selected_filter_columns:
            if col in df.columns:
                unique_values = df[col].unique()
                options.append([{'label': str(val), 'value': str(val)} for val in unique_values])
            else:
                options.append([])
        return options

@app.callback(
    [Output('column-stats', 'children'), Output('column-graph', 'children'), Output('column-filter', 'children')],
    [Input('column-dropdown', 'value'), Input('desc-graph-type-dropdown', 'value'), Input('filter-button', 'n_clicks')],
    [State('filter-column-dropdown', 'value'), State('filter-value-dropdown', 'value'), State({'type': 'filter-column-dropdown', 'index': dash.dependencies.ALL}, 'value'), State({'type': 'filter-value-dropdown', 'index': dash.dependencies.ALL}, 'value'), State('upload-data', 'contents'), State('upload-data', 'filename')]
)
def update_column_stats(selected_column, graph_type, n_clicks, filter_column, filter_value, additional_filter_columns, additional_filter_values, contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if selected_column in df.columns and filter_column in df.columns:
            if filter_value:
                df = df[df[filter_column].astype(str) == filter_value]
            for col, val in zip(additional_filter_columns, additional_filter_values):
                if col in df.columns and val:
                    df = df[df[col].astype(str) == val]
            # Exclude zeros
            df = df[df[selected_column] != 0]
            stats = html.Div([
                html.H4(f"Descriptive Statistics for {selected_column}"),
                html.Pre(df[selected_column].describe().to_string())
            ])
            if graph_type == 'line':
                graph = dcc.Graph(
                    figure={
                        'data': [{
                            'x': df.index,
                            'y': df[selected_column],
                            'type': 'line',
                            'name': selected_column
                        }],
                        'layout': {
                            'title': f'Line Graph of {selected_column}'
                        }
                    }
                )
            elif graph_type == 'bar':
                graph = dcc.Graph(
                    figure={
                        'data': [{
                            'x': df.index,
                            'y': df[selected_column],
                            'type': 'bar',
                            'name': selected_column
                        }],
                        'layout': {
                            'title': f'Bar Graph of {selected_column}'
                        }
                    }
                )
            elif graph_type == 'histogram':
                graph = dcc.Graph(
                    figure={
                        'data': [{
                            'x': df[selected_column],
                            'type': 'histogram',
                            'name': selected_column
                        }],
                        'layout': {
                            'title': f'Histogram of {selected_column}'
                        }
                    }
                )
            return stats, graph, html.Div([html.H4(f"Filtered Data for {filter_column} = {filter_value}"), html.Pre(df.to_string())])
    return html.Div(), html.Div(), html.Div()

@app.callback(Output('plot', 'children'),
              [Input('plot-button', 'n_clicks')],
              [State('x-column-dropdown', 'value'), State('y-column-dropdown', 'value'), State('plot-type-dropdown', 'value'), State('upload-data', 'contents'), State('upload-data', 'filename')])
def plot_graph(n_clicks, x_column, y_column, plot_type, contents, filename):
    if n_clicks is not None and contents is not None:
        df = parse_contents(contents, filename)
        if x_column in df.columns and y_column in df.columns:
            # Exclude zeros
            df = df[(df[x_column] != 0) & (df[y_column] != 0)]
            if plot_type == 'scatter':
                fig = px.scatter(df, x=x_column, y=y_column)
            elif plot_type == 'line':
                fig = px.line(df, x=x_column, y=y_column)
            elif plot_type == 'bar':
                fig = px.bar(df, x=x_column, y=y_column)
            elif plot_type == 'pie':
                fig = px.pie(df, names=x_column, values=y_column)
            return dcc.Graph(figure=fig)
    return html.Div()

@app.callback(Output('graph-list', 'children'),
              [Input('graph-type-dropdown', 'value')],
              [State('upload-data', 'contents'), State('upload-data', 'filename')])
def update_graph_list(graph_type, contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        graphs = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if graph_type == 'line':
                    graph = dcc.Graph(
                        figure={
                            'data': [{
                                'x': df.index,
                                'y': df[col],
                                'type': 'line',
                                'name': col
                            }],
                            'layout': {
                                'title': f'Line Graph of {col}'
                            }
                        }
                    )
                elif graph_type == 'bar':
                    graph = dcc.Graph(
                        figure={
                            'data': [{
                                'x': df.index,
                                'y': df[col],
                                'type': 'bar',
                                'name': col
                            }],
                            'layout': {
                                'title': f'Bar Graph of {col}'
                            }
                        }
                    )
                elif graph_type == 'histogram':
                    graph = dcc.Graph(
                        figure={
                            'data': [{
                                'x': df[col],
                                'type': 'histogram',
                                'name': col
                            }],
                            'layout': {
                                'title': f'Histogram of {col}'
                            }
                        }
                    )
                graphs.append(html.Div([
                    html.H4(f"Graph for {col}"),
                    graph
                ]))
        return html.Div(graphs)
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)
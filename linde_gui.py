import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
from data_import import DataImport, DataSampling
from flask_caching import Cache
import plotly.graph_objs as go

app = dash.Dash(__name__)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
}

cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


@cache.memoize()
def global_store_df():
    di = DataImport()
    df_deliveries, df_customers, df_colocations, df_levels = di.read_data_hdf()
    return df_deliveries, df_customers, df_colocations, df_levels


df_deliveries, df, df_colocations, df_levels = global_store_df()

PAGE_SIZE = 5

app.layout = html.Div(
    className="row",
    children=[
        html.Div(
            dash_table.DataTable(
                id='datatable_customers',
                columns=[
                    {"name": i, "id": i} for i in sorted(df.columns)
                ],
                pagination_settings={
                    'current_page': 0,
                    'page_size': 20
                },
                pagination_mode='be',

                filtering='be',
                filtering_settings='',
                row_selectable="single",
                selected_rows=[],
                n_fixed_rows=1,
                sorting='be',
                sorting_type='multi',
                sorting_settings=[],
                style_cell={'width': '150px'},
                style_table={'height': '300px'},
            ),
            className='six columns'
        ),
        html.Div(id='dummy', style={'display': 'none'}),
        html.Div(
#                  dcc.Graph(
#                      id='graph_lv_dl',
#                      figure={
#                          "data": [
#                              {
#                                  "x": np.arange(0, 4),
#                                  "y": np.arange(1, 5),
#                                  "type": "bar",
#                                  "marker": {"color": "#0074D9"},
#                              }
#                          ],
#                          "layout": {
#                              "xaxis": {"automargin": True},
#                              "yaxis": {"automargin": True},
#                              "height": 250,
#                              "margin": {"t": 10, "l": 10, "r": 10},
#                          },
#                      },
#                  ),
                 id='graph_lv_dl_container',
                 className="five columns",
                 )
    ]
)


@app.callback(
    Output('datatable_customers', "data"),
    [Input('datatable_customers', "pagination_settings"),
     Input('datatable_customers', "sorting_settings"),
     Input('datatable_customers', "filtering_settings")])
def update_table(pagination_settings, sorting_settings, filtering_settings):
    filtering_expressions = filtering_settings.split(' && ')
    dff = df
    for filter in filtering_expressions:
        if ' eq ' in filter:
            col_name = filter.split(' eq ')[0]
            filter_value = filter.split(' eq ')[1]
            dff = dff.loc[dff[col_name] == filter_value]
        if ' > ' in filter:
            col_name = filter.split(' > ')[0]
            filter_value = float(filter.split(' > ')[1])
            dff = dff.loc[dff[col_name] > filter_value]
        if ' < ' in filter:
            col_name = filter.split(' < ')[0]
            filter_value = float(filter.split(' < ')[1])
            dff = dff.loc[dff[col_name] < filter_value]

    if len(sorting_settings):
        dff = dff.sort_values(
            [col['column_id'] for col in sorting_settings],
            ascending=[
                col['direction'] == 'asc'
                for col in sorting_settings
            ],
            inplace=False
        )

    return dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1) *
        pagination_settings['page_size']
    ].to_dict('rows')


@app.callback(
    Output('graph_lv_dl_container', 'children'),
    [Input('datatable_customers','selected_rows')]
)
def updata_graph(row):
    if row == []:
        print('row is none')
        raise dash.exceptions.PreventUpdate
    else:
        row = row[0]

    df_deliveries, df_customers, df_colocations, df_levels = global_store_df()
    vessel_id = df_customers['VESSEL_ID'][row]
    print('got vess id', vessel_id)
    df_lv_s = df_levels.loc[df_levels['VESSEL_ID']==vessel_id]
    print('processed levels', vessel_id)
    df_dl_s = df_deliveries.loc[df_deliveries['VESSEL_ID']==vessel_id]
    print('got dataframes')
    df_l = [df_lv_s, df_dl_s]
    ids = ['levels', 'deliveries']
    graph_list = [[] for i in df_l]
    columns = ['INST_PRODUCT_AMOUNT', 'DELIVERED_VOLUME']

#     for idx,(each_id, df) in enumerate(zip(ids, df_l)):
#         graph_list[idx]= dcc.Graph(
#           id=each_id,
#           figure={
#               "data": [
#                   {
#                       "x":  df.index,
#                       "y": df[columns[idx]],
#                       "type": "scatter",
#                       "marker": {"color": "#0074D9"},
#                   }
#               ],
#               "layout": {
#                   "xaxis": {"automargin": True},
#                   "yaxis": {"automargin": True},
#                   "height": 250,
#                   "margin": {"t": 10, "l": 10, "r": 10},
#               },
#           },
#                      ),
#     print('returning')
    types=['scatter', 'bar']
    colors = []
    colors.append("#7FDBFF")
    colors.append("#0074D9")
    return html.Div(
        [
            dcc.Graph(
                id=each_id,
                figure={
                    "data": [
                        {
                            "x": df.index,
                            # check if column exists - user may have deleted it
                            # If `column.deletable=False`, then you don't
                            # need to do this check.
                            "y": df[columns[idx]],
                            "type": types[idx],
                            "marker": {"color": colors[idx]},
                        }
                    ],
                    "layout": {
                        "xaxis": {"automargin": True},
                        "yaxis": {"automargin": True},
                        "height": 250,
                        "margin": {"t": 10, "l": 10, "r": 10},
                    },
                },
            )
            for idx,(each_id, df) in enumerate(zip(ids, df_l))
        ]
    )

    return graph_list

if __name__ == '__main__':
    app.run_server(debug=True, port=8901)

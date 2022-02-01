import os
import ast
import sys
import subprocess
import traceback
import numpy as np

from flask import Flask, request, jsonify, abort, render_template, Response
from utils.preprocessing_data import load_df, retrieve_customers_id, retrieve_customers_features
from utils.model import model_selected, retrieve_model_data, neighbor_customers
from keys.api_keys import loan_api_keys


app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


script_dir = os.path.dirname(__file__)
df = load_df('df_custom_global_optim_std_w_id.pkl.gz')


# reroute to application client (streamlit)
@app.route('/dashboard/')
def display_dashboard():
    subprocess.run("streamlit run dashboard.py")
    return "Redirected to client application"


@app.route("/api/predict/", methods=['POST'])
def prediction():

    resp_api = request.args
    check_credentials = check_api_key(resp_api)

    if check_credentials:

        # noinspection PyBroadException
        try:
            json_data = request.json
            print(json_data)
            customer_id = json_data['customer_id']

            model_name = json_data['model_selected']
            model = model_selected(model_name)[0]

            # df = load_df('df_custom_global_optim_std_w_id.pkl.gz')
            query = df.drop('TARGET', axis=1)[df['SK_ID_CURR'] == customer_id]
            query = query.drop('SK_ID_CURR', axis=1).to_numpy()
            # print(query)
            predict = list(model.predict(query))
            predict_probability = list(model.predict_proba(query))

            return jsonify({'predict': str(predict), 'predict_probability': str(predict_probability)})

        except Exception:
            return jsonify({'trace': traceback.format_exc()})

    else:
        authorisation = Response("Authentication - user not allowed", mimetype='text/html'), 403
        return authorisation


@app.route("/api/predict_custom/", methods=['POST'])
def custom_prediction():

    resp_api = request.args
    check_credentials = check_api_key(resp_api)

    if check_credentials:

        # noinspection PyBroadException
        try:
            json_custom_feats = request.json
            # print("json_custom_feats:", json_custom_feats)
            custom_feats = json_custom_feats['customer_features']
            # print("custom_feats:", custom_feats)

            model_name = json_custom_feats['model_selected']
            model = model_selected(model_name)[0]

            custom_feats_to_list = ast.literal_eval(custom_feats)  # convert string list to list
            # print("custom_feats_to_list:", custom_feats_to_list)
            custom_feats_arr = np.array(custom_feats_to_list[0][1:])  # On ne prend pas en compte l'ID du client initial
            # print("custom_feats_arr:", custom_feats_arr)
            query = custom_feats_arr.reshape(1, -1)
            # print("query:", query)
            predict_custom = list(model.predict(query))
            predict_custom_proba = list(model.predict_proba(query))

            return jsonify({'predict_custom': str(predict_custom), 'predict_custom_proba': str(predict_custom_proba)})

        except Exception:
            return jsonify({'trace': traceback.format_exc()})

    else:
        authorisation = Response("Authentication - user not allowed", mimetype='text/html'), 403
        return authorisation


@app.route("/api/customers_id/", methods=['POST'])
def get_customers_id():

    resp_api = request.args
    check_credentials = check_api_key(resp_api)

    if check_credentials:
        # authorisation = Response("Streamlit client authorized", minetype='text/html'), 200

        # noinspection PyBroadException
        try:
            customer_id_list = retrieve_customers_id(df)
            return jsonify({'customer_id': str(customer_id_list)})

        except Exception:
            return jsonify({'trace': traceback.format_exc()})

    else:
        authorisation = Response("Authentication - user not allowed", mimetype='text/html'), 403

    return authorisation


@app.route("/api/customer_feats/", methods=['POST'])
def get_customer_features():

    resp_api = request.args
    check_credentials = check_api_key(resp_api)

    if check_credentials:

        # noinspection PyBroadException
        try:
            json_id = request.json
            print(json_id)

            customer_features, columns_list = retrieve_customers_features(df, json_id['customer_id'])
            return jsonify({'df_columns': str(columns_list), 'customer_features': str(customer_features)})

        except Exception:
            return jsonify({'trace': traceback.format_exc()})

    else:
        authorisation = Response("Authentication - user not allowed", mimetype='text/html'), 403
        return authorisation


@app.route("/api/model_selection/", methods=['POST'])
def select_model_prediction():

    resp_api = request.args
    check_credentials = check_api_key(resp_api)

    if check_credentials:

        # noinspection PyBroadException
        try:
            json_model = request.json
            print("json_model:", json_model)

            cm_model, cr_model, fbeta_model, fpr_model, tpr_model, auc_model, most_imp_feats = \
                retrieve_model_data(json_model['model_selected'])

            return jsonify({
                'cm_model': str(cm_model),
                'cr_model': str(cr_model),
                'fbeta_model': str(fbeta_model),
                'fpr_model': str(fpr_model),
                'tpr_model': str(tpr_model),
                'auc_model': str(auc_model),
                'most_imp_feats': str(most_imp_feats)
            })

        except Exception:
            return jsonify({'trace': traceback.format_exc()})

    else:
        authorisation = Response("Authentication - user not allowed", mimetype='text/html'), 403
        return authorisation


@app.route("/api/customer_neighbors/", methods=['POST'])
def get_neighbor_customers_pred():

    resp_api = request.args
    check_credentials = check_api_key(resp_api)

    if check_credentials:

        # noinspection PyBroadException
        try:
            json_neigh_pred = request.json
            print("json_model:", json_neigh_pred)

            # retourne "neighbors_pred_mean" pour customer neighbors et customer custom neighbors
            neighbors_pred_mean, neighbors_chars_sum = \
                neighbor_customers(
                    json_neigh_pred['model_selected'],
                    json_neigh_pred['customer_id'],
                    # int(json_neigh_pred['customer_id']),
                    json_neigh_pred['customer_custom_feats']
                )

            return jsonify({
                'neighbors_pred_mean': str(neighbors_pred_mean),
                'neighbors_chars_sum': str(neighbors_chars_sum)
            })

        except Exception:
            return jsonify({'trace': traceback.format_exc()})

    else:
        authorisation = Response("Authentication - user not allowed", mimetype='text/html'), 403
        return authorisation


def check_api_key(request_args):

    response = request_args

    # si une clé est fournie, on vérifie sa validité
    if 'loan_app_key' in response and response['loan_app_key'] in loan_api_keys.values():

        print("Ce client Streamlit est connu et autorisé à utiliser l'API")
        # authorisation = jsonify({'access': "authorized"})
        authorisation = True
    else:
        # authorisation = jsonify({'access': "not_authorized"})
        authorisation = False

    return authorisation


if __name__ == "__main__":

    # noinspection PyBroadException
    try:
        port = int(sys.argv[1])  # command line argument, [0] here !

    except Exception:
        port = 12345  # if no port is specify, this will be 12345 by default

    app.run(debug=True)  # Set to False in production

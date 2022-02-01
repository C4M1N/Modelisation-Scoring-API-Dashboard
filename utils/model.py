import os
import ast
import joblib
import requests
# import time
import pickle
import numpy as np
# import cloudpickle as cp

# from datetime import datetime
from io import BytesIO
from gzip import GzipFile
from utils.preprocessing_data import load_df
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, fbeta_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


script_dir = os.path.dirname(__file__)

models_com_dict = {
	'RandomForestClassifier': ["https://github.com/C4M1N/loan_data_models/blob/master/gs_rfc_std.pkl?raw=true", 2],
	'XGBClassifier': ["https://github.com/C4M1N/loan_data_models/blob/master/xgb_12_hyperopt.pkl?raw=true", 2.2],
	'CatBoostClassifier': ["https://github.com/C4M1N/loan_data_models/blob/master/catboost_model.pkl?raw=true", 2],
	'SVC': ["https://github.com/C4M1N/loan_data_models/blob/master/svc_hyperopt_wo_smote_beta_1-8.pkl.gz?raw=true", 1.8],
}


def model_selected(model_chosen=None):

	if not model_chosen:
		model_chosen = 'RandomForestClassifier'  # A modifier ensuite

	model_url = models_com_dict[model_chosen][0]
	model_beta = models_com_dict[model_chosen][1]

	model_file = BytesIO(requests.get(model_url).content)

	if model_chosen == 'SVC':
		model_file = GzipFile(fileobj=model_file)  # decompressed_file

	joblib_model = joblib.load(model_file)

	if model_chosen == 'RandomForestClassifier':
		print("RFC parameters found: {}".format(joblib_model.best_params_))
	elif model_chosen == 'XGBClassifier':
		print("XGB parameters found: {}".format(joblib_model.get_xgb_params()))
	elif model_chosen == 'CatBoostClassifier':
		print("CatBoost parameters found: {}".format(joblib_model.get_all_params()))
	elif model_chosen == 'SVC':
		print("SVC parameters found: {}".format(joblib_model.get_params()))  # loaded with probability=True

	print("Selected model ready to go.")

	return joblib_model, model_beta


def retrieve_model_data(model_name):

	joblib_model, model_beta = model_selected(model_name)

	X_test = pickle.load(open(os.path.join(script_dir, '../data/' + 'X_test_2.pkl'), 'rb'))
	y_pred = joblib_model.predict(X_test)
	y_pred_prob = joblib_model.predict_proba(X_test)[:, 1]  # positive class
	y_test = pickle.load(open(os.path.join(script_dir, '../data/' + 'y_test_2.pkl'), 'rb'))

	# confusion matrix
	cm_model = confusion_matrix(y_test, y_pred)
	cm_model = cm_model.tolist()

	# classification report
	cr_model = classification_report(y_test, y_pred, output_dict=True)

	# fbeta score
	fbeta_model = fbeta_score(y_test, y_pred, average='binary', beta=model_beta)

	# ROC curve & AUC
	fpr_model, tpr_model, _ = roc_curve(y_test, y_pred_prob)
	auc_model = auc(fpr_model, tpr_model)

	# Evaluation des 4 premières variables d'importance dans la prédiction du modèle (LIME/SHAP)
	# A redéfinir en fonction de l'entraînement et du type de modèle choisi.
	if model_name == 'RandomForestClassifier':
		most_imp_feats = ['EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'NAME_EDUCATION_TYPE__Higher education']
	elif model_name == 'XGBClassifier':
		most_imp_feats = ['EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'pos_cash_bal__sk_dpd']
	elif model_name == 'SVC':
		most_imp_feats = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'ORGANIZATION_TYPE__XNA', 'CODE_GENDER__M']
	elif model_name == 'CatBoostClassifier':
		most_imp_feats = \
			['bureau__amt_credit_sum_overdue', 'EXT_SOURCE_1', 'AMT_GOODS_PRICE', 'instal_pay__amt_total_diff']

	return cm_model, cr_model, fbeta_model, fpr_model, tpr_model, auc_model, most_imp_feats


def neighbor_customers(model_name, customer_id, customer_custom_features):

	df = load_df('df_custom_global_optim_std_w_id.pkl.gz')
	joblib_model, _ = model_selected(model_name)

	df.set_index(['SK_ID_CURR'], inplace=True)
	df.drop(['TARGET'], axis=1, inplace=True)
	print("df.shape:", df.shape)  # doit contenir 383 features

	n_neigh = NearestNeighbors(n_neighbors=20)  # les 20 plus proches clients

	# Liste des colonnes d'informations complémentaires sur le(s) client(s)
	customer_chars_file = open(os.path.join(script_dir, '../data/customer_chars_col_to_list.txt'), 'r')

	with customer_chars_file as file:
		customer_chars_str = file.read()
		customer_chars_list = ast.literal_eval(customer_chars_str)
		file.close()
	# print("customer_chars_list :", customer_chars_list)

	# df_w_id = df.copy()
	# print("df_w_id", df_w_id.head())

	if customer_id:

		customer_id = int(customer_id)  # est transmis en 'string' json

		# On fit le jeu sans l'ID du client demandé
		n_neigh.fit(df.drop(df[df.index == customer_id].index))

		# customer_profil = df.iloc[customer_id].values.reshape(1, -1)  # Attention retourne l'index pas l'SK_ID_CURR
		customer_profil = df.loc[customer_id].values.reshape(1, -1)  # car l'index a été setté avec SK_ID_CURR
		print("customer_profil.shape:", customer_profil.shape)

		customer_neighbors = n_neigh.kneighbors(customer_profil)
		customer_neighbors_list = customer_neighbors[1][0].tolist()
		# print("List of similar profiles IDs:", customer_neighbors_list)
		print("List of similar profiles Index (Index array IDs):", customer_neighbors_list)

		n_neigh_customers_list = df.iloc[customer_neighbors_list].values
		predict_proba_n_neigh_customers = joblib_model.predict_proba(n_neigh_customers_list)
		pred_prob_n_neigh_customers = np.mean(predict_proba_n_neigh_customers, axis=0)
		print("pred_prob_n_neigh_customers:", pred_prob_n_neigh_customers)

		# somme du nombre de clients par caractéristiques demandées
		# l'index de l'array de NK
		neighbors_chars_sum = df[customer_chars_list].iloc[customer_neighbors_list].sum().to_dict()
		# print("CUSTOMERS_CHARS_SUM --> ", neighbors_chars_sum)

		return pred_prob_n_neigh_customers, neighbors_chars_sum

	elif customer_custom_features:

		n_neigh.fit(df)

		customer_custom_feats_to_list = ast.literal_eval(customer_custom_features)  # convert string list to list
		# On ne prend pas en compte ID du client initial (qui est récupéré du profil client initial)
		customer_custom_feats_arr = np.array(customer_custom_feats_to_list[0][1:])
		customer_custom_profil = customer_custom_feats_arr.reshape(1, -1)

		customer_neighbors_custom = n_neigh.kneighbors(customer_custom_profil)
		customer_neighbors_custom_list = customer_neighbors_custom[1][0].tolist()
		print("List of similar profiles Index (Index array) with custom customer profil:", customer_neighbors_custom_list)

		n_neigh_customers_custom_list = df.iloc[customer_neighbors_custom_list].values
		predict_proba_n_neigh_customers_custom = joblib_model.predict_proba(n_neigh_customers_custom_list)
		pred_prob_n_neigh_customers_custom = np.mean(predict_proba_n_neigh_customers_custom, axis=0)
		print("pred_prob_n_neigh_customers_custom:", pred_prob_n_neigh_customers_custom)

		# somme du nombre de clients par caractéristiques "custom" demandées
		# l'index de l'array de NK
		neighbors_custom_chars_sum = df[customer_chars_list].iloc[customer_neighbors_custom_list].sum().to_dict()

		return pred_prob_n_neigh_customers_custom, neighbors_custom_chars_sum

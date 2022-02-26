import os
import ast
# import time
import requests

import seaborn as sns
# import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image


script_dir = os.path.dirname(__file__)
# here is your personal api key to fill
api_key = {"loan_app_key": "c30gs65de84zqs64qz98eex24sa1huk4765fg5dsda"}
api_url = "http://127.0.0.1:5000/"

# Import du logo du projet sur le dashboard
img_logo = Image.open(f'{script_dir}/img/app_loan_logo.png')
st.sidebar.image(img_logo, width=200)  # use_column_width=auto
selected_page = st.sidebar.selectbox("Page à afficher", ("Dashboard", "Modèles"))

if 'model_list' not in st.session_state:
	st.session_state.model_list = ["RandomForestClassifier", "XGBClassifier", "CatBoostClassifier", "SVC"]

# sélection du modèle par défaut
model_selected = "RandomForestClassifier"

if selected_page == "Modèles":

	st.write("""
    # MODÈLES - Sélection
    Affiche les statistiques des modèles utilisés.
    ***
    """)

	model_selected = st.sidebar.radio("Modèles", st.session_state.model_list)
	# permet de remettre l'élément sélectionné en première position avec le st.ratio
	st.session_state.model_list.insert(
		0,
		st.session_state.model_list.pop(st.session_state.model_list.index(model_selected))
	)
	# contrôle
	# st.write("st.session_state.model_list:", st.session_state.model_list)

	# correction si même utilisateur initial et initié (='100009') avec profil modifié et qui visite la page des modèles
	if st.session_state.last_dash_id == '100009' and st.session_state.custom_profil:
		st.session_state.custom_profil = False  # va permettre de réinitialiser l'affichage

	def change_model(model_name):

		st.write("""_Voici les statistiques de performance du modèle ------->_""", model_name)
		if model_name == "SVC":
			warning_to_display = "Le chargement du modèle SVC peut être assez long. Veuillez patienter..."
		else:
			warning_to_display = "Chargement du modèle en cours..."

		st.text("")

		with st.spinner(warning_to_display):

			# Envoi d'une requête POST pour changer de modèle de prédiction
			url_model = f'{api_url}api/model_selection/'
			resp_model_selected = requests.post(url=url_model, json={'model_selected': model_name}, params=api_key)

			if resp_model_selected.status_code == requests.codes.ok:

				model_data_json = resp_model_selected.json()
				print(model_data_json)

				# print(type(model_data_json['fpr_model']), model_data_json['fpr_model'][1:-1])
				fpr_model = np.fromstring(model_data_json['fpr_model'][1:-1], dtype=float, sep=' ').tolist()
				print("fpr_model:", type(fpr_model), fpr_model)

				tpr_model = np.fromstring(model_data_json['tpr_model'][1:-1], dtype=float, sep=' ').tolist()
				print("tpr_model:", type(tpr_model), tpr_model)

				most_imp_feats = ast.literal_eval(model_data_json['most_imp_feats'])  # convert string list to list

				st.write("Les 4 premières variables d'importance dans la prédiction")
				col_feat_1, col_feat_2, col_feat_3, col_feat_4 = st.columns(4)

				for idx, imp_feats in enumerate(most_imp_feats):
					if len(imp_feats) > 15:
						most_imp_feats[idx] = f'{imp_feats[:15]}...'

				with col_feat_1:
					st.write(f"""*{most_imp_feats[0]}*""")
				with col_feat_2:
					st.write(f"""*{most_imp_feats[1]}*""")
				with col_feat_3:
					st.write(f"""*{most_imp_feats[2]}*""")
				with col_feat_4:
					# feat_displayed = '<span style="white space: none !important;">' + most_imp_feats[3] + '</span>'
					# st.markdown(feat_displayed, unsafe_allow_html=True)
					st.write(f"""*{most_imp_feats[3]}*""")

				st.text("")  # espacement

				col_stats_1, col_stats_2, col_stats_3 = st.columns(3)

				with col_stats_1:
					# confusion matrix
					cm_model = ast.literal_eval(model_data_json['cm_model'])
					print("cm_model:", type(cm_model), cm_model)
					st.write("Matrice de confusion :")
					df_cm_model = pd.DataFrame(cm_model)
					st.write(df_cm_model)

				with col_stats_2:
					# f beta score
					fbeta_model = float(model_data_json['fbeta_model'])
					print("fbeta_model:", type(fbeta_model), fbeta_model)
					st.write("F beta score sur le jeu de test :")
					st.metric(label="Fbeta score", value="{:.2f}".format(fbeta_model))

				with col_stats_3:
					# AUC
					auc_model = float(model_data_json['auc_model'])
					print("auc_model:", type(auc_model), auc_model)
					st.write("Aire sous la courbe ROC :")
					st.metric(label="AUC", value="{:.2f}".format(auc_model))

				st.text("")  # espacement

				# classification report
				cr_model = ast.literal_eval(model_data_json['cr_model'])
				print("cr_model:", type(cr_model), cr_model)
				st.write("Rapport de classification :")
				df_cr_model = pd.DataFrame(cr_model).transpose()
				st.write(df_cr_model)

				st.text("")  # espacement

				# ROC curve
				plt.style.use('dark_background')
				fig, ax = plt.subplots(figsize=(8, 4))
				fig.patch.set_alpha(0.0)
				ax.patch.set_facecolor((14/255., 17/255., 23/255.))
				ax.plot(fpr_model, tpr_model)
				ax.plot([0, 1], [0, 1], '--')
				ax.set_xlabel("Taux de False positive")
				ax.set_ylabel("Taux de True positive")
				ax.set_title(f"Courbe ROC du modèle {str(model_name)}")
				st.pyplot(fig)

			elif resp_model_selected.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				st.stop()
				return

			else:
				resp_model_selected.raise_for_status()

		return None

	# On signale le changement de modèle à l'API pour ensuite récupérer les stats du modèle
	# On garde en mémoire le changement de modèle sélectionné
	change_model(model_selected)


elif selected_page == "Dashboard":

	# contrôle de sélection de modèle
	# st.write("st.session_state.model_list:", st.session_state.model_list)

	model_selected = st.session_state.model_list[0]
	# contrôle
	# st.write("""_Modèle de prédiction utilisé:_""", model_selected)

	if 'custom_profil' not in st.session_state:
		st.session_state.custom_profil = False

	# callback function
	def enable_custom_profil():
		# contrôles/vérifications

		# print("st.session_state.sliders from CALLBACK:", st.session_state.sliders)

		# print("st.session_state.organization_type :", st.session_state.organization_type)
		# print("st.session_state.occupation_type :", st.session_state.occupation_type)
		# print("st.session_state.name_income_type :", st.session_state.name_income_type)
		# print("st.session_state.name_housing_type :", st.session_state.name_housing_type)
		# print("st.session_state.name_family_status :", st.session_state.name_family_status)
		# print("st.session_state.name_education_type :", st.session_state.name_education_type)
		# print("st.session_state.name_contract_type :", st.session_state.name_contract_type)
		# print("st.session_state.flag_own_realty :", st.session_state.flag_own_realty)
		# print("st.session_state.flag_own_car :", st.session_state.flag_own_car)
		# print("st.session_state.code_gender :", st.session_state.code_gender)

		# Récupération des caractéristiques du profil utilisateur
		organization_df = gen_df_custom_spec_feats(organization_trad, st.session_state.organization_type)
		occupation_df = gen_df_custom_spec_feats(occupation_trad, st.session_state.occupation_type)
		income_df = gen_df_custom_spec_feats(income_trad, st.session_state.name_income_type)
		housing_df = gen_df_custom_spec_feats(housing_trad, st.session_state.name_housing_type)
		family_df = gen_df_custom_spec_feats(family_trad, st.session_state.name_family_status)
		education_df = gen_df_custom_spec_feats(education_trad, st.session_state.name_education_type)
		contract_df = gen_df_custom_spec_feats(contract_trad, st.session_state.name_contract_type)
		realty_df = gen_df_custom_spec_feats(dwelling_trad, st.session_state.flag_own_realty)
		car_df = gen_df_custom_spec_feats(car_trad, st.session_state.flag_own_car)
		gender_df = gen_df_custom_spec_feats(gender_trad, st.session_state.code_gender)

		# customer_profile_chars_df
		st.session_state.customer_profil_chars = pd.concat([
			organization_df,
			occupation_df,
			income_df,
			housing_df,
			family_df,
			education_df,
			contract_df,
			realty_df,
			car_df,
			gender_df
		],
			axis=1
		)

		# print("customer_profile_chars_df", customer_profile_chars_df.head())
		# print(st.session_state.customer_profil_chars.info(verbose=True))
		# st.write(st.session_state.customer_profil_chars)

		st.session_state.custom_profil = True
		# dash_id_last_session = st.sessions_state.dash_id_field

	st.write("""
    # DASHBOARD - Prêt à dépenser
    Aide à la décision client - 
    _(Modèle de prédiction utilisé:_ """, model_selected, """)
    ***
    """)

	if model_selected == 'SVC':
		st.write("""
		_Important: Le temps de calcul pour le modèle SVC peut être assez long._
		""")

	# 20 entrées dans le cache et refresh 5h
	@st.cache(ttl=60*5, max_entries=20, suppress_st_warning=True, show_spinner=False)
	def loading_params_for_gui():
		# Envoi d'une requête POST pour récupérer la liste des ID des clients disponibles
		url_id = f'{api_url}api/customers_id/'
		resp_customers_id = requests.post(url=url_id, params=api_key)

		if resp_customers_id.status_code == requests.codes.ok:

			# st.write("Chargement des données d'interface, merci de patienter...")
			customer_id_list = resp_customers_id.json()
			customer_id_to_list = ast.literal_eval(customer_id_list['customer_id'])  # convert string list to list
			# st.write(type(customer_id_to_list))
			# st.write(customer_id_to_list)

			print("Chargement de la liste des ID clients terminé.")

		elif resp_customers_id.status_code == 403:
			st.write("Client non autorisé à accéder à l'API.")
			st.stop()
			return

		else:
			resp_customers_id.raise_for_status()

		return customer_id_to_list

	# bar de chargement
	pg_bar = st.progress(0)

	customers_id_list = loading_params_for_gui()

	pg_bar.progress(10)

	# st.write(type(customers_id_list))

	# Création de la sidebar streamlit
	st.sidebar.header("Sélection de l'identifiant client:")
	# st.write(str(customer_id_list['customer_id']))

	# gestion et contrôle de l'ID
	if 'last_dash_id' not in st.session_state:
		st.session_state.last_dash_id = '100009'

	is_same_dash_id = True

	customer_id_text_input = st.sidebar.text_input("Entrer l'ID d'un client", '100009', key='dash_id_field')

	if st.session_state.last_dash_id != st.session_state.dash_id_field:
		st.session_state.custom_profil = False
		is_same_dash_id = False

	st.session_state.last_dash_id = customer_id_text_input
	# contrôle
	# st.write("is_same_dash_id:", is_same_dash_id)

	# par défaut on contrôle l'entrée utilisateur
	if 'valid_customer_id' not in st.session_state:
		st.session_state.valid_customer = False

	try:
		if int(customer_id_text_input) in customers_id_list:
			print("L'identifiant est connu !")
			st.session_state.valid_customer = True
	except ValueError:
		print("Identifiant inconnu")
		st.session_state.dash_id_field = "100009"
		st.session_state.valid_customer = False
		pg_bar.empty()

	# st.write("dash_id_field:", dash_id_field)
	# st.write("st.session_state.dash_id_field", st.session_state.dash_id_field)

	if st.session_state.valid_customer:

		# requêtes à l'API uniquement si on ne compare pas avec un profil modifié (custom)
		if not st.session_state.custom_profil:
			url_pred = api_url + "api/predict/"
			# en théorie il faut respécifier le modèle
			resp_predict_default = requests.post(
				url=url_pred,
				json={'model_selected': model_selected, 'customer_id': int(st.session_state.dash_id_field)},
				params=api_key
			)

			url_features = api_url + "api/customer_feats/"
			resp_customer_features = requests.post(
				# selected_id
				url=url_features, json={'customer_id': int(st.session_state.dash_id_field)},
				params=api_key
			)

			url_neighbors = api_url + "api/customer_neighbors/"
			print("model_selected:", model_selected)
			resp_customer_neighbors = requests.post(
				# selected_id
				url=url_neighbors,
				json={
					'model_selected': model_selected,
					'customer_id': int(st.session_state.dash_id_field),
					'customer_custom_feats': ""
				},
				params=api_key
			)

			url_most_imp_feats = api_url + "api/model_selection/"
			resp_most_imp_feats = requests.post(url=url_most_imp_feats, json={'model_selected': model_selected}, params=api_key)
			pg_bar.progress(10)

		# customer predictions
		try:
			if resp_predict_default.status_code == requests.codes.ok:

				customer_predict_default = resp_predict_default.json()
				print("Retour de l'API pour la prédiction de défaut client:", customer_predict_default)
				customer_pred_default = int(customer_predict_default['predict'][1])  # convert string list to list to int
				customer_pred_proba_default = customer_predict_default['predict_probability']  # convert string list to list

				pg_bar.progress(20)

			elif resp_predict_default.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				pg_bar.empty()
				st.stop()

			else:
				resp_predict_default.raise_for_status()

		except NameError:
			pg_bar.progress(20)
			pass  # car on ne souhaite pas requêter à nouveau l'API

		if st.session_state.custom_profil:
			# contrôle
			# st.write("On charge les valeurs pour comparaison de profil")
			# si on compare on récupère les valeurs déjà demandées à l'API
			# client par défaut
			customer_pred_default = st.session_state.customer_pred
			customer_pred_proba_default = st.session_state.customer_idx

		# customer features
		try:
			if resp_customer_features.status_code == requests.codes.ok:

				customer_features = resp_customer_features.json()
				customer_feats_values = customer_features['customer_features']
				columns_feats = customer_features['df_columns']

				# print("Colonnes:", columns_feats)
				# print("Features du client:", customer_feats_values)

				pg_bar.progress(30)

			elif resp_customer_features.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				pg_bar.empty()
				st.stop()

			else:
				resp_customer_features.raise_for_status()
		except NameError:
			pg_bar.progress(30)
			pass

		if st.session_state.custom_profil:
			# on récupère les valeurs pour le client par défaut pour comparer avec le profil modifié
			columns_feats = st.session_state.customer_total_feats
			customer_feats_values = st.session_state.customer_total_feats_values

		# neighbor customers proba
		try:
			if resp_customer_neighbors.status_code == requests.codes.ok:

				customer_neighbors = resp_customer_neighbors.json()
				customer_neighbors_values = np.fromstring(
					customer_neighbors['neighbors_pred_mean'][1:-1], dtype=float, sep=' '
				).tolist()

				# print("Proba des clients proches:", customer_neighbors_values)
				# st.write("Moyenne de probabilité des clients proches:", customer_neighbors_values)

				customer_neighbors_chars_sum = ast.literal_eval(customer_neighbors['neighbors_chars_sum'])
				# print("customer_neighbors_chars_sum TYPE :", type(customer_neighbors_chars_sum))
				# print("//----------- customer_neighbors_chars_sum ---------------- //", customer_neighbors_chars_sum)

				pg_bar.progress(40)

			elif resp_customer_neighbors.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				pg_bar.empty()
				st.stop()

			else:
				resp_customer_neighbors.raise_for_status()
		except NameError:
			pg_bar.progress(40)
			pass

		if st.session_state.custom_profil:
			# on récupère les valeurs pour le client par défaut pour comparer avec le profil modifié
			customer_neighbors_values = st.session_state.customer_neigh_pred

		if not st.session_state.custom_profil:
			# client normal
			# retour de l'API
			customer_feats_values = ast.literal_eval(customer_feats_values)  # convert string list to list
			columns_feats = ast.literal_eval(columns_feats)

			st.session_state.customer_total_feats = columns_feats
			st.session_state.customer_total_feats_values = customer_feats_values

		# récupération des variables les plus importantes pour la prédiction du modèle
		try:
			if resp_most_imp_feats.status_code == requests.codes.ok:
				model_data_json = resp_most_imp_feats.json()
				# print(model_data_json)

				imp_feats_model_list = ast.literal_eval(model_data_json['most_imp_feats'])  # convert string list to list

				pg_bar.progress(50)

			elif resp_most_imp_feats.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				pg_bar.empty()
				st.stop()

			else:
				resp_most_imp_feats.raise_for_status()
		except NameError:
			pg_bar.progress(50)
			pass

		if st.session_state.custom_profil:
			imp_feats_model_list = st.session_state.customer_imp_feats

		customer_df = pd.DataFrame({'feats_values': customer_feats_values[0], 'columns': columns_feats})
		# print(customer_df)

		df_criteria = customer_df.loc[customer_df['columns'].isin(imp_feats_model_list)].copy()
		df_criteria.reset_index(drop=True, inplace=True)
		df_criteria['tick_labels'] = ["Critère " + str(idx + 1) for idx in range(
			customer_df.loc[customer_df['columns'].isin(imp_feats_model_list)].shape[0]
		)]

		# sidebar sliders
		if 'sliders' not in st.session_state:
			st.session_state.sliders = {}
		for idx, feat in enumerate(imp_feats_model_list):
			st.session_state.sliders[idx] = st.sidebar.slider(
				label=df_criteria.loc[idx, 'tick_labels'],
				min_value=0.0,
				max_value=1.0,
				value=float(round(df_criteria.loc[idx, 'feats_values'], 4)),  # 0.05,
				step=0.01,
				on_change=enable_custom_profil
			)

		print("Sliders data:", st.session_state.sliders)

		pg_bar.progress(60)

		# custom profil
		# st.write("1st st.session_state.is_same_dash_id", st.session_state.is_same_dash_id)
		if st.session_state.custom_profil:
			df_custom_profil = pd.DataFrame(data=customer_feats_values, columns=columns_feats)
			print("imp_feats_model_list", imp_feats_model_list)
			print("Before: df_custom_profil\n", df_custom_profil[imp_feats_model_list].values.tolist())
			print("Sliders_data", st.session_state.sliders)
			for idx, elt in enumerate(df_criteria['columns'].values):
				# On affecte les valeurs au profil client modifié
				df_custom_profil.loc[0, elt] = st.session_state.sliders[idx]

			df_customer_profil_chars = st.session_state.customer_profil_chars
			print("df_customer_profil_chars['CODE_GENDER__M']", df_customer_profil_chars['CODE_GENDER__M'])

			for elt in df_customer_profil_chars.columns:
				df_custom_profil.loc[0, elt] = df_customer_profil_chars[elt].values

			# st.write(df_custom_profil)

			print("After: df_custom_profil\n", df_custom_profil[imp_feats_model_list].values.tolist())

			# Appel à l'API pour la prédiction du profil dérivé du client initial via les sliders
			print("Custom features sent to API:", df_custom_profil.values)
			url_pred_custom = api_url + "api/predict_custom/"
			# en théorie il faut respécifier le modèle
			resp_predict_custom = requests.post(
				url=url_pred_custom,
				json={'model_selected': model_selected, 'customer_features': str(df_custom_profil.values.tolist())},
				params=api_key
			)

			pg_bar.progress(70)

			# customer custom predictions
			if resp_predict_custom.status_code == requests.codes.ok:
				customer_predict_custom = resp_predict_custom.json()
				print("Retour de l'API pour la prédiction du client custom:", customer_predict_custom)
				# convert string list to list to int
				customer_pred_custom = int(customer_predict_custom['predict_custom'][1])
				customer_pred_proba_custom = customer_predict_custom['predict_custom_proba'][1:-1]  # =strip("'")
				# custom_preds_data = [customer_pred_custom, customer_pred_proba_custom, st.session_state.sliders]
				custom_preds_data = [customer_pred_custom, customer_pred_proba_custom]
				print("Custom_preds_data:", custom_preds_data)

				pg_bar.progress(80)

			elif resp_predict_custom.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				pg_bar.empty()
				st.stop()

			# customer custom neighbors preds
			url_neighbors_custom = api_url + "api/customer_neighbors/"
			print("model_selected:", model_selected)
			resp_customer_neighbors_custom = requests.post(
				# selected_id
				url=url_neighbors_custom,
				json={
					'model_selected': model_selected,
					'customer_id': "",
					'customer_custom_feats': str(df_custom_profil.values.tolist())},
				params=api_key
			)

			if resp_customer_neighbors_custom.status_code == requests.codes.ok:
				customer_neighbors_custom = resp_customer_neighbors_custom.json()
				print("customer_neighbors_custom (json response):", customer_neighbors_custom)
				customer_neighbors_custom_values = np.fromstring(
					customer_neighbors_custom['neighbors_pred_mean'][1:-1], dtype=float, sep=' '
				).tolist()

				# contrôles
				# print("Proba des clients proches (custom):", customer_neighbors_custom_values)
				# st.write("Moyenne de probabilité des clients proches (custom):", customer_neighbors_custom_values)

				customer_neighbors_custom_chars_sum = ast.literal_eval(customer_neighbors_custom['neighbors_chars_sum'])

				pg_bar.progress(90)

			elif resp_customer_neighbors_custom.status_code == 403:
				st.write("Client non autorisé à accéder à l'API.")
				pg_bar.empty()
				st.stop()

		# Affichage de la prédiction du client modifié (predict & predict_proba)
		if st.session_state.custom_profil:
			# custom_preds_data = (customer_pred_custom, customer_pred_proba_custom, sliders_data)
			customer_pred_custom = custom_preds_data[0]
			print("Show_differences:", custom_preds_data)
			is_customer_in_default = 'OUI' if customer_pred_custom == 1 else 'NON'
			if customer_pred_custom == 1:
				confidence_idx_custom = custom_preds_data[1].split(",")[1][-13:-3]  # donne l'indice de confiance pour OUI
			else:
				confidence_idx_custom = custom_preds_data[1].split(",")[0][-10:]  # donne l'indice de confiance pour NON

			print("confidence_idx_custom raw:", confidence_idx_custom, type(confidence_idx_custom))
			customer_idx_custom = "{:.2f} %".format(float(confidence_idx_custom)*100)  # pour éviter le format décimal (0.xx)
			print("customer_idx_custom:", customer_idx_custom)

		# Affichage de la prédiction client normal (predict & predict_proba)
		if st.session_state.custom_profil:
			# si on compare le client normal avec le profil modifié
			customer_idx = st.session_state.customer_idx
		else:
			# Affichage pour client normal (=sans comparaison)
			is_customer_in_default = 'OUI' if customer_pred_default == 1 else 'NON'
			if customer_pred_default == 1:
				# donne l'indice de confiance pour OUI
				confidence_idx = customer_pred_proba_default.split(",")[1][-13:-3]
			else:
				# donne l'indice de confiance pour NON
				confidence_idx = customer_pred_proba_default.split(",")[0][-10:]

			print("confidence_idx raw:", confidence_idx, type(confidence_idx))
			customer_idx = "{:.2f} %".format(float(confidence_idx)*100)  # pour éviter le format décimal (0.xx)
			print("customer_idx:", customer_idx)
			# confidence_idx_str = "{} %".format(int(confidence_idx))
			# print("confidence_idx_str:", confidence_idx_str)

		pg_bar.progress(95)

		# Affichage de la prédiction pour le client et son indice de confiance
		if not st.session_state.custom_profil:
			col_a, col_b = st.columns(2)
			with col_a:
				st.subheader("Statut du client " + st.session_state.dash_id_field + " ?")
			with col_b:
				st.subheader("20 clients avec profils similaires ? ")
		else:
			col_a, col_b = st.columns(2)
			with col_a:
				st.subheader("Statut du client " + st.session_state.dash_id_field + " ?")
			with col_b:
				st.subheader("Profil " + st.session_state.dash_id_field + " modifié")

		if not st.session_state.custom_profil:

			col1, col2, col3, col4 = st.columns(4)

			print("customer_neighbors_values:", customer_neighbors_values)
			is_customer_neighbors_in_default = 1 if customer_neighbors_values[0] < customer_neighbors_values[1] else 0
			print("is_customer_neighbors_in_default:", is_customer_neighbors_in_default)

			with col1:
				if customer_pred_default:
					customer_state = '<p style="font-family:sans-serif; color:red; font-size:35px;">En défaut</p>'
				else:
					customer_state = '<p style="font-family:sans-serif; color:green; font-size:35px;">En règle</p>'
				st.markdown(customer_state, unsafe_allow_html=True)
			with col2:
				neighbors_diff = \
					float(confidence_idx) - float(customer_neighbors_values[is_customer_neighbors_in_default])
				# print("neighbors_diff", neighbors_diff)
				# round(float(confidence_idx) - float(customer_neighbors_values[is_customer_neighbors_in_default]), 2)
				neighbors_delta = "{:.2f} %".format(float(neighbors_diff)*100)
				st.metric(label="indice de confiance", value=customer_idx, delta=neighbors_delta)
			with col3:
				if is_customer_neighbors_in_default:
					neighbors_state = '<p style="font-family:sans-serif; color:red; font-size:35px;">En défaut</p>'
				else:
					neighbors_state = '<p style="font-family:sans-serif; color:green; font-size:35px;">En règle</p>'
				st.markdown(neighbors_state, unsafe_allow_html=True)
			with col4:
				neighbors_idx = "{:.2f} %".format(float(customer_neighbors_values[is_customer_neighbors_in_default]) * 100)
				st.metric(label="moyenne de confiance", value=neighbors_idx)

			# print("Variables de session à converser pour comparaison avec profil modifié + évite rechargement via API\n")
			# enregistrement des valeurs du client initial
			# CONTROLES
			st.session_state.customer_pred = customer_pred_default
			# st.write("st.session_state.customer_pred:", st.session_state.customer_pred)
			st.session_state.customer_idx = customer_idx
			# st.write("st.session_state.customer_idx:", st.session_state.customer_idx)
			st.session_state.customer_diff = neighbors_diff
			# st.write("st.session_state.customer_diff:", st.session_state.customer_diff)
			st.session_state.customer_neigh_pred = is_customer_neighbors_in_default
			# st.write("st.session_state.customer_neigh_pred:", st.session_state.customer_neigh_pred)
			st.session_state.customer_neigh_idx = neighbors_idx
			# st.write("st.session_state.customer_neigh_idx:", st.session_state.customer_neigh_idx)
			st.session_state.customer_sliders = st.session_state.sliders
			# st.write("st.session_state.customer_sliders:", st.session_state.customer_sliders)
			# st.session_state.customer_total_feats
			# st.write("st.session_state.customer_total_feats:", st.session_state.customer_total_feats)
			# st.session_state.customer_total_feats_values
			# st.write("st.session_state.customer_total_feats_values:", st.session_state.customer_total_feats_values)
			st.session_state.customer_imp_feats = df_criteria['columns'].values.tolist()
			# st.write("st.session_state.customer_imp_feats:", st.session_state.customer_imp_feats)

		# profil normal (initial)
		else:
			col1, col2, col3, col4 = st.columns(4)

			if st.session_state.custom_profil:
				# si on compare le cient normal avec le profil modifié
				customer_pred_default = st.session_state.customer_pred
				is_customer_neighbors_in_default = st.session_state.customer_neigh_pred

				print("customer_neighbors_custom_values:", customer_neighbors_custom_values)
				is_customer_custom_neighbors_in_default = 1 \
					if customer_neighbors_custom_values[0] < customer_neighbors_custom_values[1] else 0
				print("is_customer_custom_neighbors_in_default:", is_customer_custom_neighbors_in_default)

			if not st.session_state.custom_profil:
				print("customer_neighbors_values:", customer_neighbors_values)
				is_customer_neighbors_in_default = 1 if customer_neighbors_values[0] < customer_neighbors_values[1] else 0
				print("is_customer_neighbors_in_default:", is_customer_neighbors_in_default)

			with col1:
				if customer_pred_default:
					customer_state = '<p style="font-family:sans-serif; color:red; font-size:35px;">En défaut</p>'
				else:
					customer_state = '<p style="font-family:sans-serif; color:green; font-size:35px;">En règle</p>'
				st.markdown(customer_state, unsafe_allow_html=True)

			with col2:
				if st.session_state.custom_profil:
					# si on compare le client normal avec le profil modifié
					customer_idx = st.session_state.customer_idx
					neighbors_delta = "{:.2f} %".format(st.session_state.customer_diff*100)
				else:
					neighbors_diff = \
						float(confidence_idx) - float(customer_neighbors_values[is_customer_neighbors_in_default])
					# round(float(confidence_idx) - float(customer_neighbors_values[is_customer_neighbors_in_default]), 2)
					neighbors_delta = "{:.2f} %".format(float(neighbors_diff)*100)

				st.metric(label="indice de confiance", value=customer_idx, delta=neighbors_delta)

			with col3:  # profil modifié
				if customer_pred_custom:
					customer_state_custom = '<p style="font-family:sans-serif; color:red; font-size:35px;">En défaut</p>'
				else:
					customer_state_custom = '<p style="font-family:sans-serif; color:green; font-size:35px;">En règle</p>'
				st.markdown(customer_state_custom, unsafe_allow_html=True)
			with col4:
				neighbors_custom_diff = \
					float(confidence_idx_custom) - float(
						customer_neighbors_custom_values[is_customer_custom_neighbors_in_default])
				# round(float(confidence_idx_custom) - float(customer_neighbors_custom_values[is_customer_custom_neighbors_in_default]), 2)
				neighbors_custom_delta = "{:.2f} %".format(float(neighbors_custom_diff)*100)
				st.metric(label="indice de confiance", value=customer_idx_custom, delta=neighbors_custom_delta)

			col_c, col_d = st.columns(2)
			with col_c:
				st.subheader("les 20 clients proches ? ")
			with col_d:
				st.subheader("les 20 clients proches ? ")

			col5, col6, col7, col8 = st.columns(4)

			with col5:
				if is_customer_neighbors_in_default:
					neighbors_state = '<p style="font-family:sans-serif; color:red; font-size:35px;">En défaut</p>'
				else:
					neighbors_state = '<p style="font-family:sans-serif; color:green; font-size:35px;">En règle</p>'
				st.markdown(neighbors_state, unsafe_allow_html=True)

			with col6:
				if st.session_state.custom_profil:
					# si on compare le client normal avec le profil modifié
					neighbors_idx = st.session_state.customer_neigh_idx
				else:
					neighbors_idx = "{:.2f} %".format(float(customer_neighbors_values[is_customer_neighbors_in_default]) * 100)

				st.metric(label="moyenne de confiance", value=neighbors_idx)

			with col7:  # prediction 20 proches clients profil modifié à faire
				if is_customer_custom_neighbors_in_default:
					neighbors_custom_state = '<p style="font-family:sans-serif; color:red; font-size:35px;">En défaut</p>'
				else:
					neighbors_custom_state = '<p style="font-family:sans-serif; color:green; font-size:35px;">En règle</p>'
				st.markdown(neighbors_custom_state, unsafe_allow_html=True)

			with col8:
				neighbors_custom_idx = "{:.2f} %".format(
					float(customer_neighbors_custom_values[is_customer_custom_neighbors_in_default]) * 100)
				st.metric(label="moyenne de confiance", value=neighbors_custom_idx)

		with st.expander("Signification et valeurs des critères de détermination du score", expanded=True):

			# Affichage des critères importants pour le client
			if not st.session_state.custom_profil:
				st.write("Critères du client initial")
				st.write(df_criteria.rename(columns={
					'feats_values': 'Valeur',
					'columns': 'Nom du critère',
					'tick_labels': 'Numéro'
				}))
			else:
				col_feats_1, col_feats_2 = st.columns(2)

				with col_feats_1:
					# normal profil avec comparaison
					st.write("Critères du client initial")
					st.write(df_criteria.rename(columns={
						'feats_values': 'Valeur',
						'columns': 'Nom du critère',
						'tick_labels': 'Numéro'
					}))

				with col_feats_2:
					# custom profil
					df_criteria_custom = df_criteria.copy()
					print("st.session_state.sliders:", st.session_state.sliders)
					for idx in range(len(st.session_state.sliders)):
						df_criteria_custom.loc[idx, 'feats_values'] = st.session_state.sliders[idx]
					# st.write("Critères les plus importants pour la prédiction")
					st.write("Critères du client avec paramètre(s) customisé(s)")
					st.write(df_criteria_custom.rename(columns={
						'feats_values': 'Valeur',
						'columns': 'Nom du critère',
						'tick_labels': 'Numéro'
					}))

			feats_color_list = ['red', 'orange', 'yellow', 'green']
			feats_color_dict = {}

			def display_feats_bar(df_bar, ajust_graph_colors=False):

				global feats_color_list, feats_color_dict

				plt.style.use('dark_background')
				fig, ax = plt.subplots(figsize=(8, 3))
				ax.patch.set_facecolor((14/255., 17/255., 23/255.))
				fig.patch.set_alpha(0.0)

				df_bar.sort_values(by='feats_values', inplace=True)
				feats_asc_order = df_bar['columns'].values.tolist()
				# st.write("feats_asc_order:", feats_asc_order)

				if ajust_graph_colors:
					feats_color_list = [
					    feats_color_dict[feat_name] for feat_name in feats_asc_order
					]
						# st.write("feats_color_list:", feats_color_list)

				ax.barh(df_bar['tick_labels'], df_bar['feats_values'], color=feats_color_list)

				# on associe une couleur à chaque variable
				for idx_color, feat_name in enumerate(df_bar['columns']):
					# st.write("idx_color:", idx_color, "feat_name:", feat_name)
					feats_color_dict[feat_name] = feats_color_list[idx_color]

				# st.write("feats_color_dict:", feats_color_dict)

				ax.set_xlabel("Valeurs normalisées")
				ax.set_ylabel("")
				ax.set_title("")

				return st.pyplot(fig)

			# Affichage du graphique (barchart) avec les features d'importances
			if not st.session_state.custom_profil:
				display_feats_bar(df_criteria)
			else:
				col_bc_1, col_bc_2 = st.columns(2)
				with col_bc_1:
					display_feats_bar(df_criteria)
				with col_bc_2:
					for idx in range(len(st.session_state.sliders)):
						df_criteria_custom.loc[idx, 'feats_values'] = st.session_state.sliders[idx]
					display_feats_bar(df_criteria_custom, ajust_graph_colors=True)

			# Affichage des sliders de customisation uniquement à partir d'un profil client (ID) existant
			# thresholds_customer_list = [float(round(value, 4)) for value in df_criteria['feats_values'].tolist()]
			# thresholds_customer_init = {key: value for key, value in enumerate(thresholds_customer_list)}
			# print("thresholds_customer_init:", thresholds_customer_init)

		# informations descriptives relatives à un client
		# création d'un dataframe client pour manipulation des données
		df_customer_all_feats = pd.DataFrame(data=customer_feats_values, columns=columns_feats)
		print(df_customer_all_feats.head())

		# fonction qui permet de peupler les selectbox avec une traduction
		def gen_customer_selectbox(feat_name, feat_trad, feat_key):

			feat_df = df_customer_all_feats[[*feat_trad]]
			feat_dict = feat_df.to_dict('list')
			feat_select = feat_df.columns.tolist()

			for name, value in feat_dict.items():
				if value[0]:  # si c'est la valeur active pour le client
					feat_select.remove(name)  # on enlève l'élément de la liste
					feat_select.insert(0, name)  # on ajoute l'élément en début de liste

			# On traduit les valeurs pour le select
			for idx, name in enumerate(feat_select):
				feat_select[idx] = feat_trad[name]

			# contrôles / vérifications
			# print("feat_df", feat_df)
			# print("feat_dict", feat_dict)
			# print("feat_select", feat_select)

			return st.selectbox(feat_name, feat_select, key=feat_key, on_change=enable_custom_profil)

		st.subheader("Informations descriptives relatives au(x) client(s)")

		col_info_1, col_info_2 = st.columns(2)

		with col_info_1:

			# Sexe
			gender_trad = {
				'CODE_GENDER__F': 'Féminin',
				'CODE_GENDER__M': 'Masculin',
				'CODE_GENDER__XNA': 'Non communiqué'
			}
			gender_selectbox = gen_customer_selectbox(
				"Sexe",
				gender_trad,
				'code_gender'
			)

			# Possède un véhicule ?
			car_trad = {
				'FLAG_OWN_CAR__N': 'Non',
				'FLAG_OWN_CAR__Y': 'Oui'
			}
			car_selectbox = gen_customer_selectbox(
				"Possède un véhicule ?",
				car_trad,
				'flag_own_car'
			)

			# Propriétaire d'un logement ?
			dwelling_trad = {
				'FLAG_OWN_REALTY__N': 'Non',
				'FLAG_OWN_REALTY__Y': 'Oui'
			}
			dwelling_selectbox = gen_customer_selectbox(
				"Propriétaire d'un logement ?",
				dwelling_trad,
				'flag_own_realty'
			)

			# Type de Prêt (espèces ou renouvelable) ?
			contract_trad = {
				'NAME_CONTRACT_TYPE__Cash loans': 'Espèces',
				'NAME_CONTRACT_TYPE__Revolving loans': 'Renouvelable'
			}
			contract_selectbox = gen_customer_selectbox(
				"Type de contrat",
				contract_trad,
				'name_contract_type'
			)

			# Niveau d'éducation
			education_trad = {
				'NAME_EDUCATION_TYPE__Academic degree': 'Universitaire',
				'NAME_EDUCATION_TYPE__Higher education': 'Enseignement supérieur',
				'NAME_EDUCATION_TYPE__Incomplete higher': 'Supérieur non achevé',
				'NAME_EDUCATION_TYPE__Lower secondary': 'Secondaire inférieur',
				'NAME_EDUCATION_TYPE__Secondary / secondary special': 'Secondaire spécial',
			}
			education_selectbox = gen_customer_selectbox(
				"Niveau d'éducation",
				education_trad,
				'name_education_type'
			)

			# Situation familiale
			family_trad = {
				'NAME_FAMILY_STATUS__Civil marriage': 'Mariage civil',
				'NAME_FAMILY_STATUS__Married': 'Marié(e)',
				'NAME_FAMILY_STATUS__Separated': 'Séparé(e)',
				'NAME_FAMILY_STATUS__Single / not married': 'Célibataire / non marié(e)',
				'NAME_FAMILY_STATUS__Unknown': 'Non communiqué',
				'NAME_FAMILY_STATUS__Widow': 'Veuf / Veuve',
			}
			family_selectbox = gen_customer_selectbox(
				"Situation familiale",
				family_trad,
				'name_family_status'
			)

			# Type de logement
			housing_trad = {
				'NAME_HOUSING_TYPE__Co-op apartment': 'Appartement en colocation',
				'NAME_HOUSING_TYPE__House / apartment': 'Maison / Appartement',
				'NAME_HOUSING_TYPE__Municipal apartment': 'Appartement municipal',
				'NAME_HOUSING_TYPE__Office apartment': 'Appartement de bureau',
				'NAME_HOUSING_TYPE__Rented apartment': 'Appartement loué ',
				'NAME_HOUSING_TYPE__With parents': 'Chez les parents',
			}
			housing_selectbox = gen_customer_selectbox(
				"Type de logement occupé",
				housing_trad,
				'name_housing_type'
			)

			# Catégorie de revenu
			income_trad = {
				'NAME_INCOME_TYPE__Businessman': 'Homme / Femme d\'affaire',
				'NAME_INCOME_TYPE__Commercial associate': 'Associé(e) commercial',
				'NAME_INCOME_TYPE__Maternity leave': 'Congé maternité / Paternité',
				'NAME_INCOME_TYPE__Pensioner': 'Retraité(e)',
				'NAME_INCOME_TYPE__State servant': 'Employé de l\'Etat',
				'NAME_INCOME_TYPE__Student': 'Etudiant(e)',
				'NAME_INCOME_TYPE__Unemployed': 'Sans emploi',
				'NAME_INCOME_TYPE__Working': 'Actif',
			}
			income_selectbox = gen_customer_selectbox(
				"Catégorie de revenu",
				income_trad,
				'name_income_type'
			)

			# Type de profession
			occupation_trad = {
				'OCCUPATION_TYPE__Accountants': 'Comptable',
				'OCCUPATION_TYPE__Cleaning staff': 'Personnel de nettoyage',
				'OCCUPATION_TYPE__Cooking staff': 'Personnel de cuisine',
				'OCCUPATION_TYPE__Core staff': 'Personnel de base',
				'OCCUPATION_TYPE__Drivers': 'Conducteur / Conductrice',
				'OCCUPATION_TYPE__HR staff': 'Personnel RH',
				'OCCUPATION_TYPE__High skill tech staff': 'Personnel technique qualifié',
				'OCCUPATION_TYPE__IT staff': 'Personnel Informatique',
				'OCCUPATION_TYPE__Laborers': 'Ouvrier / Ouvrière',
				'OCCUPATION_TYPE__Low-skill Laborers': 'Ouvrier(e) peu qualifié(e)',
				'OCCUPATION_TYPE__Managers': 'Gestionnaire',
				'OCCUPATION_TYPE__Medicine staff': 'Personnel médical',
				'OCCUPATION_TYPE__Private service staff': 'Personnel de service privé',
				'OCCUPATION_TYPE__Realty agents': 'Agent immobilier',
				'OCCUPATION_TYPE__Sales staff': 'Personnel de vente',
				'OCCUPATION_TYPE__Secretaries': 'Secrétaire',
				'OCCUPATION_TYPE__Security staff': 'Personnel de sécurité',
				'OCCUPATION_TYPE__UNSPECIFIED': 'Non communiqué',
				'OCCUPATION_TYPE__Waiters/barmen staff': 'Serveur / Barmaid',
			}
			occupation_selectbox = gen_customer_selectbox(
				"Type de profession",
				occupation_trad,
				'occupation_type'
			)

			# Type d'organisation
			organization_trad = {
				'ORGANIZATION_TYPE__Advertising': 'Publicité',
				'ORGANIZATION_TYPE__Agriculture': 'Agriculture',
				'ORGANIZATION_TYPE__Bank': 'Banque',
				'ORGANIZATION_TYPE__Business Entity Type 1': 'Commerciale Type 1',
				'ORGANIZATION_TYPE__Business Entity Type 2': 'Commerciale Type 2',
				'ORGANIZATION_TYPE__Business Entity Type 3': 'Commerciale Type 3',
				'ORGANIZATION_TYPE__Cleaning': 'Nettoyage',
				'ORGANIZATION_TYPE__Construction': 'Construction',
				'ORGANIZATION_TYPE__Culture': 'Culturelle',
				'ORGANIZATION_TYPE__Electricity': 'Electricité',
				'ORGANIZATION_TYPE__Emergency': 'Urgences',
				'ORGANIZATION_TYPE__Government': 'État',
				'ORGANIZATION_TYPE__Hotel': 'Hôtellerie',
				'ORGANIZATION_TYPE__Housing': 'l\'Habitat',
				'ORGANIZATION_TYPE__Industry: type 1': 'Industrie Type 1',
				'ORGANIZATION_TYPE__Industry: type 10': 'Industrie Type 10',
				'ORGANIZATION_TYPE__Industry: type 11': 'Industrie Type 11',
				'ORGANIZATION_TYPE__Industry: type 12': 'Industrie Type 12',
				'ORGANIZATION_TYPE__Industry: type 13': 'Industrie Type 13',
				'ORGANIZATION_TYPE__Industry: type 2': 'Industrie Type 2',
				'ORGANIZATION_TYPE__Industry: type 3': 'Industrie Type 3',
				'ORGANIZATION_TYPE__Industry: type 4': 'Industrie Type 4',
				'ORGANIZATION_TYPE__Industry: type 5': 'Industrie Type 5',
				'ORGANIZATION_TYPE__Industry: type 6': 'Industrie Type 6',
				'ORGANIZATION_TYPE__Industry: type 7': 'Industrie Type 7',
				'ORGANIZATION_TYPE__Industry: type 8': 'Industrie Type 8',
				'ORGANIZATION_TYPE__Industry: type 9': 'Industrie Type 9',
				'ORGANIZATION_TYPE__Insurance': 'Assurances',
				'ORGANIZATION_TYPE__Kindergarten': 'Jardin d\'enfants',
				'ORGANIZATION_TYPE__Legal Services': 'Services juridiques',
				'ORGANIZATION_TYPE__Medicine': 'Médicale',
				'ORGANIZATION_TYPE__Military': 'Militaire',
				'ORGANIZATION_TYPE__Mobile': 'Téléphonie',
				'ORGANIZATION_TYPE__Other': 'Autre',
				'ORGANIZATION_TYPE__Police': 'Police',
				'ORGANIZATION_TYPE__Postal': 'Postale',
				'ORGANIZATION_TYPE__Realtor': 'Immobilière',
				'ORGANIZATION_TYPE__Religion': 'Religieuse',
				'ORGANIZATION_TYPE__Restaurant': 'Restauration',
				'ORGANIZATION_TYPE__School': 'Scolaire',
				'ORGANIZATION_TYPE__Security': 'Sécurité',
				'ORGANIZATION_TYPE__Security Ministries': 'Ministère de la sécurité',
				'ORGANIZATION_TYPE__Self-employed': 'Travailleur indépendant',
				'ORGANIZATION_TYPE__Services': 'Tertiaire',
				'ORGANIZATION_TYPE__Telecom': 'Télécoms',
				'ORGANIZATION_TYPE__Trade: type 1': 'Commerce Type 1',
				'ORGANIZATION_TYPE__Trade: type 2': 'Commerce Type 2',
				'ORGANIZATION_TYPE__Trade: type 3': 'Commerce Type 3',
				'ORGANIZATION_TYPE__Trade: type 4': 'Commerce Type 4',
				'ORGANIZATION_TYPE__Trade: type 5': 'Commerce Type 5',
				'ORGANIZATION_TYPE__Trade: type 6': 'Commerce Type 6',
				'ORGANIZATION_TYPE__Trade: type 7': 'Commerce Type 7',
				'ORGANIZATION_TYPE__Transport: type 1': 'Transport Type 1',
				'ORGANIZATION_TYPE__Transport: type 2': 'Transport Type 2',
				'ORGANIZATION_TYPE__Transport: type 3': 'Transport Type 3',
				'ORGANIZATION_TYPE__Transport: type 4': 'Transport Type 4',
				'ORGANIZATION_TYPE__University': 'Universitaire',
				'ORGANIZATION_TYPE__XNA': 'Non communiquée',
			}
			organization_selectbox = gen_customer_selectbox(
				"Type d'organisation",
				organization_trad,
				'organization_type'
			)

			# feats_trad_list = [
			# 	gender_trad,
			# 	car_trad,
			# 	dwelling_trad,
			# 	contract_trad,
			# 	education_trad,
			# 	family_trad,
			# 	housing_trad,
			# 	income_trad,
			# 	occupation_trad,
			# 	organization_trad
			# ]

			# customer_chars_col_to_list = []
			# for dict in feats_trad_list:
			# 	for key in dict.keys():
			# 		customer_chars_col_to_list.append(key)
			#
			# customer_chars_file = open(script_dir + '/data/customer_chars_col_to_list.txt', 'w')
			# str_customer_chars_col_to_list = repr(customer_chars_col_to_list)
			# with customer_chars_file as file:
			# 	file.write(str_customer_chars_col_to_list)
			# 	file.close()

		with col_info_2:
			# st.write(customer_neighbors_chars_sum)  # dictionary
			# st.write(customer_neighbors_custom_chars_sum)  # dictionary

			def gen_neighbors_char_graph(feat_trad, neighbors_chars_sum, graph_type):

				plt.style.use('dark_background')

				feat_sum = {feat_value: neighbors_chars_sum[feat_name] for feat_name, feat_value in feat_trad.items()}

				# st.session_state.code_gender

				if graph_type == "pie":

					fig, ax = plt.subplots(figsize=(8, 4))
					fig.patch.set_alpha(0.0)
					ax.patch.set_facecolor((14 / 255., 17 / 255., 23 / 255.))

					# récupération de la clé avec la somme maximum
					max_feat_value = max(feat_sum.values())
					max_feat_key = [key for key, value in feat_sum.items() if value == max_feat_value][0]
					# print("max_feat_key :", max_feat_key)

					explode_pie = tuple(0 if feat_name != max_feat_key else 0.1
					                    for feat_name in feat_trad.values())
					# print("explode_pie: ", explode_pie)

					# Rappel: l'objet pie retourne patches, texts, autotexts
					_, _, autotexts = ax.pie(
						x=list(feat_sum.values()),
						labels=[label if feat_sum[label] != 0 else '' for label in list(feat_sum.keys())],
						startangle=90,
						# autopct='%.0f%%',
						autopct=lambda x: "{:.0f}%".format(round(x)) if x > 0 else "",  # cache les valeurs nulles
						explode=explode_pie,
						# shadow=True,
					)

					plt.setp(autotexts, **{'color': 'black', 'weight': 'bold', 'fontsize': 12})

				elif graph_type == "simple_barh":

					fig, ax = plt.subplots(figsize=(8, 1.5))
					fig.patch.set_alpha(0.0)
					ax.patch.set_facecolor((14 / 255., 17 / 255., 23 / 255.))

					feat_label = [""]

					ax.barh(feat_label, [list(feat_sum.values())[0]], label=list(feat_sum.keys())[0])
					ax.barh(
						feat_label,
						[list(feat_sum.values())[1]],
						left=[list(feat_sum.values())[0]],
						label=list(feat_sum.keys())[1]
					)

					for p_index, p in enumerate(ax.patches):
						width, height = p.get_width(), p.get_height()
						x, y = p.get_xy()
						ax.text(
							x + width / 2,
							y + height / 2,
							'{}: {}'.format(list(feat_sum.keys())[p_index], int(list(feat_sum.values())[p_index]))
							if p.get_width() > 0 else "",
							ha='center',
							va='center',
							color='black',
							weight='bold',
							size=15
						)
					plt.axis('off')

				elif graph_type == 'bar':

					if len(list(feat_sum.keys())) == 8:
						fig, ax = plt.subplots(figsize=(4, 6))
					elif len(list(feat_sum.keys())) == 19:
						fig, ax = plt.subplots(figsize=(4, 12))
					# elif len(list(feat_sum.keys())) == 58:
						# fig, ax = plt.subplots(figsize=(4, 20))
					else:
						fig, ax = plt.subplots(figsize=(4, 4))

					fig.patch.set_alpha(0.0)
					ax.patch.set_facecolor((14 / 255., 17 / 255., 23 / 255.))

					# ax.barh(list(feat_sum.keys()), list(feat_sum.values()))
					ax = sns.barplot(x=list(feat_sum.values()), y=list(feat_sum.keys()))

					for p in ax.patches:
						ax.annotate(
							"{:.0f}".format(p.get_width()) if p.get_width() > 0 else "",
							(p.get_width(), p.get_y()),
							ha='center',
							va='center',
							fontsize=15,
							weight='bold',
							color='black',
							xytext=(-12, -20) if p.get_width() > 1 else (-7, -20),
							textcoords='offset points'
						)

					plt.yticks(fontsize=15)
					plt.xticks([])

				return st.pyplot(fig)

			# Affichage des tendances des 20 plus proches clients en fonction du profil client (modifié & non modifié)
			if st.session_state.custom_profil:
				neighbors_chars_sum_dict = customer_neighbors_custom_chars_sum
			else:
				neighbors_chars_sum_dict = customer_neighbors_chars_sum

			st.write("")
			st.write("")
			with st.expander("Sexe des 20 clients proches"):
				gen_neighbors_char_graph(gender_trad, neighbors_chars_sum_dict, graph_type='pie')

			st.write("")
			st.write("")
			with st.expander("Véhicule pour les 20 clients proches ?"):
				gen_neighbors_char_graph(car_trad, neighbors_chars_sum_dict, graph_type='simple_barh')

			st.write("")
			st.write("")
			with st.expander("Logement pour les 20 clients proches ?"):
				gen_neighbors_char_graph(dwelling_trad, neighbors_chars_sum_dict, graph_type='simple_barh')

			st.write("")
			st.write("")
			with st.expander("Type de contrat des 20 clients proches ?"):
				gen_neighbors_char_graph(contract_trad, neighbors_chars_sum_dict, graph_type='pie')

			st.write("")
			st.write("")
			with st.expander("Niveau d'éducation des 20 clients proches ?"):
				gen_neighbors_char_graph(education_trad, neighbors_chars_sum_dict, graph_type='bar')

			st.write("")
			st.write("")
			with st.expander("Situation familiale des 20 clients proches ?"):
				gen_neighbors_char_graph(family_trad, neighbors_chars_sum_dict, graph_type='bar')

			st.write("")
			st.write("")
			with st.expander("Type de logement des 20 clients proches ?"):
				gen_neighbors_char_graph(housing_trad, neighbors_chars_sum_dict, graph_type='bar')

			st.write("")
			st.write("")
			with st.expander("Catégorie de revenu des 20 clients proches ?"):
				gen_neighbors_char_graph(income_trad, neighbors_chars_sum_dict, graph_type='pie')

			st.write("")
			st.write("")
			with st.expander("Profession des 20 clients proches ?"):
				gen_neighbors_char_graph(occupation_trad, neighbors_chars_sum_dict, graph_type='bar')

			st.write("")
			st.write("")
			with st.expander("Type d'organisation des 20 clients proches ?"):
				gen_neighbors_char_graph(organization_trad, neighbors_chars_sum_dict, graph_type='pie')


		def gen_df_custom_spec_feats(feat_trad, session_state_var):

			# réutilisation du dictionnaire de traduction
			feat_data = feat_trad

			# permet de récupérer le nom de la variable concernée à partir de la liste déroulante
			feat_value = [*feat_trad][[*feat_trad.values()].index(session_state_var)]

			# permet de réaffecter les valeurs boolénnes aux variables initiales pour reconstruire les données utilisateur
			feat_data = [{key: 1} if key == feat_value else {key: 0} for key, value in feat_data.items()]

			# création d'un dataframe avec les données de la variable utilisateur
			feat_values = {}
			for elt_data in feat_data:
				feat_values.update(elt_data)

			return pd.DataFrame(feat_values, index=[0])

		pg_bar.progress(100)
		pg_bar.empty()

	else:
		pg_bar.empty()
		st.sidebar.text("Merci d'indiquer un ID client valide.")

hide_st_block = """
		<style>
		footer {visibility: hidden;}
		</style>
"""

st.markdown(hide_st_block, unsafe_allow_html=True)

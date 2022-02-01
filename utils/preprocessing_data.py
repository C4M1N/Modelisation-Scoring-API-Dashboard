import os
import gzip
import pickle


script_dir = os.path.dirname(__file__)


def load_df(filename):
	# dataframe loading
	with gzip.open(os.path.join(script_dir, '../data/' + filename), 'rb') as df:
		df_custom_global_optim_std_w_id = pickle.load(df)

	print(df_custom_global_optim_std_w_id.shape)

	# verification
	print(df_custom_global_optim_std_w_id['TARGET'].head())

	# make a copy of original dataframe for manipulation
	df = df_custom_global_optim_std_w_id.copy()
	print(df.head())

	# retrieve column names for future functionality (categorical selection on customer's features)
	# model_columns = df.columns

	return df


def retrieve_customers_id(df):

	customer_id_list = df['SK_ID_CURR'].tolist()

	return customer_id_list


def retrieve_customers_features(df, customer_id):

	# On drop la colonne TARGET si elle existe pour cet ID client
	df = df.drop(['TARGET'], axis=1, errors='ignore')
	customer_features_list = df[df['SK_ID_CURR'] == customer_id].values.tolist()
	columns_features_list = df.columns.tolist()

	return customer_features_list, columns_features_list

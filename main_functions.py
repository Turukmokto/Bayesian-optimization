from utils import *

cols = ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf']


def random_search_algorithm(primary_df_real):
    df_random = pd.DataFrame(columns=cols + ['f1'])
    df_random = random_search(primary_df_real, 1600, df_random)
    print(df_random.iloc[df_random['f1'].idxmax()])
    return df_random


def hyper_param_df_real_scores(df_data, primary_df):
    primary_df_real = primary_df.copy()
    primary_df_real['f1'] = np.nan
    for index in range(0, len(primary_df)):
        params = primary_df.iloc[index]
        score = honest_count(params, df_data)
        primary_df_real.at[index, 'f1'] = score
    print(primary_df_real.iloc[primary_df_real['f1'].idxmax()])
    return primary_df_real


def surrogate_train(df_data, df_surrogate, primary_df):
    lambda_cur = 1
    step_update = []
    step_results = []
    for i in range(30):
        df_surrogate = initialization(primary_df, df_surrogate, 10)
        df_surrogate = for_real(df_data, df_surrogate)
        update = enhancement_function(df_surrogate, primary_df, lambda_cur)
        step_update.append(update)
        index_max = update.index(max(update))
        df_surrogate = df_surrogate.append(primary_df.iloc[index_max], ignore_index=True)
        df_surrogate = for_real(df_data, df_surrogate)
        step_result = surrogate_fun(df_surrogate).predict(primary_df)
        step_results.append(step_result)
        lambda_cur = lamda_up(i, 1)
    return df_surrogate, step_result, step_results, step_update


def hyper_param_df_creation():
    criteria = ['gini', 'entropy']
    max_depths = list(range(2, 25, 2))
    min_samples_splits = list(range(2, 12, 1))
    min_samples_leafs = list(range(2, 12, 1))
    primary_df = pd.DataFrame(columns=cols)
    for criteria in criteria:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    row = pd.DataFrame(
                        {'criterion': [criteria], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split],
                         'min_samples_leaf': [min_samples_leaf]})
                    primary_df = pd.concat([primary_df, row], ignore_index=True, axis=0)
    return primary_df

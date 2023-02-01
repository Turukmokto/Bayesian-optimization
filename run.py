import warnings

from grafics import *
from main_functions import *

warnings.filterwarnings('ignore')


def main():
    name = '/Users/esbessonngmail.com/Downloads/OpenML/data/997.arff'
    df_data = arff_load(name)

    hyper_param_df = hyper_param_df_creation()

    hyper_param_df['criterion'] = pd.factorize(hyper_param_df['criterion'])[0]
    surrogate_df = pd.DataFrame(columns=['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'f1'])

    surrogate_df, step_result, step_results, step_update = surrogate_train(df_data, surrogate_df, hyper_param_df)

    hyper_param_df_real = hyper_param_df_real_scores(df_data, hyper_param_df)

    df_random = random_search_algorithm(hyper_param_df_real)

    col = pd.DataFrame({'f1': step_result})
    hyper_param_df_pred = pd.concat([hyper_param_df, col], axis=1)
    print(hyper_param_df_pred)

    grafics_check_with_real_scores(hyper_param_df_pred, hyper_param_df_real)

    grafics_comparing_with_random_search(df_random, surrogate_df)

    grafics_objective_function_with_steps(df_random, step_update)


if __name__ == '__main__':
    main()

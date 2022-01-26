from utils import learning_loop

if __name__ == "__main__":
    dataset = ['Synthie', 'BZR_MD', 'COX2_MD', 'DHFR', 'PTC_MM', 'PTC_MR', 'PTC_FM', 'PTC_FR', 'ENZYMES', 'KKI',
               'IMDB-BINARY', 'IMDB-MULTI']
    hasnodelabel = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    for ds_name, has in zip(dataset, hasnodelabel):
        print(f'Dataset:\t{ds_name}')
        for ft in (1, 2, 3):
            for it in (0, 1, 2, 3):
                print(f'feature: {ft},  imp: {it}')
                learning_loop(ds_name, feature_type=ft, importance_type=it, hasnl=has, filter_size=3, graphlet_size=5,
                              max_h=2, k_folds=10, epochs=200, batch_size=32, lr=1.e-3)

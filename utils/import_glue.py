from trainers.options import params

# Paired sentences
if params.task in ['rte', 'mrpc', 'qqp', 'qnli', 'mnli']:
    from utils.datasets_encoders import rte_batch_data as batch_data
    from sklearn.metrics import accuracy_score as score_func
# Singe sentence
elif params.task in ['sst2', 'cola']:
    from utils.datasets_encoders import cola_batch_data as batch_data

    if params.task in ['sst2']:
        from sklearn.metrics import accuracy_score as score_func
    elif params.task in ['cola']:
        from sklearn.metrics import matthews_corrcoef as score_func
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

if params.task in ['rte', 'mrpc']:
    from utils.datasets_encoders import rte_encoder as encode
elif params.task == 'qqp':
    from utils.datasets_encoders import qqp_encoder as encode
elif params.task == 'qnli':
    from utils.datasets_encoders import qnli_encoder as encode
elif params.task == 'mnli':
    from utils.datasets_encoders import mnli_encoder as encode
elif params.task in ['sst2']:
    from utils.datasets_encoders import cola_encode as encode
elif params.task in ['cola']:
    from utils.datasets_encoders import cola_encode as encode
else:
    raise NotImplementedError



import os, random, re
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve, auc

    
# Auxiliary functions
def color_map(val):
    if type(val) == float:
        if val <= 0.5:
            color = 'red'
        elif val <= 0.6:
            color = 'orange'
        elif val >= 0.99:
            color = 'green'
        else:
            color = 'black'
    else:
        color = 'black'
    return 'color: %s' % color

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
def set_up_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    return strategy, tpu

def plot_metrics(history):
    metric_list = list(history.keys())
    size = len(metric_list)//2
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 5))
    axes = axes.flatten()
    
    for index in range(len(metric_list)//2):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+size]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)
        if 'loss' in metric_name:
            axes[index].axvline(np.argmin(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmin(history[val_metric_name]), linestyle='dashed', color='orange')
        else:
            axes[index].axvline(np.argmax(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmax(history[val_metric_name]), linestyle='dashed', color='orange')
            

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
def plot_metrics_agg(history_list, n_folds):
    metric_list = list(history_list[0].keys())
    size = len(metric_list)
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 5))
    axes = axes.flatten()
    
    for idx, history in enumerate(history_list):
        for index, metric_name in enumerate(metric_list):
            axes[index].plot([v for k,v in history.items() if k.startswith(metric_name)][0], label=f'Fold {idx+1} {metric_name}')
            axes[index].legend(loc='best', fontsize=16)
            axes[index].set_title(metric_name)

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
# Model evaluation
def evaluate_model(k_fold, n_folds=1):
    metrics_df = pd.DataFrame([], columns=['Metric', 'Train', 'Valid', 'Var'])
    metrics_df['Metric'] = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Support']
    
    for n_fold in range(n_folds):
        rows = []
        n_fold += 1
        fold_col = f'fold_{n_fold}' 
        pred_col = f'pred_fold_{n_fold}' 
        train_set = k_fold[k_fold[fold_col] == 'train']
        valid_set = k_fold[k_fold[fold_col] == 'validation'] 
        
        train_report = classification_report(train_set['target'], np.round(train_set[pred_col]), output_dict=True)
        valid_report = classification_report(valid_set['target'], np.round(valid_set[pred_col]), output_dict=True)
    
        rows.append([roc_auc_score(train_set['target'], train_set[pred_col]),
                     roc_auc_score(valid_set['target'], valid_set[pred_col])])
        rows.append([train_report['accuracy'], valid_report['accuracy']])
        rows.append([train_report['1']['precision'], valid_report['1']['precision']])
        rows.append([train_report['1']['recall'], valid_report['1']['recall']])
        rows.append([train_report['1']['f1-score'], valid_report['1']['f1-score']])
        rows.append([train_report['1']['support'], valid_report['1']['support']])
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame(rows, columns=[f'Train_fold_{n_fold}', 
                                                                        f'Valid_fold_{n_fold}'])], axis=1)
    
    metrics_df['Train'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Train_fold')]].mean(axis=1)
    metrics_df['Valid'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Valid_fold')]].mean(axis=1)
    metrics_df['Var'] = metrics_df['Train'] - metrics_df['Valid']
    
    return metrics_df.set_index('Metric')

def evaluate_model_Subset(k_fold, n_folds):
    metrics_df = pd.DataFrame([], columns=['Subset/ROC AUC', 'Train', 'Valid', 'Var'])
    
    lower_bound = [0, 26, 40, 60]
    upper_bound = [26, 40, 60, 120]
    sex_values = ['male', 'female']
    anatom_values = ['head/neck', 'upper extremity', 'lower extremity', 'torso']
    labels = (['Overall'] + sex_values + anatom_values + 
              [f'{lower} <= age < {upper_bound[idx]}' for idx, lower in enumerate(lower_bound)])
    metrics_df['Subset/ROC AUC'] = labels
    
    for n_fold in range(n_folds):
        rows = []
        n_fold += 1
        fold_col = f'fold_{n_fold}' 
        pred_col = f'pred_fold_{n_fold}' 
        train_set = k_fold[k_fold[fold_col] == 'train']
        valid_set = k_fold[k_fold[fold_col] == 'validation']        
        
        rows.append([roc_auc_score(train_set['target'], train_set[pred_col]),
                     roc_auc_score(valid_set['target'], valid_set[pred_col])])

        for sex in sex_values:
            train_subset = train_set[train_set['sex'] == sex]
            valid_subset = valid_set[valid_set['sex'] == sex]
            rows.append([roc_auc_score(train_subset['target'], train_subset[pred_col]),
                         roc_auc_score(valid_subset['target'], valid_subset[pred_col])])

        for anatom in anatom_values:
            train_subset = train_set[train_set['anatom_site_general_challenge'] == anatom]
            valid_subset = valid_set[valid_set['anatom_site_general_challenge'] == anatom]
            rows.append([roc_auc_score(train_subset['target'], train_subset[pred_col]),
                         roc_auc_score(valid_subset['target'], valid_subset[pred_col])])

        for idx, lower in enumerate(lower_bound):
            upper = upper_bound[idx]
            train_subset = train_set[(train_set['age_approx'] >= lower) & 
                                     (train_set['age_approx'] < upper)]
            valid_subset = valid_set[(valid_set['age_approx'] >= lower) & 
                                     (valid_set['age_approx'] < upper)]
            rows.append([roc_auc_score(train_subset['target'], train_subset[pred_col]),
                         roc_auc_score(valid_subset['target'], valid_subset[pred_col])])

        metrics_df = pd.concat([metrics_df, pd.DataFrame(rows, columns=[f'Train_fold_{n_fold}', 
                                                                        f'Valid_fold_{n_fold}'])], axis=1)
    
    metrics_df['Train'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Train_fold')]].mean(axis=1)
    metrics_df['Valid'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Valid_fold')]].mean(axis=1)
    metrics_df['Var'] = metrics_df['Train'] - metrics_df['Valid']
    
    return metrics_df.set_index('Subset/ROC AUC')

def plot_confusion_matrix(y_train, train_pred, y_valid, valid_pred, labels=[0, 1]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    train_cnf_matrix = confusion_matrix(y_train, train_pred)
    validation_cnf_matrix = confusion_matrix(y_valid, valid_pred)

    train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]
    validation_cnf_matrix_norm = validation_cnf_matrix.astype('float') / validation_cnf_matrix.sum(axis=1)[:, np.newaxis]

    train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)
    validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=labels, columns=labels)

    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax1).set_title('Train')
    sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8),ax=ax2).set_title('Validation')
    plt.show()
    
def plot_auc_curve(y_train, train_pred, y_valid, valid_pred):
    fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_valid, tpr_valid, _ = roc_curve(y_valid, valid_pred)
    roc_auc_valid = auc(fpr_valid, tpr_valid)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_train, tpr_train, color='blue', label='Train AUC = %0.2f' % roc_auc_train)
    plt.plot(fpr_valid, tpr_valid, color='purple', label='ValidationAUC = %0.2f' % roc_auc_valid)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
# Datasets utility functions
LABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    "target": tf.io.FixedLenFeature([], tf.int64), # shape [] means single element
    "image_name": tf.io.FixedLenFeature([], tf.string),
    # meta features
    "patient_id": tf.io.FixedLenFeature([], tf.int64),
    "sex": tf.io.FixedLenFeature([], tf.int64),
    "age_approx": tf.io.FixedLenFeature([], tf.int64),
    "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),
    "diagnosis": tf.io.FixedLenFeature([], tf.int64)
}

UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    "image_name": tf.io.FixedLenFeature([], tf.string), # shape [] means single element
    # meta features
    "patient_id": tf.io.FixedLenFeature([], tf.int64),
    "sex": tf.io.FixedLenFeature([], tf.int64),
    "age_approx": tf.io.FixedLenFeature([], tf.int64),
    "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),
}

def decode_image(image_data, height, width, channels):
    image = tf.image.decode_jpeg(image_data, channels=channels)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [height, width, channels])
    return image

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
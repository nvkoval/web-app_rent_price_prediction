import pandas as pd
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.models.model_evaluation import run_cv


# Define tree-based models for automatic detection
TREE_BASED_MODELS = (
    RandomForestRegressor,
    ExtraTreesRegressor,
    DecisionTreeRegressor,
    XGBRegressor,
    LGBMRegressor,
)


def create_pipeline(
    model: BaseEstimator,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    encoding_strategy: str = 'none',
    onehot_features: Optional[List[str]] = None,
    ordinal_categories: Optional[Dict[str, List[str]]] = None,
) -> Pipeline:
    """
    Pipeline creation function that handles all encoding strategies and model types.

    Parameters
    ----------
    model : BaseEstimator
        The ML model to include at the end of the pipeline.
    numerical_features : list, optional
        List of numerical column names. If None, no numerical features are processed.
    categorical_features : list, optional
        List of categorical column names. If None, no categorical features are processed.
    encoding_strategy : str, default='none'
        Encoding strategy for categorical features. Options:
        - 'none': No categorical encoding (numerical features only)
        - 'onehot': One-hot encoding for all categorical features
        - 'ordinal': Ordinal encoding for all categorical features
        - 'mixed': Mixed encoding strategy using onehot_features and ordinal_categories
    onehot_features : list, optional
        Features to be encoded with OneHotEncoder (used with encoding_strategy='mixed').
    ordinal_categories : dict, optional
        Dictionary mapping categorical column names to ordered lists of categories
        (used with encoding_strategy='mixed').

    Returns
    -------
    Pipeline
        Complete preprocessing + modeling pipeline.
    """

    # Validate encoding_strategy
    if encoding_strategy not in ['none', 'onehot', 'ordinal', 'mixed']:
        raise ValueError(
            f"encoding_strategy must be one of ['none', 'onehot', 'ordinal', 'mixed'], "
            f"got {encoding_strategy}"
        )

    if encoding_strategy == 'mixed':
        if onehot_features is None and ordinal_categories is None:
            raise ValueError(
                "For encoding_strategy='mixed', at least one of onehot_features or "
                "ordinal_categories must be provided"
            )

    # Initialize empty lists if None
    numerical_features = numerical_features or []
    categorical_features = categorical_features or []
    onehot_features = onehot_features or []
    ordinal_categories = ordinal_categories or {}

    # Determine if model is tree-based
    is_tree_based = isinstance(model, TREE_BASED_MODELS)

    # Build transformers list
    transformers = []

    # Handle numerical features
    if numerical_features:
        if is_tree_based:
            # Tree-based models: pass numerical features as-is
            transformers.append(
                ('numerical', 'passthrough', numerical_features)
            )
        else:
            # Linear models: impute and scale numerical features
            transformers.append((
                'num_impute_scale',
                Pipeline([
                    ('imputer', SimpleImputer()),
                    ('scaler', StandardScaler())
                ]),
                numerical_features
            ))

    # Handle categorical features based on encoding strategy
    if encoding_strategy == 'none':
        # No categorical encoding - only numerical features
        pass

    elif encoding_strategy == 'onehot' and categorical_features:
        # One-hot encoding for all categorical features
        onehot_encoder = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )
        transformers.append(
            ('categorical', onehot_encoder, categorical_features)
        )

    elif encoding_strategy == 'ordinal' and categorical_features:
        # Ordinal encoding for all categorical features
        ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1
        )
        transformers.append(
            ('categorical', ordinal_encoder, categorical_features)
        )

    elif encoding_strategy == 'mixed':
        # Mixed encoding strategy
        # One-hot encoding for specified features
        if onehot_features:
            onehot_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            )
            transformers.append(
                ('onehot', onehot_encoder, onehot_features)
            )

        # Ordinal encoding for specified features
        if ordinal_categories:
            ordinal_features = list(ordinal_categories.keys())
            categories = list(ordinal_categories.values())
            ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                categories=categories,
                unknown_value=-1,
                encoded_missing_value=-1
            )
            transformers.append(
                ('ordinal', ordinal_encoder, ordinal_features)
            )

    # Create preprocessor
    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop' if not is_tree_based else 'passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')

        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # No preprocessing needed (e.g., numeric-only tree models)
        pipeline = Pipeline([('model', model)])

    return pipeline


def evaluate_models(models, X, y, pipeline_configs, results_dict) -> pd.DataFrame:
    """
    Model evaluation function.

    Parameters:
    -----------
    models : dict
        Dictionary of model name: model instance pairs
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    pipeline_config : dict
        Configuration for pipeline type
    results_dict : dict
        Dictionary to store results

    Returns
        -------
        df_results : Dataframe
            Dataframe with results.
    """
    for model_name, model in models.items():
        print(f"\n{model_name}")
        for config_name, config in pipeline_configs.items():
            pipeline = create_pipeline(model, **config)
            results_dict[f"{config_name} {model_name}"] = run_cv(pipeline, X, y)

    df_results = pd.DataFrame(results_dict.items(), columns=['model', 'RMSLE'])
    df_results = df_results.sort_values('RMSLE').round(4)
    return df_results

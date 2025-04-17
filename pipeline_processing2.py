# Required Imports (ensure these are present)
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
# check_is_fitted is used in SpectralPreprocessor's original code
from sklearn.utils.validation import check_is_fitted

# --- Placeholder for potentially external helper functions ---
# These would need to be defined or imported elsewhere as they were used
# in the original SpectralInterpolator and SpectralPreprocessor code.
def interpolate_spectra(X_spectral, min_value, max_value, step, extrapolate):
    """Placeholder for spectral interpolation logic."""
    print(f"Warning: Using placeholder 'interpolate_spectra'.")
    num_out_points = int(np.round((max_value - min_value) / step)) + 1
    new_grid = np.linspace(min_value, max_value, num_out_points)
    new_cols = [f"spec_{val:.2f}" for val in new_grid] # Example naming
    return pd.DataFrame(np.random.rand(X_spectral.shape[0], len(new_cols)),
                        columns=new_cols, index=X_spectral.index)

def apply_pretreatments(spectral_data, combo, wavelengths):
    """Placeholder for spectral preprocessing logic."""
    print(f"Warning: Using placeholder 'apply_pretreatments'. Steps: {combo}")
    # Returning a copy to simulate modification without actual processing
    if isinstance(spectral_data, pd.DataFrame):
        return spectral_data.copy()
    else:
        return np.copy(spectral_data)
# --- End Placeholder ---


# ============================================================================
# Class: EnsureX1Presence (Original Code with Docstrings)
# ============================================================================

class EnsureX1Presence(BaseEstimator, TransformerMixin):
    """Ensure a specific feature ('x1') is present in the data.

    This transformer checks if a designated feature, referred to as 'x1',
    exists in the input data `X`. If the feature is missing, it adds a new
    column/feature at the specified position or with the specified name,
    filling it with `fill_value`. It handles both pandas DataFrames and
    NumPy/sparse arrays.

    For DataFrames, using `x1_col` is recommended for clarity. If provided,
    it prioritizes finding/adding the column by name and ensures it's at
    (or moved to) the `x1_index`. If `x1_col` is None or the input is an
    array, it relies on `x1_index` to check/add the feature at that position.

    Parameters
    ----------
    x1_index : int, default=0
        The zero-based index where the 'x1' feature should be located.
        Used primarily for array inputs or as the target position for
        `x1_col` in DataFrames.
    fill_value : scalar, default=np.nan
        The value used to populate the 'x1' feature if it needs to be added.
    x1_col : str or None, default=None
        The expected name of the 'x1' feature column if `X` is a pandas
        DataFrame. If provided:
        - If the column is missing, it's inserted at `x1_index` with this name.
        - If the column exists but is not at `x1_index`, it's moved.
        If None, the transformer operates based on `x1_index` only.

    Attributes
    ----------
    # This transformer does not learn any attributes from the data.
    """
    def __init__(self, x1_index=0, fill_value=np.nan, x1_col=None):
        # Original __init__ code
        self.x1_index = x1_index
        self.fill_value = fill_value
        self.x1_col = x1_col

    def fit(self, X, y=None):
        """Fit transformer.

        This transformer does not learn anything from the data,
        so this method just returns self.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input data. Ignored.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Original fit code
        # No learning needed
        return self

    def transform(self, X):
        """Ensure the 'x1' feature is present.

        Adds the 'x1' feature column/feature if it's missing, filled with
        `fill_value`. If `X` is a DataFrame and `x1_col` is specified,
        it ensures the column exists with that name and attempts to place
        it at `x1_index`.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or sparse matrix
            The input data to transform, shape (n_samples, n_features).

        Returns
        -------
        X_transformed : pandas.DataFrame or numpy.ndarray
            The data with the 'x1' feature ensured. Returns a DataFrame if the
            input was a DataFrame, otherwise returns a NumPy array.
            Note: Sparse matrix inputs may be converted to dense NumPy arrays.
        """
        # Original transform code
        is_dataframe = isinstance(X, pd.DataFrame)
        target_col_name = self.x1_col
        target_index = self.x1_index

        if is_dataframe:
            X_trans = X.copy()
            if target_col_name is not None:
                # If the column is missing, add it
                if target_col_name not in X_trans.columns:
                    # Insert column at the desired index if possible, else append and reorder
                    try:
                         X_trans.insert(target_index, target_col_name, self.fill_value)
                    except IndexError: # If index is out of bounds (e.g., > current num cols)
                         X_trans[target_col_name] = self.fill_value # Append first
                         cols = list(X_trans.columns)
                         cols.remove(target_col_name)
                         cols.insert(target_index, target_col_name)
                         X_trans = X_trans[cols]

                # If column exists but is not at target_index, reorder
                elif X_trans.columns.tolist().index(target_col_name) != target_index:
                    cols = list(X_trans.columns)
                    cols.remove(target_col_name)
                    # Ensure index is valid before inserting
                    insert_pos = min(target_index, len(cols))
                    cols.insert(insert_pos, target_col_name)
                    X_trans = X_trans[cols]

            else: # No column name provided, use index logic (less common for DataFrames)
                 # Check if a column needs to be added based on index
                 if X_trans.shape[1] <= target_index:
                     # Create a default name if adding based on index
                     default_name = f"feature_{target_index}"
                     # Ensure name doesn't clash if user manually named cols 'feature_x'
                     k = 0
                     while default_name in X_trans.columns:
                         k += 1
                         default_name = f"feature_{target_index}_{k}"
                     # Insert requires valid index position, ensure it exists
                     insert_pos = min(target_index, X_trans.shape[1])
                     X_trans.insert(insert_pos, default_name, self.fill_value)
                     # If inserted at end because target_index was > shape[1],
                     # reorder to try and get it to target_index
                     if insert_pos == X_trans.shape[1] - 1 and target_index < X_trans.shape[1] - 1:
                         cols = list(X_trans.columns)
                         cols.remove(default_name)
                         cols.insert(target_index, default_name)
                         X_trans = X_trans[cols]

                 # Note: Reordering existing columns based *only* on index without name is ambiguous
                 # and generally not recommended for DataFrames. We assume if x1_col is None,
                 # the user relies on the existing column order or adds if index is out of bounds.

            return X_trans

        # Handle non-DataFrame inputs (NumPy arrays, sparse matrices)
        if issparse(X):
            # Convert sparse to dense - potential memory issue for large sparse data
            X_dense = X.toarray()
            X_dense = self._ensure_index_present_dense(X_dense, target_index)
            # Return dense array, or convert back to sparse if needed (adds complexity)
            return X_dense
        else:
            # Assume NumPy array or list-of-lists
            X_array = np.array(X)
            # Handle 1D input by reshaping
            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)
            elif X_array.ndim == 0:
                raise ValueError("Input X cannot be a scalar.")

            X_array = self._ensure_index_present_dense(X_array, target_index)
            return X_array

    def _ensure_index_present_dense(self, X_dense, target_index):
        """Helper for dense arrays. Ensures target_index is valid.

        Parameters
        ----------
        X_dense : ndarray, shape (n_samples, n_features)
            The dense input array (must be 2D).
        target_index : int
            The index that needs to be present.

        Returns
        -------
        ndarray
            Array potentially with added columns.
        """
        # Original _ensure_index_present_dense code
        if X_dense.ndim != 2:
            # This case was handled in transform, but good practice to check here too
            raise ValueError("_ensure_index_present_dense expects a 2D array.")
        n_samples, n_features = X_dense.shape
        if n_features <= target_index:
            # Add missing columns to reach the target_index
            n_missing = target_index + 1 - n_features
            # Determine compatible dtype for fill_value
            fill_dtype = X_dense.dtype
            try:
                 _ = np.array([self.fill_value], dtype=fill_dtype)
            except (ValueError, TypeError):
                 fill_dtype = np.result_type(X_dense.dtype, type(self.fill_value))
                 X_dense = X_dense.astype(fill_dtype, copy=False) # Ensure compatibility

            missing_cols = np.full((n_samples, n_missing), self.fill_value, dtype=fill_dtype)
            X_dense = np.hstack([X_dense, missing_cols])
        # Note: Reordering existing columns in NumPy based on index is implicit;
        # this function primarily ensures the column *exists* at or before the index.
        return X_dense


# ============================================================================
# Class: SpectralInterpolator (Original Code with Docstrings)
# ============================================================================

class SpectralInterpolator(BaseEstimator, TransformerMixin):
    """Interpolates spectral data columns to a common grid.

    Selects specified spectral columns from the input DataFrame, interpolates
    them onto a new grid defined by `min_value`, `max_value`, and `step`,
    using an external `interpolate_spectra` function. The original spectral
    columns are dropped, and the new interpolated columns are appended.

    Parameters
    ----------
    spectral_columns : list of str
        Names of the columns containing the spectral data to be interpolated.
    min_value : float
        Minimum value (e.g., wavelength) for the target interpolation grid.
        Used as a fixed parameter.
    max_value : float
        Maximum value (e.g., wavelength) for the target interpolation grid.
        Used as a fixed parameter.
    step : float, default=1
        The step size for the target interpolation grid.
    extrapolate : bool, default=True
        Whether the underlying `interpolate_spectra` function should be
        allowed to extrapolate beyond the original data's range.

    Attributes
    ----------
    # This transformer does not learn any attributes from the data (fixed params).
    """
    def __init__(self, spectral_columns, min_value, max_value, step=1, extrapolate=True):
        # Original __init__ code
        self.spectral_columns = spectral_columns
        self.min_value = min_value  # Fixed parameter (no trailing _)
        self.max_value = max_value  # Fixed parameter
        self.step = step
        self.extrapolate = extrapolate

    def fit(self, X, y=None):
        """Fit transformer.

        No learning is needed as parameters are fixed during initialization.
        Returns self.

        Parameters
        ----------
        X : Ignored.
        y : Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Original fit code
        # No need to compute anything; return self
        # Could add parameter validation here if desired
        if not isinstance(self.spectral_columns, list) or not self.spectral_columns:
            raise ValueError("spectral_columns must be a non-empty list of strings.")
        if not isinstance(self.min_value, (int, float)):
             raise TypeError("min_value must be numeric.")
        # Add similar checks for max_value, step, extrapolate
        return self

    def transform(self, X):
        """Interpolate spectral columns.

        Selects `spectral_columns`, interpolates them using `interpolate_spectra`
        with the fixed parameters `min_value`, `max_value`, `step`, `extrapolate`.
        Drops the original `spectral_columns`, and appends the new interpolated columns.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Input data containing the `spectral_columns`. Must be a DataFrame.

        Returns
        -------
        X_transformed : pandas.DataFrame
            Data with original spectral columns replaced by interpolated ones.

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If `spectral_columns` are missing in `X`.
        RuntimeError
            If the external `interpolate_spectra` function fails.
        """
        # Original transform code
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SpectralInterpolator requires a pandas DataFrame input.")

        # Check if all spectral columns are present
        missing_cols = [col for col in self.spectral_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame missing spectral columns: {missing_cols}")

        X_spectral = X[self.spectral_columns]
        # Use the fixed parameters (self.min_value, NOT self.min_value_)
        try:
            X_spectral_interpolated = interpolate_spectra(
                X_spectral,
                min_value=self.min_value,
                max_value=self.max_value,
                step=self.step,
                extrapolate=self.extrapolate
            )
        except Exception as e:
            raise RuntimeError(f"Error calling external 'interpolate_spectra': {e}") from e

        if not isinstance(X_spectral_interpolated, pd.DataFrame):
             raise TypeError("External 'interpolate_spectra' must return a pandas DataFrame.")
        # Optional: Validate shape/columns of interpolated data if possible/needed

        # Drop original spectral columns and concatenate interpolated data
        X_dropped = X.drop(columns=self.spectral_columns)

        # Ensure indices align for concatenation
        if not X_dropped.index.equals(X_spectral_interpolated.index):
             print("Warning: Index mismatch between dropped data and interpolated data. Attempting reindex.")
             # Decide how to handle: reindex interpolated? error? For now, try reindex.
             try:
                  X_spectral_interpolated = X_spectral_interpolated.reindex(X_dropped.index)
             except Exception as e:
                  raise RuntimeError(f"Failed to align indices for concatenation: {e}") from e

        X_transformed = pd.concat([X_dropped, X_spectral_interpolated], axis=1)
        return X_transformed


# ============================================================================
# Class: SpectralPreprocessor (Original Code with Docstrings)
# ============================================================================

class SpectralPreprocessor(BaseEstimator, TransformerMixin):
    """Applies scaling to specific columns and spectral pretreatments.

    This transformer performs two main actions:

    1. Scales specified non-spectral numeric columns (`specific_columns`)
       using `sklearn.preprocessing.StandardScaler`. The original columns
       are dropped and replaced by new columns with a `_scaled` suffix. The
       scaler is fitted during the `fit` method.
    2. Applies a sequence of spectral preprocessing steps (`preprocessing_steps`)
       to the designated `spectral_columns` using an external
       `apply_pretreatments` function. These columns are modified *in place*
       during the `transform` method.

    Parameters
    ----------
    spectral_columns : list of str
        Names of the columns containing spectral data to which `preprocessing_steps`
        will be applied.
    specific_columns : list of str or None, default=None
        Names of non-spectral numeric columns to be scaled using StandardScaler.
        If None or empty, no scaling is performed. If provided, must be a list.
    preprocessing_steps : list or object
        Configuration defining the sequence of preprocessing steps (e.g.,
        ['snv', 'savgol']) passed to the `apply_pretreatments` function.

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.StandardScaler or None
        The fitted scaler instance used for `specific_columns`. Set during `fit`
        only if `specific_columns` is not empty.
    scaled_specific_cols_ : list of str
        The names generated for the scaled versions of `specific_columns`.
        Determined during `fit`.
    # Note: The original code did not explicitly store ``n_features_in_`` or
    # ``feature_names_in_``, relying on checks within transform.
    """
    def __init__(self, spectral_columns, specific_columns, preprocessing_steps):
        # Original __init__ code
        self.spectral_columns = spectral_columns
        self.specific_columns = specific_columns if specific_columns else [] # Ensure list
        self.preprocessing_steps = preprocessing_steps
        # Initialize scaler, will be fitted in fit()
        self.scaler_ = None
        # Determine scaled names now, based on init parameter
        self.scaled_specific_cols_ = [f"{col}_scaled" for col in self.specific_columns] if self.specific_columns else []

    def fit(self, X, y=None):
        """Fit the preprocessor.

        Fits the StandardScaler on `specific_columns` if they are provided.
        Validates the presence of required columns in the input data `X`.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The training input samples. Must be a DataFrame.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If `specific_columns` are provided but missing in `X`.
        """
        # Original fit code
        if not isinstance(X, pd.DataFrame):
             raise TypeError("SpectralPreprocessor requires a pandas DataFrame input for fit.")

        # Ensure specific columns are present if provided
        if self.specific_columns:
            if not all(col in X.columns for col in self.specific_columns):
                 missing = [col for col in self.specific_columns if col not in X.columns]
                 raise ValueError(f"Missing specific columns in fit data: {missing}")
            self.scaler_ = StandardScaler()
            try:
                 self.scaler_.fit(X[self.specific_columns])
            except Exception as e:
                 raise ValueError(f"Error fitting StandardScaler on specific columns: {e}. Ensure they are numeric.") from e
        # Note: Scaled column names (self.scaled_specific_cols_) were determined in __init__

        # Validate spectral columns presence (optional check in fit)
        if self.spectral_columns:
            if not all(col in X.columns for col in self.spectral_columns):
                 missing_spectral = [col for col in self.spectral_columns if col not in X.columns]
                 # Changed from raising error to warning, as transform checks anyway
                 print(f"Warning: Fit data missing some spectral columns: {missing_spectral}. Transform will fail if still missing.")

        return self

    def transform(self, X):
        """Apply scaling and spectral preprocessing.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples to transform. Must be a DataFrame.

        Returns
        -------
        X_transformed : pandas.DataFrame
            The transformed data with scaled columns (if any) replacing originals,
            and spectral columns modified in place. Column order may change.

        Raises
        ------
        NotFittedError
            If `fit` has not been called (specifically, if `scaler_` is needed
            but has not been fitted).
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If required columns (`specific_columns` during scaling, or
            `spectral_columns` during pretreatment) are missing in `X`, or if
            spectral column names cannot be converted to float for wavelengths.
        RuntimeError
            If the external `apply_pretreatments` function fails.
        """
        # Original transform code
        if not isinstance(X, pd.DataFrame):
             raise TypeError("SpectralPreprocessor requires a pandas DataFrame input for transform.")

        X_out = X.copy() # Work on a copy

        # 1. Scale specific columns and add/replace them with _scaled names
        if self.specific_columns:
            # check_is_fitted is needed here to ensure scaler_ is fitted
            try:
                 check_is_fitted(self, 'scaler_')
            except NotFittedError as e:
                 # Re-raise with more context if scaler was expected
                 if self.specific_columns:
                      raise NotFittedError("Scaler has not been fitted (or specific_columns changed). Call fit() first.") from e
                 else:
                      pass # No scaler needed if specific_columns is empty

            # Check columns are present in the transform data
            if not all(col in X_out.columns for col in self.specific_columns):
                 missing = [col for col in self.specific_columns if col not in X_out.columns]
                 raise ValueError(f"Missing specific columns in transform data: {missing}")

            try:
                scaled_data = self.scaler_.transform(X_out[self.specific_columns])
            except Exception as e:
                 raise ValueError(f"Error transforming specific columns with StandardScaler: {e}") from e

            # Create temporary DF with scaled data and _scaled names
            # Scaled names were determined in __init__ / fit
            X_specific_scaled_df = pd.DataFrame(scaled_data,
                                                columns=self.scaled_specific_cols_,
                                                index=X_out.index)
            # Drop original specific columns
            X_out = X_out.drop(columns=self.specific_columns)
            # Add the new scaled columns
            X_out = pd.concat([X_out, X_specific_scaled_df], axis=1)


        # 2. Apply pretreatments to spectral columns IN PLACE
        # Check presence of spectral columns in transform data
        if self.spectral_columns: # Only proceed if spectral columns were specified
            if not all(col in X_out.columns for col in self.spectral_columns):
                 missing_spectral = [col for col in self.spectral_columns if col not in X_out.columns]
                 raise ValueError(f"Missing spectral columns in transform data: {missing_spectral}")

            # Extract wavelengths assuming column names are numeric strings
            try:
                 wavelengths = [float(col) for col in self.spectral_columns]
            except ValueError as e:
                 raise ValueError("Spectral column names must be convertible to float to be used as wavelengths "
                                  "for 'apply_pretreatments'.") from e


            # apply_pretreatments returns a numpy array or DataFrame
            try:
                spectral_data_pretreated = apply_pretreatments(
                    X_out[self.spectral_columns],
                    combo=self.preprocessing_steps,
                    wavelengths=wavelengths
                )
            except Exception as e:
                raise RuntimeError(f"Error calling external 'apply_pretreatments': {e}") from e

            # Update the spectral columns in X_out directly
            # Check shape consistency before assignment
            if not hasattr(spectral_data_pretreated, 'shape') or \
               spectral_data_pretreated.shape != X_out[self.spectral_columns].shape:
                raise ValueError("Output shape from 'apply_pretreatments' does not match input spectral data shape.")

            # Assign back, ensuring indices/columns align if it's a DataFrame
            if isinstance(spectral_data_pretreated, pd.DataFrame):
                 if not spectral_data_pretreated.index.equals(X_out.index):
                      print("Warning: Index mismatch from apply_pretreatments output. Realigning.")
                      spectral_data_pretreated = spectral_data_pretreated.reindex(X_out.index)
                 # Use original spectral columns names for assignment target
                 X_out[self.spectral_columns] = spectral_data_pretreated.values
            else: # Assume numpy array
                 X_out[self.spectral_columns] = spectral_data_pretreated


        # 3. Return the modified DataFrame. FeatureReorder will handle final selection/ordering.
        # The original code returns X_out without enforcing a specific order here.
        return X_out


# ============================================================================
# Class: FeatureReorder (Original Code with Docstrings)
# ============================================================================

class FeatureReorder(BaseEstimator, TransformerMixin):
    """Reorders DataFrame columns to a desired order.

    Selects and reorders columns of an input DataFrame according to the
    `desired_order` list. Columns present in the input DataFrame but not
    in `desired_order` are dropped.

    Parameters
    ----------
    desired_order : list of str
        A list containing the column names in the desired output order.
        All names in this list must exist in the input DataFrame during `transform`.

    Attributes
    ----------
    # This transformer does not learn any attributes from the data.
    """
    def __init__(self, desired_order):
        # Original __init__ code
        self.desired_order = desired_order

    def fit(self, X, y=None):
        """Fit transformer.

        No fitting is needed; just returns self.

        Parameters
        ----------
        X : Ignored.
        y : Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Original fit code
        # No fitting is needed; just return self.
        # Could add validation for desired_order here
        if not isinstance(self.desired_order, list) or not self.desired_order:
            raise ValueError("desired_order must be a non-empty list of strings.")
        if not all(isinstance(item, str) for item in self.desired_order):
             raise TypeError("All items in desired_order must be strings.")
        return self

    def transform(self, X, y=None):
        """Reorder DataFrame columns.

        Selects columns specified in `desired_order` from `X` and returns
        them in that specific order.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input DataFrame whose columns need reordering. `n_features`
            must be >= number of columns in `desired_order`.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        X_reordered : pandas.DataFrame
            DataFrame containing only the columns from `desired_order`, in that order.

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If any column listed in `desired_order` is not found in `X`.
        """
        # Original transform code
        if not isinstance(X, pd.DataFrame):
             raise TypeError("FeatureReorder requires a pandas DataFrame input.")

        X = X.copy() # Work on copy as per original code intent? Or transform inplace? Be explicit.
        # Let's assume working on copy is safer.
        X_out = X

        # Check that all desired columns exist.
        missing = [col for col in self.desired_order if col not in X_out.columns]
        if missing:
            raise ValueError(f"Missing columns required by desired_order: {missing}")

        # Reorder columns according to the desired order.
        # This implicitly drops columns not in desired_order.
        return X_out[self.desired_order]


# ============================================================================
# Class: CustomPipelineWrapper (Original Code with Docstrings)
# ============================================================================

class CustomPipelineWrapper(BaseEstimator):
    """Wraps a pipeline for pH prediction with 'x1' solving capabilities.

    This estimator wraps a scikit-learn pipeline (expected to perform
    preprocessing and prediction). It adds specialized functionality for
    scenarios involving pH prediction where a key input feature (designated 'x1',
    e.g., titration rate) might be known or might need to be determined
    algorithmically to achieve a specific target pH output.

    Key features include standard prediction, automatic fallback to solving if 'x1'
    is missing, direct solver access, response curve generation, and flexible
    output formatting for delta pH or final pH models.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline object
        The underlying pipeline responsible for data transformation and making
        raw predictions. Must be a fitted or fit-able pipeline.
    initialpH_col : str
        The name of the column in the input DataFrame containing the initial pH values.
        Required and used only if `predicts_delta_ph` is True.
    x1_index : str
        The name of the column in the input DataFrame representing the 'x1' feature
        (e.g., 'Rate (Titration)'). This is the feature that can be solved for.
        *(Note: Original parameter name was `x1_index`, but used like a column name)*.
    x1_bounds : tuple (float, float), default=(0, 100)
        The default lower and upper bounds for the 'x1' feature when using
        solver methods (`solve_for_x1`, `predict_with_sampling`).
    response : {'delta', 'ph', 'both'}, default='both'
        Determines the output format of `predict` and results from
        `predict_with_sampling` when `predicts_delta_ph` is True:
        - 'delta': Output the predicted change in pH.
        - 'ph': Output the calculated final pH.
        - 'both': Output a dictionary `{'delta': ..., 'pH': ...}`.
        If `predicts_delta_ph` is False, this parameter is ignored (output is final pH).
    predicts_delta_ph : bool, default=True
        Indicates whether the underlying `pipeline` predicts the change in pH
        (True) or the final pH directly (False).
    target_ph : float, default=7.3
        The default target pH value used by the solver methods if not overridden.
    fill_value : scalar, default=np.nan
        The value used internally or by pipeline steps when 'x1' is missing or
        needs to be determined by the solver.

    Attributes
    ----------
    feature_names_in_ : list of str or None
        Names of features seen during `fit`, captured from the input DataFrame `X`.
        Set during `fit`, None otherwise.
    # The wrapped `pipeline` object is also stored (see Parameters).

    See Also
    --------
    sklearn.pipeline.Pipeline : The type of object expected for the `pipeline` parameter.
    scipy.optimize.brentq : The root-finding algorithm used by `solve_for_x1`.

    Notes
    -----
    - Requires input data `X` to be a pandas DataFrame for most methods.
    - The `x1_index` parameter acts as the column name for the 'x1' feature.
    - Assumes the underlying `pipeline` is correctly configured to handle the
      data transformations and has a `predict` method.
    - Solver methods rely on the relationship between 'x1' and predicted pH.
    """
    def __init__(self, pipeline, initialpH_col, x1_index, x1_bounds=(0, 100),
                 response="both", predicts_delta_ph=True, target_ph=7.3,
                 fill_value=np.nan):
        # Original __init__ code
        self.pipeline = pipeline
        self.initialpH_col = initialpH_col
        # NOTE: Parameter is named x1_index, but used as a column name throughout original code.
        # Docstring reflects this usage.
        self.x1_index = x1_index
        self.x1_bounds = x1_bounds
        self.response = response.lower() # Original code converts to lower here
        self.predicts_delta_ph = predicts_delta_ph
        self.target_ph = target_ph
        self.fill_value = fill_value
        self.feature_names_in_ = None  # will be set during fit

    def _filter_solver_kwargs(self, kwargs):
        """Internal: Removes keys specific to solver not meant for pipeline.predict."""
        # Original _filter_solver_kwargs code
        keys_to_remove = {"step", "max_rate", "grid_points", "use_grid_search"}
        return {k: v for k, v in kwargs.items() if k not in keys_to_remove}

    @staticmethod
    def _ensure_x1_in_sample(sample, x1_col, x1_index, fill_value=np.nan):
        """Internal: Ensures x1 col exists in a Series/DataFrame row.

        Parameters
        ----------
        sample : pd.Series or pd.DataFrame (1 row)
        x1_col : str
            Name of the x1 column.
        x1_index : int
            Target index hint for insertion (used by original code).
        fill_value : scalar
            Value to fill with.

        Returns
        -------
        pd.Series
            Sample with x1 column ensured.
        """
        # Original _ensure_x1_in_sample code
        if not isinstance(sample, pd.DataFrame):
            # Original code converted non-DataFrames here
            sample = pd.DataFrame(sample).T
        if x1_col not in sample.columns:
            try:
                # Use index hint for insertion
                sample.insert(x1_index, x1_col, fill_value)
            except Exception: # Original code had broad except
                 # Fallback: append and reorder
                 sample[x1_col] = fill_value
                 cols = list(sample.columns)
                 if x1_col in cols: # Should always be true here
                    cols.remove(x1_col)
                 # Ensure insert index is valid
                 insert_pos = min(x1_index, len(cols))
                 cols.insert(insert_pos, x1_col)
                 sample = sample[cols]
        # Return Series as expected by callers in original code
        return sample.iloc[0]

    def _get_x1_value(self, sample):
        """Internal: Retrieves x1 value from a Series/DataFrame row."""
        # Original _get_x1_value code
        # Uses x1_index as the column name
        x1_col = self.x1_index
        # Find index hint (though not strictly needed for get)
        x1_col_idx_hint = self.x1_bounds[0] # Original used lower bound as index hint? Odd. Let's use 0.
        if self.feature_names_in_ and x1_col in self.feature_names_in_:
             try: x1_col_idx_hint = self.feature_names_in_.index(x1_col)
             except ValueError: pass

        if x1_col not in sample.index:
             # Ensure sample has the column if missing (original code logic)
             # Pass the column name, index hint, and fill value
             sample = self._ensure_x1_in_sample(sample, x1_col, x1_col_idx_hint, fill_value=self.fill_value)

        # Now access the value (sample is guaranteed to be a Series here)
        if isinstance(sample, pd.Series):
            return sample[x1_col]
        else:
            # This path shouldn't be reached due to _ensure_x1_in_sample returning Series
            raise TypeError("Sample could not be converted to Series in _get_x1_value.")


    def _set_x1_value(self, sample, value):
        """Internal: Sets the x1 value in a Series."""
        # Original _set_x1_value code
        # Expects sample to be a Series and uses x1_index as column name
        if not isinstance(sample, pd.Series):
            raise TypeError("Sample must be a pandas Series for _set_x1_value.")
        if self.x1_index not in sample.index:
             # This indicates a problem, column should exist before setting
             raise ValueError(f"x1 column '{self.x1_index}' not found in sample Series for setting value.")
        sample[self.x1_index] = value
        return sample

    def fit(self, X, y):
        """Fit the underlying pipeline.

        Stores the feature names from the input DataFrame `X` and then calls
        the `fit` method of the wrapped `pipeline`.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Training data. Must contain `initialpH_col` if `predicts_delta_ph`
            is True. Must be a DataFrame.
        y : array-like, shape (n_samples,)
            Target values corresponding to `X`.

        Returns
        -------
        self : object
            The fitted wrapper instance.

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If `predicts_delta_ph` is True and `initialpH_col` is missing.
        RuntimeError
            If the underlying pipeline fitting fails.
        """
        # Original fit code
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} requires a pandas DataFrame for fit.")
        if self.predicts_delta_ph and self.initialpH_col not in X.columns:
            raise ValueError(f"Initial pH column '{self.initialpH_col}' is required for delta pH prediction.")
        self.feature_names_in_ = X.columns.tolist()

        # Check if pipeline is already fitted (original code logic)
        try:
            # Relies on check_is_fitted being available in the environment
            # Needs: from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.pipeline)
            print("Pipeline appears to be already fitted.") # Or skip fitting? Original fits anyway.
        except NotFittedError:
            # Only fit if NotFittedError is raised
             try:
                 self.pipeline.fit(X, y)
             except Exception as e:
                 raise RuntimeError(f"Error during underlying pipeline fitting: {e}") from e
        except Exception as e:
             # Catch other errors from check_is_fitted (e.g., TypeError if not an estimator)
             print(f"Warning: Could not check if pipeline was fitted ({e}). Attempting to fit anyway.")
             try:
                 self.pipeline.fit(X, y)
             except Exception as fit_e:
                 raise RuntimeError(f"Error during underlying pipeline fitting: {fit_e}") from fit_e

        # Original code fits even if check_is_fitted passes without NotFittedError
        # Replicating that behavior, although potentially inefficient:
        # try:
        #     self.pipeline.fit(X, y)
        # except Exception as e:
        #      raise RuntimeError(f"Error during underlying pipeline fitting: {e}") from e

        return self

    def predict_raw(self, X, **kwargs):
        """Call the predict method of the underlying pipeline directly.

        Filters out solver-specific keyword arguments before calling `predict`.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Data for which to make predictions. Original code did not explicitly
            require DataFrame here, but subsequent steps often assume it.
        **kwargs : dict
            Additional keyword arguments for the pipeline's predict method.
            Keys 'step', 'max_rate', 'grid_points', 'use_grid_search' are removed.

        Returns
        -------
        predictions : array-like
            Raw predictions from the pipeline.

        Raises
        ------
        RuntimeError
            If the underlying pipeline prediction fails.
        """
        # Original predict_raw code
        filtered_kwargs = self._filter_solver_kwargs(kwargs)
        try:
            # Assume pipeline can handle various input types if X isn't strictly DataFrame
            return self.pipeline.predict(X, **filtered_kwargs)
        except Exception as e:
            raise RuntimeError(f"Error during underlying pipeline predict_raw call: {e}") from e

    def _format_response(self, delta, initialpH_value):
        """Internal: Formats the prediction output based on response type."""
        # Original _format_response code
        if not self.predicts_delta_ph:
            return delta # Assumes delta is actually final pH if not predicting delta
        # Handle delta pH prediction cases
        if self.response == "delta":
            return delta
        elif self.response == "ph":
            if pd.isna(initialpH_value): raise ValueError("Initial pH is NaN.")
            return initialpH_value + delta
        elif self.response == "both":
            if pd.isna(initialpH_value): raise ValueError("Initial pH is NaN.")
            # Original code handled dict/list/array input for delta differently?
            # Let's assume delta is scalar here based on typical model output.
            ph_val = initialpH_value + delta
            return {"delta": delta, "pH": ph_val}
            # Original complex handling:
            # if isinstance(delta, (dict, list, np.ndarray)):
            #     ph_val = initialpH_value + (delta.get("delta", delta) if isinstance(delta, dict) else delta)
            #     return {"delta": delta, "pH": ph_val} # This seems odd - delta could be dict?
            # else:
            #     return {"delta": delta, "pH": initialpH_value + delta}
        else:
             raise ValueError(f"Invalid response setting: {self.response}")


    def predict(self, X, num_points=100, target=None, solver_method='vectorized', **kwargs):
        """Predict target values for X.

        Uses the pipeline's raw prediction if the 'x1' column (`self.x1_index`)
        exists and is not all NaN. Otherwise, falls back to `predict_with_sampling`.
        Formats output based on `self.response` and `self.predicts_delta_ph`.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Input data for prediction. Must be a DataFrame. Must contain
            `initialpH_col` if `predicts_delta_ph` is True.
        num_points : int, default=100
            Number of points for response curve if `predict_with_sampling` is triggered.
        target : float or None, default=None
            Target pH for solver if `predict_with_sampling` is triggered.
            Defaults to `self.target_ph`.
        solver_method : {'vectorized', 'brentq'}, default='vectorized'
            Solver method to use if `predict_with_sampling` is triggered.
            *(Note: Original code passed 'vectorized' only to one solver type)*.
        **kwargs : dict
            Additional arguments. Solver-specific ones are used by solvers,
            others are passed to `predict_raw`.

        Returns
        -------
        predictions : pandas.Series or list
            - If predicting directly: A pandas Series (dtype float or object) or a list
              of dictionaries (if `response='both'`).
            - If solving/sampling: Returns the output of `predict_with_sampling`
              (a list of dictionaries).

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If required columns are missing or contain NaNs when needed.
        RuntimeError
            If prediction or formatting fails.
        """
        # Original predict code
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} requires a pandas DataFrame for predict.")

        # Use x1_index as the column name
        x1_col = self.x1_index

        # Check presence of required columns
        if self.predicts_delta_ph:
            if self.initialpH_col not in X.columns:
                raise ValueError(f"Initial pH column '{self.initialpH_col}' not found in prediction data.")
            if X[self.initialpH_col].isna().any():
                 raise ValueError(f"Initial pH column '{self.initialpH_col}' contains NaN values.")


        # Decide whether to run solver
        run_solver = False
        if x1_col not in X.columns:
            print(f"x1 column '{x1_col}' missing. Triggering predict_with_sampling...")
            run_solver = True
        elif X[x1_col].isna().all():
            print(f"x1 column '{x1_col}' contains only NaN values. Triggering predict_with_sampling...")
            run_solver = True

        if run_solver:
            # Fallback to predict_with_sampling
            # Pass through kwargs intended for the solver / predict calls within it
            _target = target if target is not None else self.target_ph
            return self.predict_with_sampling(X, num_points=num_points, target=_target, solver_method=solver_method, **kwargs)

        # Proceed with direct prediction
        filtered_kwargs = self._filter_solver_kwargs(kwargs)
        try:
            raw_predictions = self.predict_raw(X, **filtered_kwargs)
        except Exception as e:
             raise RuntimeError(f"Failed calling predict_raw: {e}") from e

        # Format results
        try:
            if self.predicts_delta_ph:
                initialpH_values = X[self.initialpH_col]
                formatted_results = [self._format_response(raw, ip)
                                     for raw, ip in zip(raw_predictions, initialpH_values)]
                if self.response in ['delta', 'ph']:
                    # Infer dtype if possible, default to object if mixed/error
                    dtype = float if all(isinstance(r, (int, float, np.number)) for r in formatted_results) else object
                    return pd.Series(formatted_results, index=X.index, name=self.response, dtype=dtype)
                else: # response == 'both'
                    # Returns list of dicts
                    return formatted_results
            else:
                # Pipeline predicts final pH directly
                return pd.Series(raw_predictions, index=X.index, name="pH", dtype=float)
        except Exception as e:
             raise RuntimeError(f"Failed to format predictions: {e}") from e

    def solve_for_x1_vectorized_incremental(self, sample, target, step=0.1, max_rate=None, **kwargs):
        """Solve for 'x1' using vectorized grid search (for one sample).

        Finds the lowest 'x1' value (>= 0) for a single input `sample` that
        results in a predicted final pH >= `target`. Uses a grid search.

        *Note: Original code used `max_rate=100` as default in one place.*

        Parameters
        ----------
        sample : pandas.Series or single-row pandas.DataFrame
            The input sample. Must contain required features (incl. non-NaN
            `initialpH_col` if `predicts_delta_ph` is True) except 'x1'.
        target : float
            The target pH value to achieve or exceed.
        step : float, default=0.1
            The step size for the 'x1' grid search.
        max_rate : float or None, default=None
            Maximum 'x1' value for the grid search. If None, defaults to
            `self.x1_bounds[1]`. *(Original default was 100 here)*.
        **kwargs : dict
            Additional arguments passed to `self.predict` (used recursively in original)
            or `self.predict_raw`. Using `predict_raw` avoids infinite recursion.

        Returns
        -------
        x1_solution : float or np.nan
            The lowest 'x1' value found that meets the target, or `np.nan`.

        Raises
        ------
        TypeError
            If sample type is invalid.
        ValueError
            If required columns are missing or initial pH is NaN (when needed).
        RuntimeError
             If prediction within the solver fails.
        """
        # Original solve_for_x1_vectorized_incremental code
        # Needs access to x1_index (column name) and initialpH_col
        x1_col = self.x1_index

        # Handle sample input type (ensure Series)
        if isinstance(sample, pd.DataFrame):
             if sample.shape[0] != 1: raise ValueError("Input sample DataFrame must have only one row.")
             sample = sample.iloc[0].copy()
        elif isinstance(sample, pd.Series):
             sample = sample.copy()
        else:
             raise TypeError("Input sample must be a pandas Series or single-row DataFrame.")

        # Ensure sample has x1 column conceptually (original code did this here too)
        x1_col_idx_hint = 0 # Default index hint
        if self.feature_names_in_ and x1_col in self.feature_names_in_:
             try: x1_col_idx_hint = self.feature_names_in_.index(x1_col)
             except ValueError: pass
        sample = self._ensure_x1_in_sample(sample, x1_col, x1_col_idx_hint, fill_value=self.fill_value)
        # Explicitly set to fill_value for solver logic
        sample[x1_col] = self.fill_value


        # Check initial pH
        if self.predicts_delta_ph:
            if self.initialpH_col not in sample.index:
                raise ValueError(f"Initial pH column '{self.initialpH_col}' not found in sample.")
            if pd.isna(sample[self.initialpH_col]):
                 raise ValueError(f"Initial pH column '{self.initialpH_col}' is NaN in sample.")
            initialpH_value = sample[self.initialpH_col]
        else:
            initialpH_value = 0 # Not used, but set for consistency


        _max_rate = max_rate if max_rate is not None else self.x1_bounds[1]
        # Original code used default max_rate=100 if passed as None here
        # Let's stick to x1_bounds[1] for consistency with brentq solver
        # _max_rate = max_rate if max_rate is not None else 100 # Original behavior?

        if step <= 0: raise ValueError("Step must be positive.")

        candidate_rates = np.arange(0, _max_rate + step, step)
        if len(candidate_rates) == 0:
            print(f"Warning: No candidate rates generated (max_rate={_max_rate}, step={step}).")
            return np.nan

        # Create DataFrame for vectorized prediction
        # Need all columns from the original sample Series
        candidate_df = pd.DataFrame([sample] * len(candidate_rates))
        candidate_df[x1_col] = candidate_rates


        # Filter kwargs before prediction
        filtered_kwargs = self._filter_solver_kwargs(kwargs)

        try:
            # *** CRITICAL CHANGE from original: Use predict_raw to avoid recursion ***
            # Original called self.predict, leading to potential infinite loop if x1 was NaN.
            raw_predictions = self.predict_raw(candidate_df, **filtered_kwargs)
            # raw_predictions = self.predict(candidate_df, **filtered_kwargs) # Original Recursive Call

            # Calculate final pH based on raw predictions
            if self.predicts_delta_ph:
                predicted_phs = initialpH_value + raw_predictions
            else: # Assumes raw prediction is final pH
                predicted_phs = raw_predictions

            # Handle potential NaNs in predictions
            valid_mask = ~np.isnan(predicted_phs)
            if not np.any(valid_mask):
                print("Warning: All predictions in vectorized solver were NaN.")
                return np.nan

            # Find the first rate meeting target among valid predictions
            valid_indices = np.where(valid_mask)[0]
            target_met_indices = np.where(predicted_phs[valid_mask] >= target)[0]

            if len(target_met_indices) > 0:
                # Get the index relative to the valid predictions
                first_valid_idx = target_met_indices[0]
                # Map back to the original candidate_rates index
                original_idx = valid_indices[first_valid_idx]
                solution_rate = candidate_rates[original_idx]
                return solution_rate
            else:
                # Target not reached among valid predictions
                max_valid_ph = np.max(predicted_phs[valid_mask])
                print(f"Warning: Target pH {target} not reached in range [0, {_max_rate}]. Max valid pH: {max_valid_ph:.3f}")
                return np.nan

        except Exception as e:
            print(f"Error during vectorized prediction/processing in solver: {e}")
            # Wrap error?
            raise RuntimeError(f"Failure in vectorized solver prediction: {e}") from e


    def solve_for_x1(self, sample, target, bounds=None, use_grid_search=True, grid_points=15, **kwargs):
        """Solve for 'x1' using root-finding (for one sample).

        Finds the 'x1' value for a single input `sample` that results in a
        predicted final pH exactly equal to `target`, using `scipy.optimize.brentq`.
        Optionally performs a preliminary grid search.

        Parameters
        ----------
        sample : pandas.Series or single-row pandas.DataFrame
            The input sample. Must contain required features (incl. non-NaN
            `initialpH_col` if `predicts_delta_ph` is True) except 'x1'.
        target : float
            The target pH value to achieve.
        bounds : tuple (float, float) or None, default=None
            Specific bounds (min_x1, max_x1) for the solver search. If None,
            defaults to `self.x1_bounds`.
        use_grid_search : bool, default=True
            If True, performs a coarse grid search within `bounds` to find a
            narrower interval for `brentq`.
        grid_points : int, default=15
            Number of points for the preliminary grid search if `use_grid_search`
            is True.
        **kwargs : dict
            Additional arguments passed to `self.predict_raw` after filtering.

        Returns
        -------
        x1_solution : float or np.nan
            The 'x1' value found, or `np.nan` if solving fails.

        Raises
        ------
        TypeError
            If sample type is invalid.
        ValueError
            If required columns are missing or initial pH is NaN (when needed),
            or grid parameters invalid.
        RuntimeError
             If prediction within the solver objective function fails.
        """
        # Original solve_for_x1 code
        x1_col = self.x1_index # Use parameter as column name

        # Handle sample input type
        if isinstance(sample, pd.DataFrame):
             if sample.shape[0] != 1: raise ValueError("Input sample DataFrame must have only one row.")
             sample = sample.iloc[0].copy()
        elif isinstance(sample, pd.Series):
             sample = sample.copy()
        else:
             raise TypeError("Input sample must be a pandas Series or single-row DataFrame.")

        # Ensure x1 column exists conceptually and reset it
        x1_col_idx_hint = 0
        if self.feature_names_in_ and x1_col in self.feature_names_in_:
             try: x1_col_idx_hint = self.feature_names_in_.index(x1_col)
             except ValueError: pass
        sample_prepared = self._ensure_x1_in_sample(sample, x1_col, x1_col_idx_hint, fill_value=self.fill_value)
        sample_prepared = self._set_x1_value(sample_prepared, self.fill_value)

        # Check initial pH
        if self.predicts_delta_ph:
            if self.initialpH_col not in sample_prepared.index:
                 raise ValueError(f"Initial pH column '{self.initialpH_col}' not found in sample.")
            if pd.isna(sample_prepared[self.initialpH_col]):
                 raise ValueError(f"Initial pH column '{self.initialpH_col}' is NaN in sample.")
            initialpH_value = sample_prepared[self.initialpH_col]
        else:
            initialpH_value = 0


        initial_solver_bounds = bounds if bounds is not None else self.x1_bounds
        if not (isinstance(initial_solver_bounds, tuple) and len(initial_solver_bounds) == 2):
             raise ValueError("Solver bounds must be a tuple of (min, max).")
        if use_grid_search and grid_points <= 2:
             raise ValueError("grid_points must be > 2 for grid search.")


        # Define the objective function for the root finder
        def objective_func(x1_val):
            sample_candidate = sample_prepared.copy()
            sample_candidate = self._set_x1_value(sample_candidate, x1_val)
            sample_candidate_df = sample_candidate.to_frame().T

            filtered_kwargs = self._filter_solver_kwargs(kwargs)
            try:
                 raw_pred = self.predict_raw(sample_candidate_df, **filtered_kwargs)
                 # Ensure scalar output
                 if not np.isscalar(raw_pred) and len(raw_pred) == 1:
                     raw_pred_scalar = np.array(raw_pred).item()
                 elif np.isscalar(raw_pred):
                     raw_pred_scalar = raw_pred
                 else:
                      raise ValueError(f"Expected scalar prediction, got {type(raw_pred)}")
                 if pd.isna(raw_pred_scalar):
                     raise ValueError("predict_raw returned NaN.")

            except Exception as e:
                 # Let brentq handle errors from objective? Or catch here?
                 # Original code didn't explicitly catch here. Re-raising runtime.
                 # print(f"Warning: Objective function failed at x1={x1_val:.4f}: {e}")
                 raise RuntimeError(f"predict_raw failed in objective function for x1={x1_val}: {e}") from e

            # Calculate current pH based on prediction type
            if self.predicts_delta_ph:
                current_value = initialpH_value + raw_pred_scalar
            else:
                current_value = raw_pred_scalar # Assumes raw is final pH

            # Original code accessed ['pH'] from formatted response? Simpler here.
            # formatted_prediction = self._format_response(delta_ph, initialpH_value)
            # current_value = formatted_prediction['pH'] if self.predicts_delta_ph and self.response=='both' else formatted_prediction

            return current_value - target

        # --- Optional Grid Search for Better Bounds ---
        refined_bounds = initial_solver_bounds
        if use_grid_search: # grid_points check done above
            grid_x = np.linspace(initial_solver_bounds[0], initial_solver_bounds[1], grid_points)
            grid_y = []
            valid_grid_x = []
            # Evaluate objective at grid points, handling potential errors
            for x_val in grid_x:
                try:
                    y_val = objective_func(x_val)
                    # Check for non-finite values which brentq can't handle
                    if not np.isfinite(y_val):
                         print(f"Warning: Non-finite objective value ({y_val}) at x1={x_val:.4f}. Skipping grid point.")
                         continue
                    grid_y.append(y_val)
                    valid_grid_x.append(x_val)
                except RuntimeError: # Catch prediction failures from objective
                    print(f"Warning: Objective evaluation failed at grid point x1={x_val:.4f}. Skipping.")
                except Exception as e: # Catch other unexpected errors
                    print(f"Warning: Unexpected error evaluating grid point x1={x_val:.4f}: {e}. Skipping.")


            if len(valid_grid_x) < 2:
                 print("Warning: Grid search failed: Not enough valid points evaluated. Using original bounds.")
                 refined_bounds = initial_solver_bounds
            else:
                 grid_y = np.array(grid_y)
                 valid_grid_x = np.array(valid_grid_x)
                 # Find sign changes in the valid grid results
                 sign_changes = np.where(np.diff(np.sign(grid_y)))[0]
                 if len(sign_changes) > 0:
                     first_change_idx = sign_changes[0]
                     refined_bounds = (valid_grid_x[first_change_idx], valid_grid_x[first_change_idx + 1])
                 else:
                     # No sign change found in grid. Check original bounds.
                     try:
                         f_a_orig = objective_func(initial_solver_bounds[0])
                         f_b_orig = objective_func(initial_solver_bounds[1])
                         if np.sign(f_a_orig) * np.sign(f_b_orig) < 0:
                              print("Warning: No sign change in grid, but original bounds are valid. Using original bounds.")
                              refined_bounds = initial_solver_bounds
                         else:
                              # Check if target is very close to a grid point
                              closest_idx = np.argmin(np.abs(grid_y))
                              if abs(grid_y[closest_idx]) < 1e-6: # Tolerance
                                   print(f"Info: Target likely met at grid point x1={valid_grid_x[closest_idx]:.4f}.")
                                   return valid_grid_x[closest_idx]
                              print("Warning: Grid search found no sign change and original bounds invalid. Target may be out of range or function monotonic.")
                              return np.nan # brentq will fail
                     except RuntimeError: # Objective failed at bounds
                          print(f"Warning: Objective failed at initial bounds ({initial_solver_bounds}) after grid search. Cannot solve.")
                          return np.nan
                     except Exception as e: # Other error at bounds
                          print(f"Warning: Error evaluating objective at bounds ({initial_solver_bounds}): {e}. Cannot solve.")
                          return np.nan


        # --- Brentq Root Finding ---
        try:
            # Check bounds signs before calling brentq
            f_a = objective_func(refined_bounds[0])
            f_b = objective_func(refined_bounds[1])

            # Check for non-finite values at bounds
            if not (np.isfinite(f_a) and np.isfinite(f_b)):
                 print(f"Warning: Non-finite objective value at bounds {refined_bounds}: f(a)={f_a}, f(b)={f_b}. Cannot solve.")
                 return np.nan

            if np.sign(f_a) == np.sign(f_b):
                # Check if one of the bounds *is* the root (within tolerance)
                if abs(f_a) < 1e-9: return refined_bounds[0]
                if abs(f_b) < 1e-9: return refined_bounds[1]
                print(f"Warning: Objective function values at bounds {refined_bounds} have the same sign: f(a)={f_a:.4f}, f(b)={f_b:.4f}. Brentq requires opposite signs.")
                return np.nan

            # Attempt brentq
            x1_solution = brentq(objective_func, refined_bounds[0], refined_bounds[1])
            return x1_solution

        except RuntimeError: # Catch objective failure during brentq internal calls
            print(f"Warning: Objective function failed during brentq execution with bounds {refined_bounds}.")
            return np.nan
        except ValueError as e: # Catch specific brentq errors
            print(f"Warning: Brentq failed with bounds {refined_bounds}: {e}")
            return np.nan
        except Exception as e: # Catch unexpected errors
            print(f"Warning: Unexpected error in Brentq with bounds {refined_bounds}: {e}")
            return np.nan


    def predict_with_sampling(self, X, num_points=100, target=None, solver_method='vectorized', **kwargs):
        """Generate response curves (output vs. 'x1') for samples.

        Processes each row in `X` independently. For each sample, it uses a
        solver (`solver_method`) to find the 'x1' value (`x1_max`) needed to
        reach the `target` pH. It then predicts outputs across a range of 'x1'
        values from 0 to `x1_max`. Includes a 'highlight' point if the original
        sample had an 'x1' value.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Input data samples. Must be a DataFrame. Must contain `initialpH_col`
            (non-NaN) if `predicts_delta_ph` is True.
        num_points : int, default=100
            Number of points for the response curve [0, x1_max]. Must be >= 2.
        target : float or None, default=None
            The target pH used by the solver. Defaults to `self.target_ph`.
        solver_method : {'vectorized', 'brentq'}, default='vectorized'
            Method used to find `x1_max`.
        **kwargs : dict
            Additional arguments passed to the solver function and subsequently
            to `predict_raw` for curve generation.

        Returns
        -------
        results : list of dict
            List of dictionaries, one per input sample. Each contains:
            'x1_max': Solved 'x1' (float or np.nan).
            'x1_candidates': Array of 'x1' values for curve (shape `(num_points,)`).
            'predictions': List of corresponding formatted predictions.
            'highlight': Tuple `(provided_x1, prediction)` or None.

        Raises
        ------
        NotFittedError
            If `fit` has not been called.
        TypeError
            If `X` is not a pandas DataFrame.
        ValueError
            If required columns missing/NaN, or `num_points` invalid.
        RuntimeError
            If solver or prediction calls fail critically.
        """
        # Original predict_with_sampling code
        _target = target if target is not None else self.target_ph
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame for predict_with_sampling.")
        if num_points < 2: # Ensure enough points for a curve
             raise ValueError("num_points must be >= 2.")

        # --- Input Validation and Setup ---
        x1_col = self.x1_index # Use parameter as column name
        X_processed = X.copy() # Work on copy

        # Check initial pH if needed
        if self.predicts_delta_ph:
            if self.initialpH_col not in X_processed.columns:
                raise ValueError(f"Initial pH column '{self.initialpH_col}' is mandatory when predicts_delta_ph=True.")
            if X_processed[self.initialpH_col].isna().any():
                raise ValueError(f"Initial pH column '{self.initialpH_col}' contains NaN values.")

        # Ensure x1 column exists, adding it if necessary
        if x1_col not in X_processed.columns:
            print(f"Warning: x1 column '{x1_col}' missing in predict_with_sampling input. Adding with fill_value={self.fill_value}.")
            X_processed[x1_col] = self.fill_value

        results = []
        # --- Loop through samples ---
        for i, idx in enumerate(X_processed.index): # Iterate with index
            sample = X_processed.loc[idx].copy()
            provided_x1 = sample.get(x1_col, np.nan) # Store original x1

            # Always solve: reset x1 in the sample being processed.
            sample[x1_col] = self.fill_value

            # --- 1. Solve for x1_max ---
            x1_max = np.nan # Default if solver fails
            # Create copy of kwargs for solver to avoid modification side effects
            solver_kwargs = kwargs.copy()
            try:
                if solver_method == 'vectorized':
                    # Pass relevant solver params from kwargs if present
                    # Note: Original didn't explicitly check which kwargs go where
                    x1_max = self.solve_for_x1_vectorized_incremental(sample, _target, **solver_kwargs)
                elif solver_method == 'brentq':
                    x1_max = self.solve_for_x1(sample, _target, **solver_kwargs)
                else:
                    raise ValueError(f"Unknown solver_method: {solver_method}. Choose 'vectorized' or 'brentq'.")
            except Exception as e:
                 print(f"Solver failed for sample index {idx} (target={_target}): {e}")
                 x1_max = np.nan # Ensure nan on solver failure

            # Handle solver failure for this sample
            if pd.isna(x1_max):
                print(f"Skipping curve generation for sample index {idx} due to solver failure.")
                results.append({
                    'x1_max': np.nan,
                    'x1_candidates': np.array([]),
                    'predictions': [],
                    'highlight': None
                })
                continue # Skip to the next sample

            x1_max = max(0, x1_max) # Ensure rate is not negative

            # --- 2. Generate candidate x1 values and predict ---
            candidate_x1 = np.linspace(0, x1_max, num_points)
            all_predictions_formatted = [] # Initialize empty list

            if len(candidate_x1) > 0: # Should be true if num_points >= 2
                # Prepare DataFrame for batch prediction
                sample_base = sample.copy() # Use the sample where x1 was set to fill_value
                candidate_df = pd.DataFrame([sample_base] * len(candidate_x1)) # Replicate row
                candidate_df[x1_col] = candidate_x1 # Set the varying x1 values

                try:
                    # Use predict_raw for efficiency, format later
                    filtered_pred_kwargs = self._filter_solver_kwargs(kwargs)
                    raw_predictions = self.predict_raw(candidate_df, **filtered_pred_kwargs)

                    # Format predictions based on settings
                    initialpH_value = sample[self.initialpH_col] if self.predicts_delta_ph else 0
                    temp_formatted = []
                    for k, raw in enumerate(raw_predictions):
                         try:
                              temp_formatted.append(self._format_response(raw, initialpH_value))
                         except Exception as fmt_e:
                              print(f"Warning: Failed to format curve prediction for x1={candidate_x1[k]:.3f} in sample index {idx}: {fmt_e}")
                              temp_formatted.append(np.nan) # Use NaN if formatting fails
                    all_predictions_formatted = temp_formatted


                except Exception as e:
                    print(f"Error during batch prediction for curve (sample index {idx}): {e}. Predictions set to NaN.")
                    # Set all predictions to NaN for this curve
                    all_predictions_formatted = [np.nan] * len(candidate_x1)
            else:
                 # This case should ideally not be reached if num_points >= 2
                 candidate_x1 = np.array([])
                 all_predictions_formatted = []


            # --- 3. Calculate highlight point if original x1 was provided ---
            highlight = None
            # Check if original x1 was a valid number
            if not pd.isna(provided_x1):
                try:
                    # Prepare a single sample DataFrame with the *provided* x1 value
                    highlight_sample = sample.copy() # Start from base sample again
                    highlight_sample[x1_col] = provided_x1
                    highlight_df = highlight_sample.to_frame().T

                    # Predict for this specific x1 value
                    filtered_hl_kwargs = self._filter_solver_kwargs(kwargs)
                    highlight_raw_pred = self.predict_raw(highlight_df, **filtered_hl_kwargs)

                    # Format the prediction
                    initialpH_value = sample[self.initialpH_col] if self.predicts_delta_ph else 0
                    # Ensure raw pred is scalar before formatting
                    if not np.isscalar(highlight_raw_pred) and len(highlight_raw_pred)==1:
                        highlight_raw_pred_scalar = highlight_raw_pred[0]
                    elif np.isscalar(highlight_raw_pred):
                        highlight_raw_pred_scalar = highlight_raw_pred
                    else:
                         raise ValueError(f"Highlight prediction not scalar: {highlight_raw_pred}")

                    highlight_pred_formatted = self._format_response(highlight_raw_pred_scalar, initialpH_value)
                    highlight = (provided_x1, highlight_pred_formatted)

                except Exception as e:
                    print(f"Warning: Failed to calculate highlight prediction for sample index {idx} at provided x1={provided_x1}: {e}")
                    highlight = (provided_x1, "Prediction Error") # Indicate error

            # --- 4. Store results for this sample ---
            results.append({
                'x1_max': x1_max,
                'x1_candidates': candidate_x1,
                'predictions': all_predictions_formatted, # List of formatted preds or NaNs
                'highlight': highlight
            })

        return results # List of dictionaries

    # Note: get_params and set_params are inherited from BaseEstimator
    # and function correctly with the __init__ structure. No need to redefine
    # unless custom logic is added.
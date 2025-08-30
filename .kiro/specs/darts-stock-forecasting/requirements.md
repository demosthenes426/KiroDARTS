# Requirements Document

## Introduction

This feature implements a comprehensive DARTS-based time series forecasting system to predict stock prices 5 business days into the future. The system will process scaled multi-variate data, handle missing business days as custom holidays, and test multiple neural network models to find the best performing approach for stock price prediction.

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to convert stock price data into DARTS TimeSeries objects with custom holiday handling, so that I can properly model business day patterns while accounting for missing trading days.

#### Acceptance Criteria

1. WHEN a pandas DataFrame with datetime index and "Adjusted Close" column is provided THEN the system SHALL create a complete business day range from min to max timestamps
2. WHEN missing business days are identified THEN the system SHALL treat each missing date as a custom holiday
3. WHEN creating the TimeSeries THEN the system SHALL use a custom business day frequency that excludes all inferred holiday dates
4. WHEN validating the TimeSeries THEN the system SHALL ensure no NaN values are present and the index is strictly increasing
5. IF the DataFrame lacks "Close" or "Adjusted Close" columns THEN the system SHALL raise an appropriate error

### Requirement 2

**User Story:** As a data scientist, I want to properly scale and split my data for training, so that I can ensure no data leakage and maintain consistent scaling across train/validation/test sets.

#### Acceptance Criteria

1. WHEN preparing data for modeling THEN the system SHALL create a target y column for 5-day future price prediction
2. WHEN splitting data THEN the system SHALL use a 70/15/15 split for training/validation/testing
3. WHEN scaling data THEN the system SHALL fit StandardScaler only on training data and transform all sets consistently
4. WHEN creating target columns THEN the system SHALL ensure no data leakage by only using data available at prediction time
5. WHEN preprocessing THEN the system SHALL drop all non-numeric columns before feeding to models

### Requirement 3

**User Story:** As a data scientist, I want to test multiple DARTS neural network models, so that I can identify the best performing approach for stock price forecasting.

#### Acceptance Criteria

1. WHEN selecting models THEN the system SHALL test every DARTS neural network model that supports multiple feature columns
2. WHEN training models THEN the system SHALL exclude machine learning models like ARIMA
3. WHEN training THEN the system SHALL record and monitor train and validation loss for each model
4. WHEN training completes THEN the system SHALL save all necessary artifacts (model files, scalers) for future predictions
5. WHEN testing THEN the system SHALL run models in CPU mode for local execution

### Requirement 4

**User Story:** As a data scientist, I want comprehensive evaluation and visualization of model performance, so that I can assess prediction accuracy and compare different approaches.

#### Acceptance Criteria

1. WHEN models complete training THEN the system SHALL generate graphs showing prediction vs actual values for each model
2. WHEN evaluating performance THEN the system SHALL output statistics showing accuracy metrics for each model
3. WHEN displaying results THEN the system SHALL show head and tail of feature columns before feeding to models
4. WHEN training THEN the system SHALL ensure train and validation loss are decreasing
5. WHEN testing THEN the system SHALL validate that model performance has not substantially degraded

### Requirement 5

**User Story:** As a developer, I want a well-structured Python project with comprehensive testing and documentation, so that the system is maintainable and reliable.

#### Acceptance Criteria

1. WHEN structuring the project THEN the system SHALL be written as a central Python script structured like a Jupyter notebook with standardized cell division comments
2. WHEN creating documentation THEN the system SHALL maintain a PROJECT_REFERENCE.md file as the source of truth
3. WHEN implementing functionality THEN the system SHALL include unit tests that verify script interactions and data integrity
4. WHEN creating test files THEN the system SHALL prepend "TESTING" to all test-related filenames
5. WHEN troubleshooting THEN the system SHALL prepend "TROUBLESHOOTING" to all debugging-related filenames

### Requirement 6

**User Story:** As a data scientist, I want proper data validation and integrity checks, so that I can ensure the quality and consistency of input data throughout the pipeline.

#### Acceptance Criteria

1. WHEN processing data THEN the system SHALL validate data integrity using a comprehensive checklist
2. WHEN loading data THEN the system SHALL ensure the datetime index is properly sorted ascending
3. WHEN creating TimeSeries THEN the system SHALL validate that the number of TimeSeries points equals DataFrame rows
4. WHEN identifying holidays THEN the system SHALL ensure the custom holidays exactly match missing business days
5. WHEN preparing features THEN the system SHALL validate that all feature columns are numeric before model training
6. WHEN scaling features THEN the system SHALL validate that the mean and standard deviation for each scaled feature column are correct
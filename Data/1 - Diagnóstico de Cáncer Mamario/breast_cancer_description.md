# Dataset Description: Breast Cancer Diagnosis

## Overview

The Breast Cancer Diagnostic dataset is a commonly used dataset in machine learning for classification tasks, particularly in the domain of medical diagnostics. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, which are used to predict whether the mass is benign or malignant.

## Data Source

The data used in this analysis is sourced from the UCI Machine Learning Repository, available at [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

## Dataset Information

- **File Names**: breast_cancer_train.csv, breast_cancer_valid.csv, breast_cancer_test.csv
- **Purpose**: Dataset for machine learning analysis
- **Format**: Comma-separated values (CSV)
- **Variables**:
  - `mean_radius`: Mean of distances from center to points on the perimeter.
  - `mean_texture`: Standard deviation of gray-scale values.
  - `mean_perimeter`: Mean size of the core tumor.
  - `mean_area`: Mean area of the core tumor.
  - `mean_smoothness`: Mean of local variation in radius lengths.
  - `mean_compactness`: Mean of perimeter^2 / area - 1.0.
  - `mean_concavity`: Mean severity of concave portions of the contour.
  - `mean_concave_points`: Mean number of concave portions of the contour.
  - `mean_symmetry`: Mean symmetry.
  - `mean_fractal_dimension`: Mean of "coastline approximation" - 1.
  - `radius_se`: Standard error of the mean of distances from center to points on the perimeter.
  - `texture_se`: Standard error of gray-scale values.
  - `perimeter_se`: Standard error of the size of the core tumor.
  - `area_se`: Standard error of the area of the core tumor.
  - `smoothness_se`: Standard error of local variation in radius lengths.
  - `compactness_se`: Standard error of perimeter^2 / area - 1.0.
  - `concavity_se`: Standard error of severity of concave portions of the contour.
  - `concave_points_se`: Standard error for number of concave portions of the contour.
  - `symmetry_se`: Standard error for symmetry.
  - `fractal_dimension_se`: Standard error for "coastline approximation" - 1.
  - `radius_worst`: Worst or largest mean value for radius.
  - `texture_worst`: Worst or largest mean value for texture.
  - `perimeter_worst`: Worst or largest mean value for perimeter.
  - `area_worst`: Worst or largest mean value for area.
  - `smoothness_worst`: Worst or largest mean value for smoothness.
  - `compactness_worst`: Worst or largest mean value for compactness.
  - `concavity_worst`: Worst or largest mean value for concavity.
  - `concave_points_worst`: Worst or largest mean value for concave portions of the contour.
  - `symmetry_worst`: Worst or largest mean value for symmetry.
  - `fractal_dimension_worst`: Worst or largest mean value for "coastline approximation" - 1.
  - `diagnosis`: The diagnosis of breast tissues (1 = malignant, 0 = benign). Target variable.

## License

The data used in this analysis is freely available for public use from the UCI Machine Learning Repository. Please refer to their website for more information on data usage and licensing.

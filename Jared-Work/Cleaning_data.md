## Cleaning Data - Introduction
When it comes to cleaning data there are several approaches that we can take to dealing with 'unknown' values including:
- treat 'unknown' as a separate category
- remove rows with 'uknown' (deletion)
- impute missing values
	- impute with the mode (most frequent value)
	- impute using k-nearest neighbors
## Pros - Cons
1.  Treat 'unknown' as a separate category: 
	- Pros:
		-   Simple and easy to implement.
		-   Preserves all data points, making use of the entire dataset.
		-   Can be useful when 'unknown' values actually provide meaningful information.
	- Cons:
		-   May introduce bias if the 'unknown' category is not representative of the true underlying distribution.
		-   The model might learn patterns specific to the 'unknown' category, which might not be useful for predicting new, unseen data.

2.  Remove rows with 'unknown' values (deletion): 
	- Pros:
		-   Easy to implement.
		-   Can lead to more accurate predictions if missing values are completely at random and do not contain any useful information.
	- Cons:
		-   May result in loss of valuable information if missing values are not completely at random.
		-   Reduces the dataset size, which can negatively impact model performance, especially if a significant portion of the data contains 'unknown' values.

3.  Impute missing values: 
	-  Impute with the mode (most frequent value): 
		- Pros:
			-   Simple and easy to implement.
			-   Can be effective when missing values are few and not significantly different from the most frequent value in the attribute.
		- Cons:
			-   May introduce bias if the imputed values do not accurately represent the true underlying distribution.
			-   Assumes that the most frequent value is the best approximation for missing data, which might not always be the case.
	- Impute using K-Nearest Neighbors: 
		- Pros:
			-   More accurate imputation compared to mode imputation, as it considers similarities between data points.
			-   Can handle both numerical and categorical data.
		- Cons:
			-   Computationally expensive, especially for large datasets.
			-   Sensitive to the choice of k (number of neighbors) and distance metric used.
			-   Requires scaling of numerical features to ensure equal importance in distance computation.
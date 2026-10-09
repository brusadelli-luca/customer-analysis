# customer-analysis

Customer segmentation with clustering (k-means, hierarchical clustering, DBSCAN) on a marketing dataset, plus a k-means written from scratch and tested on the iris dataset.

Code written in March-April 2023. Review and fixes done with Claude in October 2026 (see the commits marked `Co-Authored-By: Claude`).

## Goal of the project
Group the customers of a company into segments from their profile and purchases, and compare several clustering methods:
1. explore and clean the data;
2. compare k-means on original and standardized data, and choose the number of clusters;
3. reduce the number of dimensions with a PCA (principal component analysis);
4. compare k-means with hierarchical clustering (CAH) and DBSCAN.

## Data
[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) by Akash Patel on Kaggle, license **CC0: Public Domain**.

The file is included in the repository: `data/marketing_campaign.csv` (tab-separated, 2240 rows, 29 columns, one row per customer). Column dictionary (from the dataset page):

| Group | Column | Meaning |
|---|---|---|
| People | `ID` | Customer's unique identifier |
| | `Year_Birth` | Customer's birth year |
| | `Education` | Customer's education level |
| | `Marital_Status` | Customer's marital status |
| | `Income` | Customer's yearly household income |
| | `Kidhome` | Number of children in customer's household |
| | `Teenhome` | Number of teenagers in customer's household |
| | `Dt_Customer` | Date of customer's enrollment with the company (day-month-year) |
| | `Recency` | Number of days since customer's last purchase |
| | `Complain` | 1 if the customer complained in the last 2 years, 0 otherwise |
| Products | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | Amount spent on wine, fruits, meat, fish, sweets and gold in the last 2 years |
| Promotion | `NumDealsPurchases` | Number of purchases made with a discount |
| | `AcceptedCmp1` to `AcceptedCmp5` | 1 if the customer accepted the offer in campaign 1 to 5, 0 otherwise |
| | `Response` | 1 if the customer accepted the offer in the last campaign, 0 otherwise |
| Place | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` | Number of purchases made through the website, a catalogue, or in stores |
| | `NumWebVisitsMonth` | Number of visits to the company's website in the last month |

`Z_CostContact` and `Z_Revenue` have the same value on every row and are not used.

## Requirements
* Python 3 (tested with Python 3.12 and 3.14)
* matplotlib, numpy, pandas, plotly, scikit-learn, listed in `requirements.txt`

From the project folder, create a virtual environment and install the libraries (Windows commands):

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
Open the notebooks in VS Code (with the Jupyter extension), or install Jupyter with `pip install jupyterlab` and run `jupyter lab`. Run each notebook with "Run All", from top to bottom: the notebooks read the data with the relative path `data/marketing_campaign.csv`.

The saved outputs come from a clean run (fresh kernel, top to bottom) with Python 3.12, so the results can be read on GitHub without running anything. Plotly charts (the dendrogram in `customer_analysis_compare`) are not displayed by GitHub: run the notebook to see them.

## Files
Customer analysis, in this order (each notebook repeats the cleaning steps, so each one runs on its own). The last cell of each notebook, **Conclusion**, prints the computed results.

| Notebook | What it does | Conclusion cell |
|---|---|---|
| `customer_analysis_visualization.ipynb` | Explores every column, decides the cleaning (rows with missing income, birth year before 1920, absurd marital status, income 666666), charts after cleaning, acceptance rate of each campaign (14.99 % for the last one) | 2240 rows before cleaning, 2208 after |
| `customer_analysis_preprocess.ipynb` | k-means with 3 clusters on original vs standardized data, elbow curves to choose the number of clusters | Silhouette 0.59 (original) / 0.16 (standardized) |
| `customer_analysis_features_eng.ipynb` | PCA: number of dimensions by Kaiser's rule, elbow and 80 % of variance, then k-means with 2 clusters, and a 3D view | Kaiser's rule 7 dimensions, 80 % threshold 14 dimensions; silhouette 0.26 (14 dimensions) / 0.46 (3 dimensions) |
| `customer_analysis_compare.ipynb` | Same 3-dimension PCA, hierarchical clustering (CAH) with 2 clusters and DBSCAN, with 2D and 3D views | CAH 0.45, DBSCAN -0.18 (8 clusters + noise) |

Custom k-means:
* `functions.py`: distance between two 2D points.
* `k_means_class.py`: class `kmeans` with `fit`, `predict` and `elbow` (option `random_state` for reproducible results).
* `k_means.ipynb`: step-by-step construction of the algorithm on iris (2 features), then use of the class.
* `iris_kmeans.ipynb`: comparison with scikit-learn on iris, silhouette 0.45 (scikit-learn) / 0.43 (custom), and elbow curve.

The silhouette score goes from -1 to 1: the higher it is, the better the clusters are separated.

## Known limitations
* The number of clusters is chosen with the elbow method in `customer_analysis_preprocess`, and the curve has no clear elbow: the conclusion is "2 or 3". The silhouette score (not computed there for this choice) is higher with 2 clusters, which is the number used in the next notebooks.
* On the original (not standardized) data, `Dt_Customer` is converted to seconds (about 10^9) and outweighs every other column, so the high silhouette score there mostly reflects a split by enrollment date.
* The DBSCAN parameter `eps = 0.5` was read on the nearest-neighbour distance chart.
* Custom k-means: works on 2 features only, runs a fixed number of iterations (10), and its "inertia" is a sum of distances (scikit-learn uses squared distances). A cluster that receives no point keeps its previous centroid; with an unlucky random start, a cluster can stay empty.
* Since scikit-learn 1.4, `KMeans` runs its initialization only once by default: the notebooks set `n_init = 10` to keep the 2023 results.

## Contributors
An exploration notebook by [hamdain-mazen](https://github.com/hamdain-mazen) (April 2023) was removed from the current version during the 2026 review because it was unfinished and overlapped `customer_analysis_visualization`; it is still in the git history.

import numpy as np
import pandas as pd
import json

# Perform matrix factorization with SVD
def funk_svd(matrix, k=5, learning_rate=0.01, iterations=100):
    # Initialize the user and item matrices with random values
    num_users, num_items = matrix.shape
    user_factors = np.random.rand(num_users, k)
    item_factors = np.random.rand(num_items, k)

    for _ in range(iterations):
        for i in range(num_users):
            for j in range(num_items):
                if matrix[i][j] != 0:
                    error_ij = matrix[i][j] - \
                        np.dot(user_factors[i], item_factors[j])
                    for f in range(k):
                        user_factors[i][f] += learning_rate * \
                            (2 * error_ij * item_factors[j][f])
                        item_factors[j][f] += learning_rate * \
                            (2 * error_ij * user_factors[i][f])

    return np.dot(user_factors, item_factors.T)

def perform_martix_factorization():
    # ? File that contains all the users in the mongodb cluster
    user_file = open(
        r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
    users = json.load(user_file)

    # ? The following lines are needed only the first time you run the code
    # import nltk
    # nltk.download('stopwords')

    # ? File that contains all the urls in the mongodb cluster
    url_file = open(
        r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf-8")
    urls = json.load(url_file)

    # Extract user IDs and URLs from the loaded data
    user_ids = [user["_id"]["$oid"] for user in users]
    all_urls = [url["url"] for url in urls]

    # Create an empty DataFrame with users as rows and URLs as columns
    user_url_df = pd.DataFrame(columns=all_urls, index=user_ids)
    for user in users:
        for url in user["links"]:
            if(url["url"] not in all_urls):
                continue
            user_url_df.loc[user["_id"]["$oid"], url["url"]] = url["rating"]

    # Assuming user_url_df is your pandas DataFrame
    # Fill NaN values with 0 
    user_url_df = user_url_df.fillna(0)
    user_url_matrix = user_url_df.to_numpy()

    # Perform matrix factorization
    predicted_matrix = funk_svd(user_url_matrix)

    # ? Round predicted number to the closest integer between [1,5]
    predicted_integers = np.round(predicted_matrix).astype(int)
    predicted_capped = np.clip(predicted_integers, a_min=1, a_max=5)

    new_user_url_df = pd.DataFrame(predicted_capped, columns=all_urls, index=user_ids)
    return new_user_url_df

# Optionally, you can save the user-url matrix to a CSV file
# new_user_url_df.to_csv(
#     r"C:\Users\ptria\source\repos\FlaskApi\ML\user_url_matrix.csv")

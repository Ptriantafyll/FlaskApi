from scipy.sparse import csr_matrix
import implicit
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
import json

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

print(len(user_ids))
print(len(all_urls))


# Create an empty DataFrame with users as rows and URLs as columns
user_url_df = pd.DataFrame(columns=all_urls, index=user_ids)
# print(user_url_df)
# print(len(user_url_df.columns))
# print("---")

# 'https://parade.com/947956/parade/riddles/'

# print(user_url_df['https://parade.com/947956/parade/riddles/'])

# print('https://parade.com/947956/parade/riddles/' in all_urls)
# print('https://chat.openai.com/?model=text-davinci-002-render-sha' in all_urls)
# print(all_urls[0])
# print(all_urls[781])
# print(all_urls[780])
# print(all_urls[779])
# print("https://parade.com/947956/parade/riddles/" in all_urls)

# for url in all_urls:
    # print(url)  


for user in users:
    # print(user["links"])
    for url in user["links"]:
        # print(user["_id"]["$oid"])
        # print(url["url"])
        # print(url["rating"])
        if(url["url"] not in all_urls):
            print(url["url"])
            continue
        user_url_df.loc[user["_id"]["$oid"], url["url"]] = url["rating"]

# Optionally, you can save the user-url matrix to a CSV file
# user_url_df.to_csv(
#     r"C:\Users\ptria\source\repos\FlaskApi\ML\user_url_matrix.csv")


# Assuming user_url_df is your pandas DataFrame
# Fill NaN values with 0 
user_url_df = user_url_df.fillna(0)
user_url_matrix = user_url_df.to_numpy()

# print(user_url_df)
# print(user_url_matrix)

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

    print(user_factors)
    print(item_factors)
    return np.dot(user_factors, item_factors.T)


# Perform matrix factorization
predicted_matrix = funk_svd(user_url_matrix)

print("original matrix")
print(user_url_matrix)
print("Predicted user-item matrix:")
# print(predicted_matrix)
predicted_integers = np.round(predicted_matrix).astype(int)
# print("Predicted user-item matrix integers:")
# print(predicted_integers)
predicted_capped = np.clip(predicted_integers, a_min=1, a_max=5)
# print("Predicted user-item matrix capped:")
print(predicted_capped)


new_user_url_df = pd.DataFrame(predicted_capped, columns=all_urls, index=user_ids)
# Optionally, you can save the user-url matrix to a CSV file
new_user_url_df.to_csv(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\user_url_matrix_new.csv")

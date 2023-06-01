import pandas as pd

# Load the csv file into a pandas DataFrame
# try:
#     df = pd.read_csv('data.csv')
# except FileNotFoundError:
#     print("CSV file not found.")
# except Exception as e:
#     print("Unexpected error: ", e)



# chunksize = 10 ** 6  # adjust this value depending on your available memory
# chunks = []
# for chunk in pd.read_csv('data.csv', chunksize=chunksize):
#     # process your data here
#     # For instance, you can filter rows and append to the chunks list
#     chunks.append(chunk[chunk['column_name'] == 'value'])

# # Concatenate all chunks
# df = pd.concat(chunks, axis=0)



df = pd.read_csv('Sport_Shoes.csv')
data = df[["category_id", "sub_category_id", "sub_sub_category_id", "brand_id", "image"]]
print(data)
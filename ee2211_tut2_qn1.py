import pandas as pd
import matplotlib.pyplot as plt

# def plot_data(file_path):
#     df = pd.read_csv(file_path)

#     df.set_index("DataSeries", inplace=True)

#     row = df.loc["Total Government Expenditure On Education"]

#     plt.plot(row.index, row.values)
#     plt.xlabel("Year")
#     plt.ylabel("Expenditure")
#     plt.title("Total Government Expenditure On Education")
#     plt.show()
    

df = pd.read_csv('GovernmentExpenditureOnEducationAnnual.csv')

exp = df['']

df.set_index("DataSeries", inplace=True)

row = df.loc["Total Government Expenditure On Education"]

plt.plot(row.index, row.values)
#     plt.xlabel("Year")
#     plt.ylabel("Expenditure")
#     plt.title("Total Government Expenditure On Education")
plt.show()


# plot_data('GovernmentExpenditureOnEducationAnnual.csv')


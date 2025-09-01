import pandas as pd
import matplotlib.pyplot as plt

def plot_data(file_path):
    df = pd.read_csv(file_path)

    df.set_index("DataSeries", inplace=True)

    expenditure = df.loc["Total Government Expenditure On Education"]

    plt.plot(expenditure.index, expenditure.values)
    plt.gca().invert_xaxis()
    plt.xlabel("Year")
    plt.ylabel("Expenditure")
    plt.title("Total Government Expenditure On Education")
    plt.show()
    plt.savefig("expenditure_plot2.png")
    


plot_data('GovernmentExpenditureOnEducationAnnual.csv')

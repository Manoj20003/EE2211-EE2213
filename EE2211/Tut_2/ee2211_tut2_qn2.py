import pandas as pd
import matplotlib.pyplot as plt 


def plot_data(file_path):
    df = pd.read_csv(file_path)


    df.set_index("year", inplace=True)

    
    omnibuses = df[df["type"] == "Omnibuses"]
    excursion_buses = df[df["type"] == "Excursion buses"]
    private_buses = df[df["type"] == "Private buses"]
    
    start_year = df.index.min()
    end_year = df.index.max()
    xticks = list(range(start_year, end_year + 1, 2))

    plt.plot(omnibuses.index, omnibuses["number"], label="Omnibuses")
    plt.plot(excursion_buses.index, excursion_buses["number"], label="Excursion buses")
    plt.plot(private_buses.index, private_buses["number"], label="Private buses")

    plt.xticks(xticks)

    plt.title("Annual Motor Vehicle Population by Vehicle Type")
    plt.xlabel("Year")
    plt.ylabel("Number")
    plt.legend()
    plt.savefig("qn_2.png")
    plt.show()
    


plot_data("AnnualMotorVehiclePopulationbyVehicleType.csv")





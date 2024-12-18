import matplotlib.pyplot as plt
import numpy as np
def plot_alpha(dataset):
    fig, ax = plt.subplots(nrows=4, ncols=len(dataset))
    x= [0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0]
    yl = ["CPD", "Total WL", "Average WL", "VPR Runtime"]
    
    
    for i in range(len(dataset)):
        ax[0,i].set_title("design "+str(i), fontsize=14)
        ax[3,i].set_xlabel("Alpha Value ", fontsize=14)
        for j in range(len(dataset[i])):
            y = [dataset[i][j][0],dataset[i][j][3],dataset[i][j][2],dataset[i][j][1],dataset[i][j][4],dataset[i][j][5],dataset[i][j][6]  ]
            ax[j,i].plot(x, dataset[i][j])        
            ax[j,0].set_ylabel(yl[j], fontsize=14)

        
    plt.legend()
    plt.show()


if __name__ == "__main__":

    first_data = [
        # cpd = 
        [
        7.20122,
        5.00433,
        5.9729,
        5.22486,
        5.18015,
        5.30066,
        5.03558
        ],

        # wl =
        [
        876634,
        1033036,
        938705,
        911698,
        1072826,
        964584,
        1078714
        ],

        # avg_wl=
        [
        14.8168,
        14.2307,
        13.9974,
        15.1068,
        14.5092,
        12.9647,
        14.3519],

        # time=
        [
        1070.39,
        1107.69,
        1088.24,
        1069.09,
        1102.43,
        1100.47,
        1106.84]


    ]
    second_data = [
        [191.425,
        186.031,
        187.029,
        186.762,
        190.811,
        191.437,
        192.751],

        [428354,
        448947,
        447596,
        429716,
        442627,
        451800,
        509735],

        [13.2638,
        13.8282,
        13.8211,
        13.2485,
        13.6774,
        13.7046,
        13.9215],

        [285.33,
        305.67,
        298.86,
        293.40,
        284.54,
        290.41,
        311.91]

    ]



    plot_alpha([first_data, second_data])






from MASOIE.masoie import masoie
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    funcID = "100D5n-2"
    repeatTimes = 1

    archives = []
    for ep in range(repeatTimes):
        print(f"{'-' * 20} START[{ep}] {'-' * 20}")
        archives.append(masoie(funcID))
        print(f"{'-' * 20} FINISH[{ep}] {'-' * 20}")
    
    meanResult = dict()
    meanResult["TotalEvaluationTimes"] = []
    meanResult["Fitness"] = []
    meanResult["Disagreement"] = []

    sampleIndex = 0
    while True:
        counter = 0
        TotalEvaluationTimes = 0
        Fitness = 0
        Disagreement = 0
        for i in range(repeatTimes):
            if sampleIndex < len(archives[i]["TotalEvaluationTimes"]):
                counter += 1
                TotalEvaluationTimes += archives[i]["TotalEvaluationTimes"][sampleIndex]
                Fitness += archives[i]["Fitness"][sampleIndex]
                Disagreement += archives[i]["Disagreement"][sampleIndex]
        if counter == 0:
            break

        meanResult["TotalEvaluationTimes"].append(TotalEvaluationTimes / counter)
        meanResult["Fitness"].append(Fitness / counter)
        meanResult["Disagreement"].append(Disagreement / counter)

        sampleIndex += 1
    
    BestArchiveIndex = 0
    WorstArchiveIndex = 0
    for i in range(repeatTimes):
        if archives[i]["Fitness"][-1] < archives[BestArchiveIndex]["Fitness"][-1]:
            BestArchiveIndex = i
        if archives[i]["Fitness"][-1] > archives[WorstArchiveIndex]["Fitness"][-1]:
            WorstArchiveIndex = i
    
    print(f"{'-' * 50}")
    print(f"{funcID}")
    print(f"Mean:")
    print(f"\tTotalEva: {meanResult['TotalEvaluationTimes'][-1]}")
    print(f"\tFitness:  {'{:g}'.format(meanResult['Fitness'][-1])}")
    print(f"\tDisagree: {meanResult['Disagreement'][-1]}")
    print(f"Best:")
    print(f"\tTotalEva: {archives[BestArchiveIndex]['TotalEvaluationTimes'][-1]}")
    print(f"\tFitness:  {'{:g}'.format(archives[BestArchiveIndex]['Fitness'][-1])}")
    print(f"\tDisagree: {archives[BestArchiveIndex]['Disagreement'][-1]}")
    print(f"Worst:")
    print(f"\tTotalEva: {archives[WorstArchiveIndex]['TotalEvaluationTimes'][-1]}")
    print(f"\tFitness:  {'{:g}'.format(archives[WorstArchiveIndex]['Fitness'][-1])}")
    print(f"\tDisagree: {archives[WorstArchiveIndex]['Disagreement'][-1]}")
    print(f"{'-' * 50}")
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    
    axs[0].set_xlabel('TotalEvaluations')
    axs[0].set_ylabel("Fitness")
    axs[0].plot(np.array(meanResult["TotalEvaluationTimes"]), np.array(meanResult["Fitness"]), '#ff9900', label="mean")
    axs[0].plot(np.array(archives[BestArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[BestArchiveIndex]["Fitness"]), '#6aa84f', label="best")
    axs[0].plot(np.array(archives[WorstArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[WorstArchiveIndex]["Fitness"]), '#4a86e8', label="worst")
    axs[0].legend()

    axs[1].set_xlabel('TotalEvaluations')
    axs[1].set_ylabel("Disagreement")
    axs[1].plot(np.array(meanResult["TotalEvaluationTimes"]), np.array(meanResult["Disagreement"]), '#ff9900', label="mean")
    axs[1].plot(np.array(archives[BestArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[BestArchiveIndex]["Disagreement"]), '#6aa84f', label="best")
    axs[1].plot(np.array(archives[WorstArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[WorstArchiveIndex]["Disagreement"]), '#4a86e8', label="worst")
    axs[1].legend()

    plt.savefig(f'{funcID}.png')
    plt.show()
#ifndef BENCHMARK_CSV_EXPORTER_H
#define BENCHMARK_CSV_EXPORTER_H

#include <fstream>
#include <string>

void exportBenchmarkResultsToCSV(const std::string& filename, const std::string& benchmarkName, int iterations, double realTime, double cpuTime)
{
    std::ofstream outputFile(filename, std::ios::out | std::ios::app);
    if (!outputFile.is_open()) {
        // Handle error if unable to open file
        std::cout << "Cannot open file. Fix the issue before trying again. \n No data was written.\n";
        return;
    }

    outputFile << benchmarkName << "," << iterations << "," << realTime << "," << cpuTime << "\n";

    outputFile.close();
}

#endif // BENCHMARK_CSV_EXPORTER_H

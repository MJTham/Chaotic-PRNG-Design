import os
import sys
from randomness_testsuite.FrequencyTest import FrequencyTest
from randomness_testsuite.RunTest import RunTest
from randomness_testsuite.Matrix import Matrix
from randomness_testsuite.Spectral import SpectralTest
from randomness_testsuite.TemplateMatching import TemplateMatching
from randomness_testsuite.Universal import Universal
from randomness_testsuite.Complexity import ComplexityTest
from randomness_testsuite.Serial import Serial
from randomness_testsuite.ApproximateEntropy import ApproximateEntropy
from randomness_testsuite.CumulativeSum import CumulativeSums
from randomness_testsuite.RandomExcursions import RandomExcursions

# ✅ Define Result Saving Function
def save_results_to_file(results, filename="output/NIST_test_results.txt"):
    os.makedirs("output", exist_ok=True)  # ✅ Ensure output folder exists
    with open(filename, "w", encoding="utf-8") as file:
        file.writelines(results)
    print(f"\n✅ NIST test results saved to: {filename}")

# ✅ Open PRNG Output File in Binary Mode
data_path = os.path.join(os.getcwd(), 'output', 'prng_output', 'prng_output.bin')

with open(data_path, "rb") as handle:
    binary_data = ''.join(format(byte, '08b') for byte in handle.read())  # ✅ Convert binary data to a bit string

# ✅ Perform NIST Randomness Tests & Capture Results
results = []
results.append('The statistical test of the PRNG Output\n\n')

results.append(f'2.01. Frequency Test:                               {FrequencyTest.monobit_test(binary_data[:1000000])}\n')
results.append(f'2.02. Block Frequency Test:                         {FrequencyTest.block_frequency(binary_data[:1000000])}\n')
results.append(f'2.03. Run Test:                                     {RunTest.run_test(binary_data[:1000000])}\n')
results.append(f'2.04. Run Test (Longest Run of Ones):              {RunTest.longest_one_block_test(binary_data[:1000000])}\n')
results.append(f'2.05. Binary Matrix Rank Test:                     {Matrix.binary_matrix_rank_text(binary_data[:1000000])}\n')
results.append(f'2.06. Discrete Fourier Transform (Spectral) Test:  {SpectralTest.spectral_test(binary_data[:1000000])}\n')
results.append(f'2.07. Non-overlapping Template Matching Test:      {TemplateMatching.non_overlapping_test(binary_data[:1000000], "000000001")}\n')
results.append(f'2.08. Overlapping Template Matching Test:         {TemplateMatching.overlapping_patterns(binary_data[:1000000])}\n')
results.append(f'2.09. Universal Statistical Test:                  {Universal.statistical_test(binary_data[:1000000])}\n')
results.append(f'2.10. Linear Complexity Test:                      {ComplexityTest.linear_complexity_test(binary_data[:1000000])}\n')
results.append(f'2.11. Serial Test:                                 {Serial.serial_test(binary_data[:1000000])}\n')
results.append(f'2.12. Approximate Entropy Test:                    {ApproximateEntropy.approximate_entropy_test(binary_data[:1000000])}\n')
results.append(f'2.13. Cumulative Sums (Forward):                   {CumulativeSums.cumulative_sums_test(binary_data[:1000000], 0)}\n')
results.append(f'2.13. Cumulative Sums (Backward):                  {CumulativeSums.cumulative_sums_test(binary_data[:1000000], 1)}\n')

# ✅ Random Excursion Test
results.append('2.14. Random Excursion Test:\n')
results.append('\t\t STATE \t\t\t xObs \t\t\t P-Value \t\t\t Conclusion\n')

excursion_results = RandomExcursions.random_excursions_test(binary_data[:1000000])
for item in excursion_results:
    results.append(f'\t\t {repr(item[0]).rjust(4)} \t\t {item[2]} \t\t {repr(item[3]).ljust(14)} \t\t {(item[4] >= 0.01)}\n')

# ✅ Random Excursion Variant Test
results.append('2.15. Random Excursion Variant Test:\n')
results.append('\t\t STATE \t\t COUNTS \t\t\t P-Value \t\t Conclusion\n')

variant_results = RandomExcursions.variant_test(binary_data[:1000000])
for item in variant_results:
    results.append(f'\t\t {repr(item[0]).rjust(4)} \t\t {item[2]} \t\t {repr(item[3]).ljust(14)} \t\t {(item[4] >= 0.01)}\n')

# ✅ Print results to console & Save to File
print(''.join(results))
save_results_to_file(results)

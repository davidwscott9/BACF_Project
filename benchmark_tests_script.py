from benchmark_tests import benchmark_tests
# This script runs the benchmark testing to generate success plots and obtain the FPS.
# Available test_types are:
# 'OTB50'
# 'OTB100'
# 'TC128'
OP_vid, FPS_vid = benchmark_tests(test_type='OTB50')

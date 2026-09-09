# Finite supplied coefficients can still overflow the recurrence. The comparison
# must reject that output instead of accepting a NaN relative error.
file(MAKE_DIRECTORY "${OUTPUT}")
file(WRITE "${OUTPUT}/parameters.txt" "overflow 0.4 3e38 -0.95 0.1 0.004 0.0003 -0.95 0.1 0.001\n")
execute_process(COMMAND "${RENDERER}" "${OUTPUT}" --parameters "${OUTPUT}/parameters.txt"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(result EQUAL 0 OR NOT stderr MATCHES "Nonfinite Conan GPU output")
    message(FATAL_ERROR "Expected nonfinite synthesis rejection, got ${result}: ${stdout}${stderr}")
endif()

file(WRITE "${OUTPUT}/parameters.txt" "paper 0.35 0.04 -0.95 0.195 0.00475 0.00052 -0.95 0.005 0.000394\n")
execute_process(COMMAND "${RENDERER}" "${OUTPUT}" --parameters "${OUTPUT}/parameters.txt"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(NOT result EQUAL 0 OR NOT EXISTS "${OUTPUT}/paper-seed0.wav")
    message(FATAL_ERROR "Paper-parameter reproduction failed: ${result}: ${stdout}${stderr}")
endif()

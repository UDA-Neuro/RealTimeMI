BCI2000_ADD_SIGNAL_PROCESSING_MODULE(RealTimeMI
    PipeDefinition.cpp
    ModulationIndexFilter.cpp
)
target_include_directories(RealTimeMI PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_features(RealTimeMI PRIVATE cxx_std_17)

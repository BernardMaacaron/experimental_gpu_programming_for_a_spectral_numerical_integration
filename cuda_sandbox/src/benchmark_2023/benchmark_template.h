    
    //Template

    
    ::benchmark::RegisterBenchmark("<NAME>", [&](::benchmark::State &t_state){
        for (auto _ : t_state) {
            auto start = std::chrono::high_resolution_clock::now();
            //insert benchmark code here
            auto end = std::chrono::high_resolution_clock::now();

            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            t_state.SetIterationTime(elapsed_seconds.count());
        }
    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);



    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
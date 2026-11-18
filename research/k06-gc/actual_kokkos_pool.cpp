// K06 controlled native CPU backend test for Kokkos Random Pool.
// This runs the REAL Kokkos::Random_XorShift64_Pool::get_state(), not a model.
// GPU cannot be tested on standard GitHub-hosted Ubuntu runners.
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: actual_kokkos_pool <states> <count>\n");
    return 2;
  }
  const int num_states = std::atoi(argv[1]);
  const int count      = std::atoi(argv[2]);
  if (num_states < 1 || count < 1) return 2;

  Kokkos::initialize(argc, argv);
  {
    using Exec = Kokkos::OpenMP;
    using Pool = Kokkos::Random_XorShift64_Pool<Exec>;
    auto exec = Exec{};
    std::printf("K06_NATIVE_BACKEND=%s OMP_CONCURRENCY=%d STATES=%d COUNT=%d\n",
                Exec::name(), exec.concurrency(), num_states, count);
    std::fflush(stdout);

    Pool pool(42, static_cast<uint64_t>(num_states));
    Kokkos::View<uint64_t*, Exec> output("output", count);
    Kokkos::parallel_for(
        "K06_native_random_pool", Kokkos::RangePolicy<Exec>(exec, 0, count),
        KOKKOS_LAMBDA(const int i) {
          auto generator = pool.get_state();
          output(i) = generator.urand64();
          pool.free_state(generator);
        });
    exec.fence("K06 native random pool fence");

    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    uint64_t checksum = 0;
    for (int i = 0; i < count; ++i) checksum ^= host(i);
    std::printf("K06_NATIVE_COMPLETE states=%d checksum=%llu\n", num_states,
                static_cast<unsigned long long>(checksum));
  }
  Kokkos::finalize();
  return 0;
}

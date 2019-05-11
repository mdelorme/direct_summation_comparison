#include <bits/stdc++.h>
#include <Kokkos_Core.hpp>

#ifdef VERBOSE
#define DBG(x) (std::cerr << x << std::endl;)
#else
#define DBG(x)
#endif

// Timers
using time_point = std::chrono::time_point<std::chrono::system_clock>;
using duration = std::chrono::duration<float>;
#define now std::chrono::system_clock::now

double dt = 1.0e-4;
int max_ite = 1000;
int dump_freq = 10;

std::string input_filename = "../data/ic_1000.dat";

// KOKKOS Kernels and structures
using StdVecRef = std::vector<double>;
using KokkosVec = Kokkos::View<double*>;
using KokkosArray = Kokkos::View<double**, Kokkos::LayoutLeft>;
struct Sim { // Isn't there any way to avoid encapsulating this in a struct ?
  KokkosArray x, v, a;
  KokkosVec m;

  Sim(int N) :
    x("x", N, 3),
    v("v", N, 3),
    m("m", N),
    a("a", N, 3) {};
};

// Resetting Accelerations
struct ResetAccelerationsFunc {
  KokkosArray a;
  ResetAccelerationsFunc(const Sim &sim) : a(sim.a) {}; 

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    a(i, 0) = 0.0;
    a(i, 1) = 0.0;
    a(i, 2) = 0.0;
  }
};

// Updating positions
struct UpdatePositionsFunc {
  KokkosArray x, v;
  UpdatePositionsFunc(const Sim &sim) : x(sim.x), v(sim.v) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    x(i, 0) += v(i, 0) * dt;
    x(i, 1) += v(i, 1) * dt;
    x(i, 2) += v(i, 2) * dt;
  };
};

// Updating velocities
struct UpdateVelocitiesFunc {
  KokkosArray v, a;
  UpdateVelocitiesFunc(const Sim &sim) : v(sim.v), a(sim.a) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    v(i, 0) += 0.5 * a(i, 0) * dt;
    v(i, 1) += 0.5 * a(i, 1) * dt;
    v(i, 2) += 0.5 * a(i, 2) * dt;
  };
};

// Computing accelerations
struct ComputeAccelerationsFunc {
  KokkosArray x, a;
  KokkosVec m;
  int i;

  ComputeAccelerationsFunc(const Sim &sim) : x(sim.x), a(sim.a), m(sim.m) {};

  void set_active_particle(int i) {
    this->i = i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (int j) const {
    double dx = x(i, 0) - x(j, 0);
    double dy = x(i, 1) - x(j, 1);
    double dz = x(i, 2) - x(j, 2);
    
    double rij = sqrt(dx*dx+dy*dy+dz*dz);
    
    double f = -m(i) * m(j)/ (rij*rij*rij);
    double ai = f / m(i);
    double aj = f / m(j);
    
    a(i, 0) += dx * ai;
    a(i, 1) += dy * ai;
    a(i, 2) += dz * ai;
    a(j, 0) -= dx * aj;
    a(j, 1) -= dy * aj;
    a(j, 2) -= dz * aj;
  }
};



int N;

void parse_args(int argc, char **argv) {
  int iarg = 1;
  while (iarg < argc) {
    std::string arg {argv[iarg]};
    
    if (arg == "--ic") {
      input_filename = argv[iarg+1];
      iarg++;
    }
    else if (arg == "-N") {
      max_ite = std::atoi(argv[iarg+1]);
      iarg++;
    }
    else if (arg == "-D") {
      dump_freq = std::atoi(argv[iarg+1]);
      iarg++;
    }
    else if (arg == "--dt") {
      dt = std::strtod(argv[iarg+1], nullptr);
      iarg++;
    }
    
    iarg++;
  }
}

Sim load_ics() {
  std::ifstream f_in;
  f_in.open(input_filename);

  if (!f_in.good()) {
    std::cerr << "Error opening file " << input_filename << std::endl;
    std::exit(1);
  }

  N = 0;
  DBG("Reading ICs ... ");
  std::vector<double> xv, yv, zv, vxv, vyv, vzv, mv;
  while (f_in.good()) {
    double _x, _y, _z, _vx, _vy, _vz, _m;
    f_in >> _x >> _y >> _z >> _vx >> _vy >> _vz >> _m;

    if (!f_in.good())
      break;

    // Storing
    xv.push_back(_x);
    yv.push_back(_y);
    zv.push_back(_z);
    vxv.push_back(_vx);
    vyv.push_back(_vy);
    vzv.push_back(_vz);
    mv.push_back(_m);
    N++;
  }
  
  DBG(N << " particles read");
  f_in.close();

  // Creating the structures and copying data
  Sim s(N);
  
  // Inefficient, but quick to code ... (Not included in timing)
  Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
      s.x(i, 0) = xv[i];
      s.x(i, 1) = yv[i];
      s.x(i, 2) = zv[i];

      s.v(i, 0) = vxv[i];
      s.v(i, 1) = vyv[i];
      s.v(i, 2) = vzv[i];

      s.m(i) = mv[i];
    });

  return s;
}


void dump_data(const Sim &s, int ite) {
  std::ostringstream oss;
  oss << "output/snap_" << std::setw(5) << std::setfill('0') << ite << ".dat";
  std::string filename {oss.str()};

  std::ofstream f_out;

  f_out.open(filename);

  if (!f_out.good()) {
    std::cerr << "Error trying to save output" << std::endl;
    std::exit(1);
  }

  for (int i=0; i < N; ++i)
    f_out << s.x(i, 0) << " " << s.x(i, 1) << " " << s.x(i, 2) << " "
	  << s.v(i, 0) << " " << s.v(i, 1) << " " << s.v(i, 2) << std::endl;

  f_out.close();
}

void compute_all_accelerations(auto &reset_accelerations,
			       auto &compute_accelerations) {
  Kokkos::parallel_for(N, reset_accelerations);
  for (int i=0; i < N-1; ++i) {
    compute_accelerations.set_active_particle(i);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(i+1, N), compute_accelerations);
  }
}

void run_sim(const Sim &s) {
  time_point start = now();
  
  // Initializing Kernels
  ResetAccelerationsFunc   reset_accelerations(s);
  ComputeAccelerationsFunc compute_accelerations(s);
  UpdatePositionsFunc      update_positions(s);
  UpdateVelocitiesFunc     update_velocities(s);
  
  // First half-step
  DBG("Calculating ...");

  compute_all_accelerations(reset_accelerations, compute_accelerations);
  Kokkos::parallel_for(N, update_velocities);
  
  dump_data(s, 0);
  
  int dump_ite = dump_freq;
  int dump_counter = 1;

  for (int i=0; i < max_ite; ++i) {
    // Update positions
    Kokkos::parallel_for(N, update_positions);

    compute_all_accelerations(reset_accelerations, compute_accelerations);
    Kokkos::parallel_for(N, update_velocities);

    dump_ite--;
    if (dump_ite == 0) {
      dump_data(s, dump_counter++);
      dump_ite = dump_freq;

      DBG("Dumping iteration " << i+1 << "/" << max_ite)
    }
  }
  time_point end = now();

  duration elapsed = end-start;
  std::cerr << max_ite << " iterations computed in " << elapsed.count() << "s" << std::endl;
  dump_data(s, dump_counter);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    parse_args(argc, argv);
    Sim s = load_ics();
    run_sim(s);
  }
  Kokkos::finalize();
  
  return 0;
}

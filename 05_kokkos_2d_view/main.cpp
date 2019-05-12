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
int N;

std::string input_filename = "../data/ic_1000.dat";

// KOKKOS Kernels and structures
using Layout = Kokkos::LayoutRight; // Pick the adequate Layout here

using StdVecRef   = std::vector<double>;
using KokkosVec   = Kokkos::View<double*>;
using KokkosArray = Kokkos::View<double**, Layout>;

struct Sim { // Isn't there any way to avoid encapsulating this in a struct ?
  KokkosArray x, v, a;
  KokkosVec m;

  Sim(int N) :
    x("x", N, 3),
    v("v", N, 3),
    a("a", N, 3),
    m("m", N) {};
};

// Copying from vector to vector View 
struct CopyVecFunc {
  KokkosVec dest;
  const StdVecRef &src;
  CopyVecFunc(const StdVecRef &src, const KokkosVec dest) : src(src), dest(dest) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    dest(i) = src[i];
  }
};

// Copying from vectors to array View
struct CopyArrayFunc {
  KokkosArray dest;
  const StdVecRef &x, &y, &z;
  CopyArrayFunc(const StdVecRef &x, const StdVecRef &y, const StdVecRef &z,
		const KokkosArray dest) : x(x), y(y), z(z), dest(dest) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    dest(i, 0) = x[i];
    dest(i, 1) = y[i];
    dest(i, 2) = z[i];
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
  KokkosVec m;
  KokkosArray x, a;
  int i;

  ComputeAccelerationsFunc(const Sim &sim) : x(sim.x), a(sim.a), m(sim.m) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    double tmp_ax = 0.0;
    double tmp_ay = 0.0;
    double tmp_az = 0.0;
    
    for (int j=0; j < N; ++j) {
      if (i==j)
	continue;
      
      double dx = x(i, 0) - x(j, 0);
      double dy = x(i, 1) - x(j, 1);
      double dz = x(i, 2) - x(j, 2);
    
      double rij = sqrt(dx*dx+dy*dy+dz*dz);
      
      double a = -m(j)/ (rij*rij*rij);
      
      tmp_ax += dx * a;
      tmp_ay += dy * a;
      tmp_az += dz * a;
    }

    a(i, 0) = tmp_ax;
    a(i, 1) = tmp_ay;
    a(i, 2) = tmp_az;
  }
};

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
  
  // Copying in a parallel_for
  Kokkos::parallel_for(N, CopyArrayFunc( xv,  yv,  zv, s.x));
  Kokkos::parallel_for(N, CopyArrayFunc(vxv, vyv, vzv, s.v));
  Kokkos::parallel_for(N, CopyVecFunc(mv, s.m));

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
	  << s.v(i, 0) << " " << s.x(i, 1) << " " << s.x(i, 2) << std::endl;

  f_out.close();
}

void run_sim(const Sim &s) {
  time_point start = now();
  
  // Initializing Kernels
  ComputeAccelerationsFunc compute_accelerations(s);
  UpdatePositionsFunc      update_positions(s);
  UpdateVelocitiesFunc     update_velocities(s);
  
  // First half-step
  DBG("Calculating ...");

  Kokkos::parallel_for(N, compute_accelerations);
  Kokkos::parallel_for(N, update_velocities);
  
  dump_data(s, 0);
  
  int dump_ite = dump_freq;
  int dump_counter = 1;

  for (int i=0; i < max_ite; ++i) {
    // Update positions
    Kokkos::parallel_for(N, update_positions);

    Kokkos::parallel_for(N, compute_accelerations);
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

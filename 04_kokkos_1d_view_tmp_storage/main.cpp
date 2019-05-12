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
using Kokkos::MemoryTraits;
using Kokkos::Atomic;
using StdVecRef = std::vector<double>;
using KokkosVec = Kokkos::View<double*>;
using KokkosAtomicVec = Kokkos::View<double*, MemoryTraits<Atomic>>;
struct Sim { // Isn't there any way to avoid encapsulating this in a struct ?
  KokkosVec x, y, z, vx, vy, vz, m, ax, ay, az;

  Sim(int N) :
    x ("x",  N),
    y ("y",  N),
    z ("z",  N),
    vx("vx", N),
    vy("vy", N),
    vz("vz", N),
    m ("m",  N),
    ax("ax", N),
    ay("ay", N),
    az("az", N) {};
};

// Copying from vector to View
struct CopyDataFunc {
  KokkosVec dest;
  const StdVecRef &src;
  CopyDataFunc(const StdVecRef &src, const KokkosVec dest) : src(src), dest(dest) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    dest(i) = src[i];
  }
};

// Resetting Accelerations
struct ResetAccelerationsFunc {
  KokkosVec ax, ay, az;
  ResetAccelerationsFunc(const Sim &sim) : ax(sim.ax), ay(sim.ay), az(sim.az) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    ax(i) = 0.0;
    ay(i) = 0.0;
    az(i) = 0.0;
  }
};

// Updating positions
struct UpdatePositionsFunc {
  KokkosVec x, y, z, vx, vy, vz;
  UpdatePositionsFunc(const Sim &sim) : x(sim.x), y(sim.y), z(sim.z),
					vx(sim.vx), vy(sim.vy), vz(sim.vz) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    x(i) += vx(i) * dt;
    y(i) += vy(i) * dt;
    z(i) += vz(i) * dt;
  };
};

// Updating velocities
struct UpdateVelocitiesFunc {
  KokkosVec vx, vy, vz, ax, ay, az;
  UpdateVelocitiesFunc(const Sim &sim) : vx(sim.vx), vy(sim.vy), vz(sim.vz),
					 ax(sim.ax), ay(sim.ay), az(sim.az) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    vx(i) += 0.5 * ax(i) * dt;
    vy(i) += 0.5 * ay(i) * dt;
    vz(i) += 0.5 * az(i) * dt;
  };
};

// Computing accelerations
struct ComputeAccelerationsFunc {
  KokkosVec x, y, z, m;
  KokkosAtomicVec ax, ay, az;
  int i;

  ComputeAccelerationsFunc(const Sim &sim) : x(sim.x), y(sim.y), z(sim.z), m(sim.m) {
    ax = KokkosAtomicVec(sim.ax);
    ay = KokkosAtomicVec(sim.ay);
    az = KokkosAtomicVec(sim.az);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    double tmp_ax = 0.0;
    double tmp_ay = 0.0;
    double tmp_az = 0.0;
    
    for (int j=0; j < N; ++j) {
      if (i==j)
	continue;
      
      double dx = x(i) - x(j);
      double dy = y(i) - y(j);
      double dz = z(i) - z(j);
    
      double rij = sqrt(dx*dx+dy*dy+dz*dz);
      
      double a = -m(j)/ (rij*rij*rij);
      
      tmp_ax += dx * a;
      tmp_ay += dy * a;
      tmp_az += dz * a;
    }

    ax(i) = tmp_ax;
    ay(i) = tmp_ay;
    az(i) = tmp_az;
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
  Kokkos::parallel_for(N, CopyDataFunc(xv,  s.x));
  Kokkos::parallel_for(N, CopyDataFunc(yv,  s.y));
  Kokkos::parallel_for(N, CopyDataFunc(zv,  s.z));
  Kokkos::parallel_for(N, CopyDataFunc(vxv, s.vx));
  Kokkos::parallel_for(N, CopyDataFunc(vyv, s.vy));
  Kokkos::parallel_for(N, CopyDataFunc(vzv, s.vz));
  Kokkos::parallel_for(N, CopyDataFunc(mv,  s.m));

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
    f_out << s.x(i)  << " " << s.y(i)  << " " << s.z(i) << " "
	  << s.vx(i) << " " << s.vy(i) << " " << s.vz(i) << std::endl;

  f_out.close();
}

inline void compute_all_accelerations(auto &reset_accelerations,
			       auto &compute_accelerations) {
  //  Kokkos::parallel_for(N, reset_accelerations);
  Kokkos::parallel_for(N, compute_accelerations);
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

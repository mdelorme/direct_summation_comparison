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

std::vector<double> x, y, z, vx, vy, vz, m, ax, ay, az;
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

void load_ics() {
  std::ifstream f_in;
  f_in.open(input_filename);

  if (!f_in.good()) {
    std::cerr << "Error opening file " << input_filename << std::endl;
    std::exit(1);
  }

  N = 0;
  DBG("Reading ICs ... ");
  while (f_in.good()) {
    double _x, _y, _z, _vx, _vy, _vz, _m;
    f_in >> _x >> _y >> _z >> _vx >> _vy >> _vz >> _m;

    if (!f_in.good())
      break;

    // Storing
    x.push_back(_x);
    y.push_back(_y);
    z.push_back(_z);
    vx.push_back(_vx);
    vy.push_back(_vy);
    vz.push_back(_vz);
    m.push_back(_m);
    N++;
  }

  ax.resize(N);
  ay.resize(N);
  az.resize(N);

  DBG(N << " particles read")

  f_in.close();
}

// Computation kernels
inline void compute_acceleration() {
  Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
      ax[i] = 0.0;
      ay[i] = 0.0;
      az[i] = 0.0;
    });

  
  for (int i=0; i < N-1; ++i) {
    Kokkos::parallel_for(Kokkos::RangePolicy<>(i+1, N),
			 KOKKOS_LAMBDA (int j) {
			   double dx = x[i] - x[j];
			   double dy = y[i] - y[j];
			   double dz = z[i] - z[j];
			   
			   double rij = sqrt(dx*dx+dy*dy+dz*dz);

			   double a = -m[i] * m[j]/ (rij*rij*rij);
			   double ai = a / m[i];
			   double aj = a / m[j];
			   
			   ax[i] += dx * ai;
			   ay[i] += dy * ai;
			   az[i] += dz * ai;
			   ax[j] -= dx * aj;
			   ay[j] -= dy * aj;
			   az[j] -= dz * aj;
			 });
  }
}

inline void update_velocities() {
  Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
      vx[i] += ax[i] * 0.5 * dt;
      vy[i] += ay[i] * 0.5 * dt;
      vz[i] += az[i] * 0.5 * dt;
    });
}

void dump_data(int ite) {
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
    f_out << x[i] << " " << y[i] << " " << z[i] << " " << vx[i] << " " << vy[i] << " " << vz[i] << std::endl;

  f_out.close();
}

void run_sim() {
  time_point start = now();

  // First half-step
  DBG("Calculating ...")
  compute_acceleration();
  update_velocities();

  dump_data(0);
  
  int dump_ite = dump_freq;
  int dump_counter = 1;

  for (int i=0; i < max_ite; ++i) {
    // Update positions
    Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
	x[i] += vx[i] * dt;
	y[i] += vy[i] * dt;
	z[i] += vz[i] * dt;
      });
    
    compute_acceleration();
    update_velocities();

    dump_ite--;
    if (dump_ite == 0) {
      dump_data(dump_counter++);
      dump_ite = dump_freq;

      DBG("Dumping iteration " << i+1 << "/" << max_ite)
    }
  }
  time_point end = now();

  duration elapsed = end-start;
  std::cerr << max_ite << " iterations computed in " << elapsed.count() << "s" << std::endl;
  dump_data(dump_counter);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    parse_args(argc, argv);
    load_ics();
    run_sim();
  }
  Kokkos::finalize();
  return 0;
}

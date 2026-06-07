#include "rem_fd.h"

#if defined(CUDA3D_PML_ZMEM_IN_P) && !defined(CUDA3D_PML_RECOMPUTE_Z)
#error "CUDA3D_PML_ZMEM_IN_P requires CUDA3D_PML_RECOMPUTE_Z"
#endif

#if defined(CUDA3D_PML_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_INTERIOR_COMPARE)
#include <sys/stat.h>
#endif

void check_gpu_error_2(const char *msg){  
  cudaError_t err = cudaGetLastError ();
  if (cudaSuccess !=err){ 
    printf("Cuda error: %s: %s", msg, cudaGetErrorString(err));
    exit(0); 
  }
} 

#ifdef CUDA3D_DEBUG_CHECKS
#define check_gpu_error_loop(msg) check_gpu_error_2(msg)
#else
#define check_gpu_error_loop(msg) ((void)0)
#endif

#if defined(CUDA3D_PML_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_INTERIOR_COMPARE)
static void dump_device_float_array(const char *dump_dir, const char *name,
				    int mytid, int snum, int it,
				    const float *dptr, size_t count) {
  if (dump_dir == NULL || dump_dir[0] == '\0') return;

  float *host = (float*)malloc(count * sizeof(float));
  if (host == NULL) {
    printf("ERROR allocating debug dump host buffer for %s\n", name);
    exit(0);
  }

  cudaMemcpy(host, dptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  check_gpu_error_2("copy debug dump array");

  char path[1024];
  snprintf(path, sizeof(path), "%s/rank_%d_shot_%d_it_%d_%s.bin",
	   dump_dir, mytid, snum, it, name);
  FILE *fp = fopen(path, "wb");
  if (fp == NULL) {
    printf("ERROR opening debug dump file %s\n", path);
    free(host);
    exit(0);
  }
  fwrite(host, sizeof(float), count, fp);
  fclose(fp);
  free(host);
}
#endif

#ifdef CUDA3D_PML_DEBUG_DUMP
static void dump_pml_debug_state(const char *dump_dir,
				 int mytid, int snum, int it,
				 int nby, int nbx, int nbz,
				 int nypad, int nxpad, int nzpad,
				 int nbd,
				 float *d_p0, float *d_p1,
				 float *d_vy, float *d_vx, float *d_vz,
				 float *d_memory_dy, float *d_memory_dx, float *d_memory_dz,
				 float *d_memory_dyy, float *d_memory_dxx, float *d_memory_dzz) {
  if (dump_dir == NULL || dump_dir[0] == '\0') return;

  mkdir(dump_dir, 0775);

  const size_t nxyzpad_debug = (size_t)nypad * (size_t)nxpad * (size_t)nzpad;
  const size_t mem_y_count = (size_t)2 * (size_t)nbd * (size_t)nbx * (size_t)nbz;
  const size_t mem_x_count = (size_t)nby * (size_t)2 * (size_t)nbd * (size_t)nbz;
  const size_t mem_z_count = (size_t)nby * (size_t)nbx * (size_t)2 * (size_t)nbd;

  char meta_path[1024];
  snprintf(meta_path, sizeof(meta_path), "%s/rank_%d_shot_%d_it_%d_meta.txt",
	   dump_dir, mytid, snum, it);
  FILE *meta = fopen(meta_path, "w");
  if (meta == NULL) {
    printf("ERROR opening debug dump meta file %s\n", meta_path);
    exit(0);
  }
  fprintf(meta, "mytid=%d\nsnum=%d\nit=%d\n", mytid, snum, it);
  fprintf(meta, "nby=%d\nnbx=%d\nnbz=%d\n", nby, nbx, nbz);
  fprintf(meta, "nypad=%d\nnxpad=%d\nnzpad=%d\nnbd=%d\n", nypad, nxpad, nzpad, nbd);
  fprintf(meta, "nxyzpad=%zu\nmem_y_count=%zu\nmem_x_count=%zu\nmem_z_count=%zu\n",
	  nxyzpad_debug, mem_y_count, mem_x_count, mem_z_count);
  fclose(meta);

  dump_device_float_array(dump_dir, "p0", mytid, snum, it, d_p0, nxyzpad_debug);
  dump_device_float_array(dump_dir, "p1", mytid, snum, it, d_p1, nxyzpad_debug);
  dump_device_float_array(dump_dir, "vy", mytid, snum, it, d_vy, nxyzpad_debug);
  dump_device_float_array(dump_dir, "vx", mytid, snum, it, d_vx, nxyzpad_debug);
  dump_device_float_array(dump_dir, "vz", mytid, snum, it, d_vz, nxyzpad_debug);
  dump_device_float_array(dump_dir, "memory_dy", mytid, snum, it, d_memory_dy, mem_y_count);
  dump_device_float_array(dump_dir, "memory_dx", mytid, snum, it, d_memory_dx, mem_x_count);
  dump_device_float_array(dump_dir, "memory_dz", mytid, snum, it, d_memory_dz, mem_z_count);
  dump_device_float_array(dump_dir, "memory_dyy", mytid, snum, it, d_memory_dyy, mem_y_count);
  dump_device_float_array(dump_dir, "memory_dxx", mytid, snum, it, d_memory_dxx, mem_x_count);
  dump_device_float_array(dump_dir, "memory_dzz", mytid, snum, it, d_memory_dzz, mem_z_count);
}
#endif

#if defined(CUDA3D_CORE_2STEP_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_INTERIOR_COMPARE)
static int parse_core2step_region(const char *text,
				  int *z0, int *z1, int *x0, int *x1, int *y0, int *y1) {
  if (text == NULL || text[0] == '\0') return 0;
  if (sscanf(text, "%d:%d,%d:%d,%d:%d", z0, z1, x0, x1, y0, y1) == 6) return 1;
  if (sscanf(text, "%d,%d,%d,%d,%d,%d", z0, z1, x0, x1, y0, y1) == 6) return 1;
  return 0;
}

static int point_in_core2step_region(int z, int x, int y,
				     int z0, int z1, int x0, int x1, int y0, int y1) {
  return z >= z0 && z < z1 && x >= x0 && x < x1 && y >= y0 && y < y1;
}

static void dump_core2step_region_array(const char *dump_dir, const char *name,
					int mytid, int snum, int it,
					const float *dptr,
					int nypad, int nxpad, int nzpad,
					int z0, int z1, int x0, int x1, int y0, int y1) {
  const int nzr = z1 - z0;
  const int nxr = x1 - x0;
  const int nyr = y1 - y0;
  const size_t count = (size_t)nyr * (size_t)nxr * (size_t)nzr;
  const size_t nxyzpad_debug = (size_t)nypad * (size_t)nxpad * (size_t)nzpad;

  float *host = (float*)malloc(nxyzpad_debug * sizeof(float));
  float *region = (float*)malloc(count * sizeof(float));
  if (host == NULL || region == NULL) {
    printf("ERROR allocating core 2step debug buffers for %s\n", name);
    free(host);
    free(region);
    exit(0);
  }

  cudaMemcpy(host, dptr, nxyzpad_debug * sizeof(float), cudaMemcpyDeviceToHost);
  check_gpu_error_2("copy core 2step debug array");

  size_t out = 0;
  for (int y = y0; y < y1; ++y) {
    const size_t yoff = (size_t)(y + radius) * (size_t)nxpad * (size_t)nzpad;
    for (int x = x0; x < x1; ++x) {
      const size_t xoff = (size_t)(x + radius) * (size_t)nzpad;
      for (int z = z0; z < z1; ++z) {
	region[out++] = host[yoff + xoff + (size_t)(z + radius)];
      }
    }
  }

  char path[1024];
  snprintf(path, sizeof(path), "%s/rank_%d_shot_%d_it_%d_%s_core.bin",
	   dump_dir, mytid, snum, it, name);
  FILE *fp = fopen(path, "wb");
  if (fp == NULL) {
    printf("ERROR opening core 2step dump file %s\n", path);
    free(host);
    free(region);
    exit(0);
  }
  fwrite(region, sizeof(float), count, fp);
  fclose(fp);
  free(host);
  free(region);
}

static void dump_core2step_debug_state(const char *dump_dir,
				       int mytid, int snum, int it,
				       int nby, int nbx, int nbz,
				       int nypad, int nxpad, int nzpad,
				       int nbd,
				       int yl, int xl,
				       int indxy, int indxx, int indxz,
				       int **rec0_indx, size_t nr,
				       float *d_p0, float *d_p1) {
  if (dump_dir == NULL || dump_dir[0] == '\0') return;

  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != 1) {
    printf("ERROR CUDA3D_CORE_2STEP debug dump is single-GPU/single-rank only; MPI size=%d\n", mpi_size);
    exit(0);
  }

  mkdir(dump_dir, 0775);

  int margin = 2 * CUDA3D_CORE_STENCIL_RADIUS;
  const char *margin_env = getenv("CUDA3D_CORE_2STEP_MARGIN");
  if (margin_env != NULL && margin_env[0] != '\0') margin = atoi(margin_env);

  const int core_z0 = nbd + CorePmlMargin;
  const int core_x0 = nbd + CorePmlMargin;
  const int core_y0 = nbd + CorePmlMargin;
  const int core_z1 = nbz - nbd - CorePmlMargin;
  const int core_x1 = nbx - nbd - CorePmlMargin;
  const int core_y1 = nby - nbd - CorePmlMargin;
  int z0 = core_z0 + margin;
  int z1 = core_z1 - margin;
  int x0 = core_x0 + margin;
  int x1 = core_x1 - margin;
  int y0 = core_y0 + margin;
  int y1 = core_y1 - margin;

  const char *region_env = getenv("CUDA3D_CORE_2STEP_REGION");
  if (parse_core2step_region(region_env, &z0, &z1, &x0, &x1, &y0, &y1)) {
    margin = -1;
  }

  if (z0 < 0 || x0 < 0 || y0 < 0 || z1 > nbz || x1 > nbx || y1 > nby ||
      z0 >= z1 || x0 >= x1 || y0 >= y1) {
    printf("ERROR invalid CUDA3D_CORE_2STEP debug region z=[%d,%d) x=[%d,%d) y=[%d,%d), n=(%d,%d,%d)\n",
	   z0, z1, x0, x1, y0, y1, nbz, nbx, nby);
    exit(0);
  }

  const int source_z_local = nbd + indxz;
  const int source_x_local = nbd + indxx - xl;
  const int source_y_local = nbd + indxy - yl;
  const int source_in_region = point_in_core2step_region(source_z_local, source_x_local, source_y_local,
							 z0, z1, x0, x1, y0, y1);
  size_t receivers_in_region = 0;
  for (size_t ir = 0; ir < nr; ++ir) {
    const int rz = nbd + rec0_indx[ir][2];
    const int rx = nbd + rec0_indx[ir][1] - xl;
    const int ry = nbd + rec0_indx[ir][0] - yl;
    if (point_in_core2step_region(rz, rx, ry, z0, z1, x0, x1, y0, y1)) ++receivers_in_region;
  }

  char meta_path[1024];
  snprintf(meta_path, sizeof(meta_path), "%s/rank_%d_shot_%d_it_%d_core_meta.txt",
	   dump_dir, mytid, snum, it);
  FILE *meta = fopen(meta_path, "w");
  if (meta == NULL) {
    printf("ERROR opening core 2step meta file %s\n", meta_path);
    exit(0);
  }
  fprintf(meta, "mytid=%d\nsnum=%d\nit=%d\nstage=post_inject_pre_swap\n", mytid, snum, it);
  fprintf(meta, "nby=%d\nnbx=%d\nnbz=%d\nnypad=%d\nnxpad=%d\nnzpad=%d\nnbd=%d\n",
	  nby, nbx, nbz, nypad, nxpad, nzpad, nbd);
  fprintf(meta, "radius=%d\ncore_stencil_radius=%d\ncore_pml_margin=%d\n",
	  radius, CUDA3D_CORE_STENCIL_RADIUS, CorePmlMargin);
  fprintf(meta, "core_z0=%d\ncore_z1=%d\ncore_x0=%d\ncore_x1=%d\ncore_y0=%d\ncore_y1=%d\n",
	  core_z0, core_z1, core_x0, core_x1, core_y0, core_y1);
  fprintf(meta, "margin=%d\nz0=%d\nz1=%d\nx0=%d\nx1=%d\ny0=%d\ny1=%d\n",
	  margin, z0, z1, x0, x1, y0, y1);
  fprintf(meta, "count=%zu\n", (size_t)(z1 - z0) * (size_t)(x1 - x0) * (size_t)(y1 - y0));
  fprintf(meta, "yl=%d\nxl=%d\nsource_z_raw=%d\nsource_x_raw=%d\nsource_y_raw=%d\n",
	  yl, xl, indxz, indxx, indxy);
  fprintf(meta, "source_z=%d\nsource_x=%d\nsource_y=%d\n",
	  source_z_local, source_x_local, source_y_local);
  fprintf(meta, "source_in_region=%d\nreceiver_count=%zu\nreceivers_in_region=%zu\n",
	  source_in_region, nr, receivers_in_region);
  fclose(meta);

  dump_core2step_region_array(dump_dir, "p0", mytid, snum, it, d_p0,
			      nypad, nxpad, nzpad, z0, z1, x0, x1, y0, y1);
  dump_core2step_region_array(dump_dir, "p1", mytid, snum, it, d_p1,
			      nypad, nxpad, nzpad, z0, z1, x0, x1, y0, y1);
}
#endif

#if defined(CUDA3D_PML_ZMEM_IN_P) && defined(CUDA3D_PML_ZMEM_DEBUG_FILL)
static void check_zmem_new_written(float *d_memory_dz_next, size_t count,
				   int mytid, int snum, int it) {
  float *host = (float*)malloc(count * sizeof(float));
  if (host == NULL) {
    printf("ERROR allocating ZMEM debug host buffer\n");
    exit(0);
  }

  cudaMemcpy(host, d_memory_dz_next, count * sizeof(float), cudaMemcpyDeviceToHost);
  check_gpu_error_2("copy ZMEM debug buffer");

  size_t unwritten = 0;
  size_t first_unwritten = 0;
  for (size_t i = 0; i < count; ++i) {
    if (host[i] != host[i]) {
      if (unwritten == 0) first_unwritten = i;
      ++unwritten;
    }
  }
  free(host);

  if (unwritten != 0) {
    printf("ERROR ZMEM_IN_P unwritten entries: rank=%d shot=%d it=%d count=%zu first=%zu\n",
	   mytid, snum, it, unwritten, first_unwritten);
    exit(0);
  }
}
#endif

#ifdef CUDA3D_PML_TILE_LIST
#ifndef CUDA3D_PML_TILE_LIST_V
#define CUDA3D_PML_TILE_LIST_V
#endif
#ifndef CUDA3D_PML_TILE_LIST_P
#define CUDA3D_PML_TILE_LIST_P
#endif
#endif

#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
#ifndef CUDA3D_PML_TILE_LIST_P
#define CUDA3D_PML_TILE_LIST_P
#endif
#endif

#if defined(CUDA3D_PML_TILE_LIST_V) || defined(CUDA3D_PML_TILE_LIST_P)
static int ceil_div_int(int n, int d) {
  return (n + d - 1) / d;
}

static int min_int_local(int a, int b) {
  return a < b ? a : b;
}

static int tile_fully_inside_box(int z0, int x0, int y0,
				 int bz, int bx, int by,
				 int n1, int n2, int n3,
				 int zlo, int zhi,
				 int xlo, int xhi,
				 int ylo, int yhi) {
  const int z1 = min_int_local(z0 + bz, n1);
  const int x1 = min_int_local(x0 + bx, n2);
  const int y1 = min_int_local(y0 + by, n3);
  return z0 >= zlo && z1 <= zhi &&
    x0 >= xlo && x1 <= xhi &&
    y0 >= ylo && y1 <= yhi;
}

static int tile_may_need_component(int z0, int x0, int y0,
				   int bz, int bx, int by,
				   int n1, int n2, int n3,
				   int zlo, int zhi,
				   int xlo, int xhi,
				   int ylo, int yhi) {
  if (zhi <= zlo || xhi <= xlo || yhi <= ylo) return 1;
  return !tile_fully_inside_box(z0, x0, y0,
				bz, bx, by,
				n1, n2, n3,
				zlo, zhi,
				xlo, xhi,
				ylo, yhi);
}

static unsigned int make_pml_tile_mask(int z0, int x0, int y0,
				       int bz, int bx, int by,
				       int n1, int n2, int n3,
				       int npml) {
  const int z1 = min_int_local(z0 + bz, n1);
  const int x1 = min_int_local(x0 + bx, n2);
  const int y1 = min_int_local(y0 + by, n3);
  const int z_active = (z0 < npml) || (z1 > n1 - npml);
  const int x_active = (x0 < npml) || (x1 > n2 - npml);
  const int y_active = (y0 < npml) || (y1 > n3 - npml);
  unsigned int mask = 0u;
  int axes = 0;
  if (z_active) { mask |= PML_TILE_MASK_Z; ++axes; }
  if (x_active) { mask |= PML_TILE_MASK_X; ++axes; }
  if (y_active) { mask |= PML_TILE_MASK_Y; ++axes; }
  if (axes > 1) mask |= PML_TILE_MASK_MIXED;
  return mask;
}

static int build_pml_tile_list(PmlTile **tiles_out,
			       int n3, int n2, int n1,
			       int npml, int for_velocity) {
  const int grid1 = ceil_div_int(n1, PmlTileBlockSize1);
  const int grid2 = ceil_div_int(n2, PmlTileBlockSize2);
  const int grid3 = ceil_div_int(n3, PmlTileBlockSize3);
  const int max_tiles = grid1 * grid2 * grid3;
  PmlTile *tiles = (PmlTile*)malloc((size_t)max_tiles * sizeof(PmlTile));
  if (tiles == NULL) {
    printf("ERROR allocating PML tile list\n");
    exit(0);
  }

  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  const int skip1_lo = for_velocity ? core1_lo + 3 : core1_lo;
  const int skip2_lo = for_velocity ? core2_lo + 3 : core2_lo;
  const int skip3_lo = for_velocity ? core3_lo + 3 : core3_lo;
  const int skip1_hi = for_velocity ? core1_hi - 4 : core1_hi;
  const int skip2_hi = for_velocity ? core2_hi - 4 : core2_hi;
  const int skip3_hi = for_velocity ? core3_hi - 4 : core3_hi;

  int ntile = 0;
  int original_ntile = 0;
  int mask_counts[16] = {0};
  for (int by = 0; by < grid3; ++by) {
    const int y0 = by * PmlTileBlockSize3;
    for (int bx = 0; bx < grid2; ++bx) {
      const int x0 = bx * PmlTileBlockSize2;
      for (int bz = 0; bz < grid1; ++bz) {
	const int z0 = bz * PmlTileBlockSize1;
	const int old_skip = (skip1_hi > skip1_lo &&
			      skip2_hi > skip2_lo &&
			      skip3_hi > skip3_lo &&
			      tile_fully_inside_box(z0, x0, y0,
						    PmlTileBlockSize1,
						    PmlTileBlockSize2,
						    PmlTileBlockSize3,
						    n1, n2, n3,
						    skip1_lo, skip1_hi,
						    skip2_lo, skip2_hi,
						    skip3_lo, skip3_hi));
	int skip = old_skip;
	if (!old_skip) ++original_ntile;
#if defined(CUDA3D_PML_ZMEM_IN_P) && defined(CUDA3D_PML_ZMEM_V_TILE_PRUNE)
	if (for_velocity) {
	  const int may_need_vz = 0;
	  const int may_need_vx =
	    tile_may_need_component(z0, x0, y0,
				    PmlTileBlockSize1,
				    PmlTileBlockSize2,
				    PmlTileBlockSize3,
				    n1, n2, n3,
				    core1_lo, core1_hi,
				    core2_lo + 3, core2_hi - 4,
				    core3_lo, core3_hi);
	  const int may_need_vy =
	    tile_may_need_component(z0, x0, y0,
				    PmlTileBlockSize1,
				    PmlTileBlockSize2,
				    PmlTileBlockSize3,
				    n1, n2, n3,
				    core1_lo, core1_hi,
				    core2_lo, core2_hi,
				    core3_lo + 3, core3_hi - 4);
	  skip = !(may_need_vz || may_need_vx || may_need_vy);
	}
#endif
	if (!skip) {
	  const unsigned int mask = make_pml_tile_mask(z0, x0, y0,
						      PmlTileBlockSize1,
						      PmlTileBlockSize2,
						      PmlTileBlockSize3,
						      n1, n2, n3, npml);
	  tiles[ntile].z0 = z0;
	  tiles[ntile].x0 = x0;
	  tiles[ntile].y0 = y0;
	  tiles[ntile].mask = mask;
	  ++mask_counts[mask & 15u];
	  ++ntile;
	}
      }
    }
  }

#if defined(CUDA3D_PML_ZMEM_IN_P) && defined(CUDA3D_PML_ZMEM_V_TILE_PRUNE)
  if (for_velocity) {
    const int pruned = original_ntile - ntile;
    const double ratio = original_ntile > 0 ? (double)pruned / (double)original_ntile : 0.0;
    printf("PML V tile prune: original=%d kept=%d pruned=%d prune_ratio=%.6f block=%dx%dx%d\n",
	   original_ntile, ntile, pruned, ratio,
	   PmlTileBlockSize1, PmlTileBlockSize2, PmlTileBlockSize3);
  }
#endif
#ifdef CUDA3D_PML_TILE_MASK_FASTPATH
  printf("PML tile mask stats: kind=%s total=%d none=%d z=%d x=%d y=%d mixed=%d block=%dx%dx%d\n",
	 for_velocity ? "v" : "p",
	 ntile,
	 mask_counts[0],
	 mask_counts[PML_TILE_MASK_Z],
	 mask_counts[PML_TILE_MASK_X],
	 mask_counts[PML_TILE_MASK_Y],
	 ntile - mask_counts[0] - mask_counts[PML_TILE_MASK_Z] - mask_counts[PML_TILE_MASK_X] - mask_counts[PML_TILE_MASK_Y],
	 PmlTileBlockSize1, PmlTileBlockSize2, PmlTileBlockSize3);
#endif
  *tiles_out = tiles;
  return ntile;
}

#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
static int build_pml_zface_tile_list(PmlTile **tiles_out,
				      int n3, int n2, int n1,
				      int npml) {
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  const int core_n2 = core2_hi - core2_lo;
  const int core_n3 = core3_hi - core3_lo;
  if (npml <= 0 || core_n2 <= 0 || core_n3 <= 0) {
    *tiles_out = NULL;
    return 0;
  }

  const int grid1 = ceil_div_int(npml, PmlZFaceBlockSize1);
  const int grid2 = ceil_div_int(core_n2, PmlZFaceBlockSize2);
  const int grid3 = ceil_div_int(core_n3, PmlZFaceBlockSize3);
  const int max_tiles = 2 * grid1 * grid2 * grid3;
  PmlTile *tiles = (PmlTile*)malloc((size_t)max_tiles * sizeof(PmlTile));
  if (tiles == NULL) {
    printf("ERROR allocating zface PML tile list\n");
    exit(0);
  }

  int ntile = 0;
  for (int by = 0; by < grid3; ++by) {
    const int y0 = core3_lo + by * PmlZFaceBlockSize3;
    for (int bx = 0; bx < grid2; ++bx) {
      const int x0 = core2_lo + bx * PmlZFaceBlockSize2;
      for (int bz = 0; bz < grid1; ++bz) {
	const int z0_lo = bz * PmlZFaceBlockSize1;
	tiles[ntile].z0 = z0_lo;
	tiles[ntile].x0 = x0;
	tiles[ntile].y0 = y0;
	tiles[ntile].mask = PML_TILE_MASK_Z;
	++ntile;

	const int z0_hi = n1 - npml + bz * PmlZFaceBlockSize1;
	tiles[ntile].z0 = z0_hi;
	tiles[ntile].x0 = x0;
	tiles[ntile].y0 = y0;
	tiles[ntile].mask = PML_TILE_MASK_Z;
	++ntile;
      }
    }
  }

  *tiles_out = tiles;
  return ntile;
}
#endif
#endif

void fd_3d_f(float *src, float bscl, float ***cw2, float **h_est,
	     int ny, int nx, int nz, float dy, float dx, float dz,
	     int nt, float dt, int ntw, int ntj, int *vek,
	     int **src0_indx, int **rec0_indx,
	     size_t nr, size_t icc, int ns, int snum, int yl, int xl,
	     float *sw000, float *sw001, float *sw010, float *sw011,
	     float *sw100, float *sw101, float *sw110, float *sw111,
	     float *rw000, float *rw001, float *rw010, float *rw011,
	     float *rw100, float *rw101, float *rw110, float *rw111,
	     float *bt, float *bb, float *bl, float *br, int nbd, float vmax,
	     float *ay, float *by, float *ax, float *bx, float *az, float *bz,
	     float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
	     int mytid, char *order){

  int iy, ix, iz, it, itc, nby, nbx, nbz, nbell=1;
#ifdef CUDA3D_PML_DEBUG_DUMP
  const char *pml_dump_dir = getenv("CUDA3D_PML_DUMP_DIR");
  int pml_dump_step = 0;
  const char *pml_dump_step_env = getenv("CUDA3D_PML_DUMP_STEP");
  if (pml_dump_step_env != NULL && pml_dump_step_env[0] != '\0')
    pml_dump_step = atoi(pml_dump_step_env);
#endif
#if defined(CUDA3D_CORE_2STEP_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_INTERIOR_COMPARE)
  const char *core2step_dump_dir = getenv("CUDA3D_CORE_2STEP_DUMP_DIR");
  int core2step_dump_step = -1;
  const char *core2step_dump_step_env = getenv("CUDA3D_CORE_2STEP_DUMP_STEP");
  if (core2step_dump_step_env != NULL && core2step_dump_step_env[0] != '\0')
    core2step_dump_step = atoi(core2step_dump_step_env);
#endif
  int nypad, nxpad, nzpad;
  float ss, temp, tdy, tdx, tdz, dt2;
  float *h_bell;
  size_t nxyz, byte, ir, nxyzpad;

  // device variables
  int indxx, indxy, indxz;
  int *d_src0_indx, *d_rec0_indx;
  float *d_bell, *d_src, *d_cw2, *d_est;
  float *d_sw000, *d_sw001, *d_sw010, *d_sw011, *d_sw100, *d_sw101, *d_sw110, *d_sw111;
  float *d_rw000, *d_rw001, *d_rw010, *d_rw011, *d_rw100, *d_rw101, *d_rw110, *d_rw111;
  float *d_memory_dy, *d_memory_dx, *d_memory_dz;
#ifdef CUDA3D_PML_ZMEM_IN_P
  float *d_memory_dz_next = NULL;
#endif
  float *d_memory_dyy, *d_memory_dxx, *d_memory_dzz;

  //wavefields
  float *d_p0, *d_p1, *ptr, *d_vx, *d_vy, *d_vz;
  // pml
  float *d_ax=NULL, *d_bx=NULL, *d_ay=NULL, *d_by=NULL, *d_az=NULL, *d_bz=NULL;
  float *d_ax_h=NULL, *d_bx_h=NULL, *d_ay_h=NULL, *d_by_h=NULL, *d_az_h=NULL, *d_bz_h=NULL;
#if defined(CUDA3D_PML_TILE_LIST_V) || defined(CUDA3D_PML_TILE_LIST_P)
  PmlTile *h_v_pml_tiles = NULL, *h_p_pml_tiles = NULL;
  PmlTile *d_v_pml_tiles = NULL, *d_p_pml_tiles = NULL;
  int n_v_pml_tiles = 0, n_p_pml_tiles = 0;
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  PmlTile *h_zface_p_pml_tiles = NULL, *d_zface_p_pml_tiles = NULL;
  int n_zface_p_pml_tiles = 0;
#endif

  // src y, x, and z location index

  indxy=src0_indx[snum][0];
  indxx=src0_indx[snum][1];
  indxz=src0_indx[snum][2];
  if(mytid==0)
    printf(" indxy=%d, indxx=%d, indxz=%d\n", indxy, indxx, indxz);

  nby=ny+2*(nbd);
  nbx=nx+2*(nbd);
  nbz=nz+2*(nbd);
  // need radius!!
  nypad=ny+2*(nbd+radius);
  nxpad=nx+2*(nbd+radius);
  nzpad=nz+2*(nbd+radius);

  nxyz=nbx*nbz*nby;
  byte=sizeof(float)*nxyz;
  nxyzpad=nypad*nxpad*nzpad;
#ifdef CUDA3D_PML_ZMEM_IN_P
  const size_t mem_z_count = (size_t)nby * (size_t)nbx * (size_t)2 * (size_t)nbd;
  const size_t mem_z_bytes = mem_z_count * sizeof(float);
#endif
  //  if(mytid==3)
  //    printf("id=%d nzpad, nxpad, nzpad= %d, %d, %d, shot#=%d, ns=%d, nr=%d, nsize=%zu\n", 
  //	   mytid, nypad, nxpad, nzpad, snum, ns, nr, nxyzpad);
  //  printf("lap nbx=%d nby=%d nbz=%d, ny=%d nx=%d nz=%d nbd=%d\n",nbx, nby, nbz, ny, nx, nz, nbd);

  dt2=dt*dt;
  tdy=1./dy;
  tdx=1./dx;
  tdz=1./dz;

  //-------------------- setup bell function ------------------------
  h_bell=alloc1float((2*nbell+1)*(2*nbell+1)*(2*nbell+1));
  ss=0.5*nbell;
  for(iy=-nbell; iy<=nbell; iy++)
    for(ix=-nbell; ix<=nbell; ix++)
      for(iz=-nbell; iz<=nbell; iz++)
	h_bell[(nbell+iy)*(2*nbell+1)*(2*nbell+1)+(nbell+ix)*(2*nbell+1)+nbell+iz]=exp(-bscl*bscl*(iy*iy+iz*iz+ix*ix)/ss);

  cudaMalloc((void**)&d_bell, (2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float));
  cudaMemcpy(d_bell, h_bell, (2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float), cudaMemcpyHostToDevice);

  // -------------- copy wavelet and velocity to device------------------------
  cudaMalloc((void**)&d_src, nt*sizeof(float));
  cudaMemcpy(d_src, &src[0], nt*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_cw2, nxyzpad*sizeof(float));
  cudaMemcpy(d_cw2, &cw2[0][0][0], nxyzpad*sizeof(float), cudaMemcpyHostToDevice);
  fflush(stdout);
  // ----------------initialize src and rec index and interpolation parameters----------------------
  cudaMalloc((void**)&d_sw000, ns*sizeof(float));   cudaMalloc((void**)&d_sw001, ns*sizeof(float));
  cudaMalloc((void**)&d_sw010, ns*sizeof(float));   cudaMalloc((void**)&d_sw011, ns*sizeof(float));
  cudaMalloc((void**)&d_sw100, ns*sizeof(float));   cudaMalloc((void**)&d_sw101, ns*sizeof(float));
  cudaMalloc((void**)&d_sw110, ns*sizeof(float));   cudaMalloc((void**)&d_sw111, ns*sizeof(float));
  cudaMemcpy(d_sw000, &sw000[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw001, &sw001[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw010, &sw010[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw011, &sw011[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw100, &sw100[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw101, &sw101[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw110, &sw110[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw111, &sw111[0], ns*sizeof(float), cudaMemcpyHostToDevice);

  // receivers
  cudaMalloc((void**)&d_rw000, nr*sizeof(float));   cudaMalloc((void**)&d_rw001, nr*sizeof(float));
  cudaMalloc((void**)&d_rw010, nr*sizeof(float));   cudaMalloc((void**)&d_rw011, nr*sizeof(float));
  cudaMalloc((void**)&d_rw100, nr*sizeof(float));   cudaMalloc((void**)&d_rw101, nr*sizeof(float));
  cudaMalloc((void**)&d_rw110, nr*sizeof(float));   cudaMalloc((void**)&d_rw111, nr*sizeof(float));
  cudaMemcpy(d_rw000, &rw000[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw001, &rw001[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw010, &rw010[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw011, &rw011[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw100, &rw100[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw101, &rw101[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw110, &rw110[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw111, &rw111[icc], nr*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_src0_indx, 3*ns*sizeof(int));
  cudaMalloc((void**)&d_rec0_indx, 3*nr*sizeof(int));
  cudaMemcpy(d_src0_indx, &src0_indx[0], 3*ns*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rec0_indx, &rec0_indx[icc][0], 3*nr*sizeof(int), cudaMemcpyHostToDevice);

  // -------------------initialize wavefields and data----------------------
  cudaMalloc((void**)&d_est, nt*nr*sizeof(float));
  cudaMemset(d_est, 0., nt*nr*sizeof(float));  // note ir is the fast direction in 3D
  check_gpu_error_2("Error in Memset");
 
  cudaMalloc((void**)&d_p0, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_p1, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_vy, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_vx, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_vz, nxyzpad*sizeof(float));
  cudaMemset(d_p0, 0, nxyzpad*sizeof(float));
  cudaMemset(d_p1, 0, nxyzpad*sizeof(float));
  cudaMemset(d_vy, 0, nxyzpad*sizeof(float));
  cudaMemset(d_vx, 0, nxyzpad*sizeof(float));
  cudaMemset(d_vz, 0, nxyzpad*sizeof(float));
  check_gpu_error_2("Error in Memset");

  // ----------------------------initialize pml arrays-------------------------------
  if (nbd > CUDA3D_MAX_PML) {
    printf("ERROR nbd=%d exceeds CUDA3D_MAX_PML=%d\n", nbd, CUDA3D_MAX_PML);
    exit(0);
  }
  upload_pml_constants(nbd, ay, by, ax, bx, az, bz, ay_h, by_h, ax_h, bx_h, az_h, bz_h);
  check_gpu_error_2("Error copying PML constants");

  // ----------------- initialize memeory varibles pml note the order of nbd nbx when used-----------
  cudaMalloc((void**)&d_memory_dy, 2*nbd*nbx*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dx, nby*2*nbd*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dz, nby*nbx*2*nbd*sizeof(float));
#ifdef CUDA3D_PML_ZMEM_IN_P
  cudaMalloc((void**)&d_memory_dz_next, mem_z_bytes);
#endif
  cudaMalloc((void**)&d_memory_dyy, 2*nbd*nbx*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dxx, nby*2*nbd*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dzz, nby*nbx*2*nbd*sizeof(float));

  cudaMemset(d_memory_dy, 0., 2*nbd*nbx*nbz*sizeof(float));
  cudaMemset(d_memory_dx, 0., nby*2*nbd*nbz*sizeof(float));
  cudaMemset(d_memory_dz, 0., nby*nbx*2*nbd*sizeof(float));
#ifdef CUDA3D_PML_ZMEM_IN_P
  cudaMemset(d_memory_dz_next, 0, mem_z_bytes);
#endif
  cudaMemset(d_memory_dyy, 0., 2*nbd*nbx*nbz*sizeof(float));
  cudaMemset(d_memory_dxx, 0., nby*2*nbd*nbz*sizeof(float));
  cudaMemset(d_memory_dzz, 0., nby*nbx*2*nbd*sizeof(float));

  check_gpu_error_2("Error in Memset");

  dim3 dimg_v, dimb_v, dimg_p, dimb_p, dimg_pml, dimb_pml, dimg_pml_zface, dimb_pml_zface, dims, dimbs, dimr, dimbr;// dims(1,1), dimbs(2*nbell+1, 2*nbell+1);
  dims.x=1;
  dims.y=1;
  dims.z=1;
  dimbs.x=2*nbell+1;
  dimbs.y=2*nbell+1;
  dimbs.z=2*nbell+1;

  //  dimbs.x=1;
  //  dimbs.y=1;
  int BS=1024;
  dimr.x=((nr+BS-1)/BS); // 1 
  dimr.y=1;
  dimbr.x=BS; //nr // need check!!!!!!
  dimbr.y=1;


  //  dimg.x=(int)((nbz+BlockSize1-1)/BlockSize1);
  //  dimg.y=(int)((nbx+BlockSize2-1)/BlockSize2);
  //  dimb.x=BlockSize1;
  //  dimb.y=BlockSize2;

  dimg_v.x=(int)((nbz+VBlockSize1-1)/VBlockSize1);
  dimg_v.y=(int)((nbx+VBlockSize2-1)/VBlockSize2);
  dimg_v.z=(int)((nby+VBlockSize3-1)/VBlockSize3);
  dimb_v.x=VBlockSize1;
  dimb_v.y=VBlockSize2;
  dimb_v.z=VBlockSize3;

  const int core_z0 = nbd + CorePmlMargin;
  const int core_x0 = nbd + CorePmlMargin;
  const int core_y0 = nbd + CorePmlMargin;
  const int core_z1 = nbz - nbd - CorePmlMargin;
  const int core_x1 = nbx - nbd - CorePmlMargin;
  const int core_y1 = nby - nbd - CorePmlMargin;
  const int core_nz = core_z1 - core_z0;
  const int core_nx = core_x1 - core_x0;
  const int core_ny = core_y1 - core_y0;
  dimg_p.x=(int)((core_nz+PBlockSize1-1)/PBlockSize1);
  dimg_p.y=(int)((core_nx+PBlockSize2-1)/PBlockSize2);
  dimg_p.z=(int)((core_ny+PBlockSize3-1)/PBlockSize3);
  dimb_p.x=PBlockSize1;
  dimb_p.y=PBlockSize2;
  dimb_p.z=PBlockSize3;

  dimg_pml.x=(int)((nbz+PmlBlockSize1-1)/PmlBlockSize1);
  dimg_pml.y=(int)((nbx+PmlBlockSize2-1)/PmlBlockSize2);
  dimg_pml.z=(int)((nby+PmlBlockSize3-1)/PmlBlockSize3);
  dimb_pml.x=PmlBlockSize1;
  dimb_pml.y=PmlBlockSize2;
  dimb_pml.z=PmlBlockSize3;

#if defined(CUDA3D_PML_TILE_LIST_V) || defined(CUDA3D_PML_TILE_LIST_P)
#ifdef CUDA3D_PML_TILE_LIST_V
  n_v_pml_tiles = build_pml_tile_list(&h_v_pml_tiles, nby, nbx, nbz, nbd, 1);
  cudaMalloc((void**)&d_v_pml_tiles, (size_t)n_v_pml_tiles * sizeof(PmlTile));
  cudaMemcpy(d_v_pml_tiles, h_v_pml_tiles, (size_t)n_v_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
  n_p_pml_tiles = build_pml_tile_list(&h_p_pml_tiles, nby, nbx, nbz, nbd, 0);
  cudaMalloc((void**)&d_p_pml_tiles, (size_t)n_p_pml_tiles * sizeof(PmlTile));
  cudaMemcpy(d_p_pml_tiles, h_p_pml_tiles, (size_t)n_p_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  n_zface_p_pml_tiles = build_pml_zface_tile_list(&h_zface_p_pml_tiles, nby, nbx, nbz, nbd);
  if (n_zface_p_pml_tiles > 0) {
    cudaMalloc((void**)&d_zface_p_pml_tiles, (size_t)n_zface_p_pml_tiles * sizeof(PmlTile));
    cudaMemcpy(d_zface_p_pml_tiles, h_zface_p_pml_tiles, (size_t)n_zface_p_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
  }
#endif
  check_gpu_error_2("Error building PML tile list");
#ifdef CUDA3D_PML_TILE_LIST_V
  dimb_v.x=PmlTileBlockSize1;
  dimb_v.y=PmlTileBlockSize2;
  dimb_v.z=PmlTileBlockSize3;
  dimg_v.x=n_v_pml_tiles;
  dimg_v.y=1;
  dimg_v.z=1;
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
  dimb_pml.x=PmlTileBlockSize1;
  dimb_pml.y=PmlTileBlockSize2;
  dimb_pml.z=PmlTileBlockSize3;
  dimg_pml.x=n_p_pml_tiles;
  dimg_pml.y=1;
  dimg_pml.z=1;
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  dimb_pml_zface.x=PmlZFaceBlockSize1;
  dimb_pml_zface.y=PmlZFaceBlockSize2;
  dimb_pml_zface.z=PmlZFaceBlockSize3;
  dimg_pml_zface.x=n_zface_p_pml_tiles;
  dimg_pml_zface.y=1;
  dimg_pml_zface.z=1;
#endif
  if(mytid==0)
    printf("PML tile-list enabled: v_tiles=%d, p_tiles=%d, block=%dx%dx%d\n",
	   n_v_pml_tiles, n_p_pml_tiles,
	   PmlTileBlockSize1, PmlTileBlockSize2, PmlTileBlockSize3);
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  if(mytid==0)
    printf("PML zface pressure specialize enabled: zface_p_tiles=%d, block=%dx%dx%d\n",
	   n_zface_p_pml_tiles,
	   PmlZFaceBlockSize1, PmlZFaceBlockSize2, PmlZFaceBlockSize3);
#endif
#endif

  cudaFuncSetCacheConfig(cuda_fd3d_v_pml_ns, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(cuda_fd3d_p_core_ns, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_ns, cudaFuncCachePreferL1);
#ifdef CUDA3D_PML_TILE_LIST_V
  cudaFuncSetCacheConfig(cuda_fd3d_v_pml_tile_ns, cudaFuncCachePreferL1);
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_tile_ns, cudaFuncCachePreferL1);
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_zface_ns, cudaFuncCachePreferL1);
#endif

  //  dim3 dimgu, dimbu;
  //  dimgu.x=(nz+BlockSize1-1)/BlockSize1;
  //  dimgu.y=(nx+BlockSize2-1)/BlockSize2;
  //  dimbu.x=BlockSize1;
  //  dimbu.y=BlockSize2;

  //////
  cudaEvent_t t1, t2, t3, t4;
  float mill;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);
  cudaEventCreate(&t3);
  cudaEventCreate(&t4);

  //////////
  //  float ***out, **out2, **out3;
  //  out=alloc3float(nz, nx, ny);
  //  out2=alloc2float(nz, nx);
  //  out3=alloc2float(nx, ny);
  //  bell=alloc3float(2*nbell+1, 2*nbell+1, 2*nbell+1);
  //  ss=0.5*nbell;

  itc=0;
  fflush(stdout);
  cudaEventRecord(t1,0);

  for(it=0; it<nt; it++){		
    //    fflush(stdout);
    if(it%500==0 && mytid==0)
      printf("FP it=%d\n", it);
    // this is 2nd order time
#ifdef CUDA3D_PML_TILE_LIST_V
    cuda_fd3d_v_pml_tile_ns<<<dimg_v, dimb_v>>>(d_p1, d_vy, d_vx, d_vz,
				    tdy, tdx, tdz,
				    nby, nbx, nbz, nbd, dt,
				    d_ay_h, d_by_h, d_ax_h, d_bx_h, d_az_h, d_bz_h,
				    d_memory_dy, d_memory_dx, d_memory_dz,
				    d_v_pml_tiles, n_v_pml_tiles);
#else
    cuda_fd3d_v_pml_ns<<<dimg_v, dimb_v>>>(d_p1, d_vy, d_vx, d_vz,
				    tdy, tdx, tdz,
				    nby, nbx, nbz, nbd, dt,
				    d_ay_h, d_by_h, d_ax_h, d_bx_h, d_az_h, d_bz_h,
				    d_memory_dy, d_memory_dx, d_memory_dz);
#endif
    check_gpu_error_loop("compute V");
    cuda_fd3d_p_core_ns<<<dimg_p, dimb_p >>>(d_p0, d_p1, d_cw2,
			     tdy, tdx, tdz,
			     nby, nbx, nbz, nbd, dt2);
    check_gpu_error_loop("compute P core");
#if defined(CUDA3D_PML_ZMEM_IN_P) && defined(CUDA3D_PML_ZMEM_DEBUG_FILL)
    cudaMemset(d_memory_dz_next, 0xff, mem_z_bytes);
    check_gpu_error_loop("fill ZMEM next");
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
    if (n_zface_p_pml_tiles > 0) {
      cuda_fd3d_p_pml_zface_ns<<<dimg_pml_zface, dimb_pml_zface >>>(d_p0, d_p1, d_vy, d_vx, d_vz,
				       d_cw2, tdy, tdx, tdz,
				       nby, nbx, nbz, nbd, dt2,
				       d_memory_dzz, d_memory_dz,
				       d_zface_p_pml_tiles, n_zface_p_pml_tiles);
      check_gpu_error_loop("compute P pml zface");
    }
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
    cuda_fd3d_p_pml_tile_ns<<<dimg_pml, dimb_pml >>>(d_p0, d_p1, d_vy, d_vx, d_vz,
				     d_cw2, tdy, tdx, tdz,
				     nby, nbx, nbz, nbd, dt2,
				     d_ay, d_by, d_ax, d_bx, d_az, d_bz,
				     d_memory_dyy, d_memory_dxx, d_memory_dzz,
				     d_memory_dz,
#ifdef CUDA3D_PML_ZMEM_IN_P
				     d_memory_dz_next,
#else
				     d_memory_dz,
#endif
				     d_memory_dx, d_memory_dy,
				     d_p_pml_tiles, n_p_pml_tiles);
#else
    cuda_fd3d_p_pml_ns<<<dimg_pml, dimb_pml >>>(d_p0, d_p1, d_vy, d_vx, d_vz,
				     d_cw2, tdy, tdx, tdz,
				     nby, nbx, nbz, nbd, dt2,
				     d_ay, d_by, d_ax, d_bx, d_az, d_bz,
				     d_memory_dyy, d_memory_dxx, d_memory_dzz,
				     d_memory_dz,
#ifdef CUDA3D_PML_ZMEM_IN_P
				     d_memory_dz_next,
#else
				     d_memory_dz,
#endif
				     d_memory_dx, d_memory_dy);
#endif
    check_gpu_error_loop("compute P pml");
#ifdef CUDA3D_PML_ZMEM_IN_P
#ifdef CUDA3D_PML_ZMEM_DEBUG_FILL
    cudaDeviceSynchronize();
    check_gpu_error_2("sync before ZMEM coverage check");
    check_zmem_new_written(d_memory_dz_next, mem_z_count, mytid, snum, it);
#endif
    {
      float *tmp_dz = d_memory_dz;
      d_memory_dz = d_memory_dz_next;
      d_memory_dz_next = tmp_dz;
    }
#endif
#ifdef CUDA3D_PML_DEBUG_DUMP
    if (it == pml_dump_step) {
      cudaDeviceSynchronize();
      check_gpu_error_2("sync before PML debug dump");
      dump_pml_debug_state(pml_dump_dir, mytid, snum, it,
			   nby, nbx, nbz, nypad, nxpad, nzpad, nbd,
			   d_p0, d_p1, d_vy, d_vx, d_vz,
			   d_memory_dy, d_memory_dx, d_memory_dz,
			   d_memory_dyy, d_memory_dxx, d_memory_dzz);
    }
#endif
    if (nr <= (size_t)BS) {
      lint3d_inject_bell_extract_gpu_zz<<<1, dimbr>>>(d_p0, nbd, yl, xl, it, nt, snum,
					    d_src, d_bell, nbell,
					    indxy, indxx, indxz, nypad, nxpad, nzpad,
					    d_sw000, d_sw001, d_sw010, d_sw011,
					    d_sw100, d_sw101, d_sw110, d_sw111,
					    d_est, d_rec0_indx, nr,
					    d_rw000, d_rw001, d_rw010, d_rw011,
					    d_rw100, d_rw101, d_rw110, d_rw111);
      check_gpu_error_loop("inject src and extract");
    } else {
      // inject bell src
      lint3d_inject_bell_gpu<<<dims, dimbs>>>(d_p0, nbd, yl, xl, it, snum,
					      d_src, d_bell, nbell,
					      indxy, indxx, indxz, nypad, nxpad, nzpad,
					      d_sw000, d_sw001, d_sw010, d_sw011,
					      d_sw100, d_sw101, d_sw110, d_sw111);
      check_gpu_error_loop("inject src");
      // extract 
      lint3d_extract_gpu_zz<<<dimr, dimbr>>>(d_p0, nbd, yl, xl, it, nt,
					     d_est, d_rec0_indx, nr, nypad, nxpad, nzpad, // ir is fast direction in 3D
					     d_rw000, d_rw001, d_rw010, d_rw011,
					     d_rw100, d_rw101, d_rw110, d_rw111);
    }

#if defined(CUDA3D_CORE_2STEP_DEBUG_DUMP) || defined(CUDA3D_CORE_2STEP_INTERIOR_COMPARE)
    if (core2step_dump_dir != NULL && core2step_dump_dir[0] != '\0' &&
	(core2step_dump_step < 0 || core2step_dump_step == it)) {
      cudaDeviceSynchronize();
      check_gpu_error_2("sync before core 2step debug dump");
      dump_core2step_debug_state(core2step_dump_dir, mytid, snum, it,
				 nby, nbx, nbz, nypad, nxpad, nzpad, nbd,
				 yl, xl, indxy, indxx, indxz,
				 rec0_indx, nr, d_p0, d_p1);
    }
#endif

    ptr=d_p0; d_p0=d_p1; d_p1=ptr;
   
    //might need this or FD CPML, to improve numerical stability for large dt and dx
    //    bc_3d(fu, nby, nbx, nbz, nbd, bt, bb, bl, br);
    //    bc_3d(pfu, nby, nbx, nbz, nbd, bt, bb, bl, br);
  }// end time stepping

  check_gpu_error_2("time stepping");
  cudaMemcpy(&h_est[0][0], d_est, nt*nr*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(t2, 0);
  cudaEventElapsedTime(&mill, t1, t2);
  if(mytid==0)
    printf("mod time %fs\n", (float)(mill)/(1000.));

  free1float(h_bell);  //free3float(out); free2float(out2); free2float(out3);
  cudaFree(d_src0_indx); cudaFree(d_rec0_indx);
  cudaFree(d_bell); cudaFree(d_src); cudaFree(d_cw2); cudaFree(d_est);
  cudaFree(d_sw000); cudaFree(d_sw001); cudaFree(d_sw010); cudaFree(d_sw011);
  cudaFree(d_sw100); cudaFree(d_sw101); cudaFree(d_sw110); cudaFree(d_sw111);
  cudaFree(d_rw000); cudaFree(d_rw001); cudaFree(d_rw010); cudaFree(d_rw011);
  cudaFree(d_rw100); cudaFree(d_rw101); cudaFree(d_rw110); cudaFree(d_rw111);
  cudaFree(d_memory_dy); cudaFree(d_memory_dx); cudaFree(d_memory_dz);
#ifdef CUDA3D_PML_ZMEM_IN_P
  cudaFree(d_memory_dz_next);
#endif
  cudaFree(d_memory_dyy); cudaFree(d_memory_dxx); cudaFree(d_memory_dzz);
#if defined(CUDA3D_PML_TILE_LIST_V) || defined(CUDA3D_PML_TILE_LIST_P)
  if (d_v_pml_tiles) cudaFree(d_v_pml_tiles);
  if (d_p_pml_tiles) cudaFree(d_p_pml_tiles);
  if (h_v_pml_tiles) free(h_v_pml_tiles);
  if (h_p_pml_tiles) free(h_p_pml_tiles);
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  if (d_zface_p_pml_tiles) cudaFree(d_zface_p_pml_tiles);
  if (h_zface_p_pml_tiles) free(h_zface_p_pml_tiles);
#endif
  cudaFree(d_p0); cudaFree(d_p1);
  cudaFree(d_vy); cudaFree(d_vx); cudaFree(d_vz); 
  cudaFree(d_ay); cudaFree(d_by); cudaFree(d_ax); cudaFree(d_bx); cudaFree(d_az); cudaFree(d_bz);
  cudaFree(d_ay_h); cudaFree(d_by_h); cudaFree(d_ax_h); cudaFree(d_bx_h); cudaFree(d_az_h); cudaFree(d_bz_h);
  check_gpu_error_2("Error in FREE");
}

#include "rem_fd.h"

#if defined(CUDA3D_PML_ZMEM_IN_P) && !defined(CUDA3D_PML_RECOMPUTE_Z)
#error "CUDA3D_PML_ZMEM_IN_P requires CUDA3D_PML_RECOMPUTE_Z"
#endif

#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && !defined(CUDA3D_PML_ZMEM_IN_P)
#error "CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL currently requires the stable CUDA3D_PML_ZMEM_IN_P path"
#endif

#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && !defined(CUDA3D_CPML_VMEM_DISABLE_MPI)
#error "CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL phase 1 is single-rank only; define CUDA3D_CPML_VMEM_DISABLE_MPI to acknowledge this gate"
#endif

#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) && !defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL)
#error "CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY requires CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL"
#endif

#if defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG) && !defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL)
#error "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG requires CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL"
#endif

#if defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG) && defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY)
#error "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG replaces the direct fused zface prototype; do not enable both"
#endif

#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) && defined(CUDA3D_PML_ZFACE_P_SPECIALIZE)
#error "CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY replaces the old pressure-only zface specialize path"
#endif

#if defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG) && defined(CUDA3D_PML_ZFACE_P_SPECIALIZE)
#error "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG replaces the old pressure-only zface specialize path"
#endif

#ifdef CUDA3D_PML_DEBUG_DUMP
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

#ifdef CUDA3D_PML_DEBUG_DUMP
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

#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && defined(CUDA3D_CPML_VMEM_DEBUG_FILL)
static void check_cpml_vmem_next_written(const char *name, float *d_memory_next,
					 size_t count, int mytid, int snum, int it) {
  float *host = (float*)malloc(count * sizeof(float));
  if (host == NULL) {
    printf("ERROR allocating CPML VMEM debug host buffer for %s\n", name);
    exit(0);
  }

  cudaMemcpy(host, d_memory_next, count * sizeof(float), cudaMemcpyDeviceToHost);
  check_gpu_error_2("copy CPML VMEM debug buffer");

  size_t non_finite = 0;
  size_t first_non_finite = 0;
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(host[i])) {
      if (non_finite == 0) first_non_finite = i;
      ++non_finite;
    }
  }
  free(host);

  if (non_finite != 0) {
    printf("ERROR CPML VMEM unwritten entries: name=%s rank=%d shot=%d it=%d count=%zu first=%zu\n",
	   name, mytid, snum, it, non_finite, first_non_finite);
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

static int max_int_local(int a, int b) {
  return a > b ? a : b;
}

static int interval_len_local(int lo, int hi) {
  return hi > lo ? hi - lo : 0;
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

#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
static int pml_tile_whole_len16(const PmlTile *tile,
				int n3, int n2, int n1, int npml) {
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  const int z0 = tile->z0;
  const int x0 = tile->x0;
  const int y0 = tile->y0;
  const int z1 = min_int_local(z0 + PmlTileBlockSize1, n1);
  const int x1 = min_int_local(x0 + PmlTileBlockSize2, n2);
  const int y1 = min_int_local(y0 + PmlTileBlockSize3, n3);
  if (x1 - x0 != PmlTileBlockSize2 || y1 - y0 != PmlTileBlockSize3)
    return 0;
  if (!(x0 >= core2_lo && x1 <= core2_hi && y0 >= core3_lo && y1 <= core3_hi))
    return 0;
  const int core_z_overlap = interval_len_local(max_int_local(z0, core1_lo),
						min_int_local(z1, core1_hi));
  const int active_z_len = (z1 - z0) - core_z_overlap;
  return active_z_len == 16;
}

static void split_pml_len16_tiles(PmlTile **p_tiles,
				  int *p_ntile,
				  PmlTile **len16_tiles,
				  int *len16_ntile,
				  int n3, int n2, int n1, int npml) {
  PmlTile *src = *p_tiles;
  const int ntile = *p_ntile;
  PmlTile *residual = (PmlTile*)malloc((size_t)ntile * sizeof(PmlTile));
  PmlTile *packed = (PmlTile*)malloc((size_t)ntile * sizeof(PmlTile));
  if (residual == NULL || packed == NULL) {
    printf("ERROR allocating pressure PML len16 split lists\n");
    exit(0);
  }
  int n_residual = 0;
  int n_packed = 0;
  for (int i = 0; i < ntile; ++i) {
    if (pml_tile_whole_len16(&src[i], n3, n2, n1, npml)) {
      packed[n_packed++] = src[i];
    } else {
      residual[n_residual++] = src[i];
    }
  }
  free(src);
  *p_tiles = residual;
  *p_ntile = n_residual;
  *len16_tiles = packed;
  *len16_ntile = n_packed;
}
#endif

#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
static int v_pml_tile_whole_len16(const PmlTile *tile,
				  int n3, int n2, int n1, int npml) {
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  const int z0 = tile->z0;
  const int x0 = tile->x0;
  const int y0 = tile->y0;
  const int z1 = min_int_local(z0 + PmlTileBlockSize1, n1);
  const int x1 = min_int_local(x0 + PmlTileBlockSize2, n2);
  const int y1 = min_int_local(y0 + PmlTileBlockSize3, n3);
  if (x1 - x0 != PmlTileBlockSize2 || y1 - y0 != PmlTileBlockSize3)
    return 0;
  if (!(x0 >= core2_lo + 3 && x1 <= core2_hi - 4 &&
	y0 >= core3_lo + 3 && y1 <= core3_hi - 4))
    return 0;
  const int core_z_overlap = interval_len_local(max_int_local(z0, core1_lo),
						min_int_local(z1, core1_hi));
  const int active_z_len = (z1 - z0) - core_z_overlap;
  return active_z_len == 16;
}

static void split_v_pml_len16_tiles(PmlTile **v_tiles,
				    int *v_ntile,
				    PmlTile **len16_tiles,
				    int *len16_ntile,
				    int n3, int n2, int n1, int npml) {
  PmlTile *src = *v_tiles;
  const int ntile = *v_ntile;
  PmlTile *residual = (PmlTile*)malloc((size_t)ntile * sizeof(PmlTile));
  PmlTile *packed = (PmlTile*)malloc((size_t)ntile * sizeof(PmlTile));
  if (residual == NULL || packed == NULL) {
    printf("ERROR allocating velocity PML len16 split lists\n");
    exit(0);
  }
  int n_residual = 0;
  int n_packed = 0;
  for (int i = 0; i < ntile; ++i) {
    if (v_pml_tile_whole_len16(&src[i], n3, n2, n1, npml)) {
      packed[n_packed++] = src[i];
    } else {
      residual[n_residual++] = src[i];
    }
  }
  free(src);
  *v_tiles = residual;
  *v_ntile = n_residual;
  *len16_tiles = packed;
  *len16_ntile = n_packed;
}
#endif

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
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
  float *d_memory_dy_next = NULL, *d_memory_dx_next = NULL;
#endif
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
#if defined(CUDA3D_PML_TILE_LIST_V) && defined(CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK)
  PmlTile *h_v_len16_tiles = NULL, *d_v_len16_tiles = NULL;
  int n_v_len16_tiles = 0;
#endif
#if defined(CUDA3D_PML_TILE_LIST_P) && defined(CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK)
  PmlTile *h_p_len16_tiles = NULL, *d_p_len16_tiles = NULL;
  int n_p_len16_tiles = 0;
#endif
#ifdef CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
  float *d_p_len16_compact_dzz16 = NULL;
  float *d_p_len16_compact_dz_old23 = NULL;
  float *d_p_len16_compact_dz_next23 = NULL;
  float *d_p_len16_compact_err_sum = NULL;
  float *d_p_len16_compact_ref_sum = NULL;
  float *d_p_len16_compact_max_abs = NULL;
  int *d_p_len16_compact_bad_count = NULL;
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
  const size_t mem_z_count = (size_t)nby * (size_t)nbx * (size_t)2 * (size_t)nbd;
  const size_t mem_x_count = (size_t)nby * (size_t)2 * (size_t)nbd * (size_t)nbz;
  const size_t mem_y_count = (size_t)2 * (size_t)nbd * (size_t)nbx * (size_t)nbz;
  const size_t mem_z_bytes = mem_z_count * sizeof(float);
  const size_t mem_x_bytes = mem_x_count * sizeof(float);
  const size_t mem_y_bytes = mem_y_count * sizeof(float);
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
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
  cudaMalloc((void**)&d_memory_dy_next, mem_y_bytes);
  cudaMalloc((void**)&d_memory_dx_next, mem_x_bytes);
#endif
#ifdef CUDA3D_PML_ZMEM_IN_P
  cudaMalloc((void**)&d_memory_dz_next, mem_z_bytes);
#endif
  cudaMalloc((void**)&d_memory_dyy, 2*nbd*nbx*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dxx, nby*2*nbd*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dzz, nby*nbx*2*nbd*sizeof(float));

  cudaMemset(d_memory_dy, 0., 2*nbd*nbx*nbz*sizeof(float));
  cudaMemset(d_memory_dx, 0., nby*2*nbd*nbz*sizeof(float));
  cudaMemset(d_memory_dz, 0., nby*nbx*2*nbd*sizeof(float));
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
  cudaMemset(d_memory_dy_next, 0, mem_y_bytes);
  cudaMemset(d_memory_dx_next, 0, mem_x_bytes);
#endif
#ifdef CUDA3D_PML_ZMEM_IN_P
  cudaMemset(d_memory_dz_next, 0, mem_z_bytes);
#endif
  cudaMemset(d_memory_dyy, 0., 2*nbd*nbx*nbz*sizeof(float));
  cudaMemset(d_memory_dxx, 0., nby*2*nbd*nbz*sizeof(float));
  cudaMemset(d_memory_dzz, 0., nby*nbx*2*nbd*sizeof(float));

  check_gpu_error_2("Error in Memset");

  dim3 dimg_v, dimb_v, dimg_p, dimb_p, dimg_pml, dimb_pml, dimg_pml_zface, dimb_pml_zface, dimg_pml_zface_shared, dimb_pml_zface_shared, dims, dimbs, dimr, dimbr;// dims(1,1), dimbs(2*nbell+1, 2*nbell+1);
#if defined(CUDA3D_PML_TILE_LIST_V) && defined(CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK)
  dim3 dimg_v_len16, dimb_v_len16;
#endif
#if defined(CUDA3D_PML_TILE_LIST_P) && defined(CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK)
  dim3 dimg_pml_len16, dimb_pml_len16;
#endif
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

#ifdef CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
  const size_t pml_zface_shared_smem =
    (size_t)(PmlZFaceSharedOut1 + 2 * PmlZFaceSharedHalo) *
    (size_t)(PmlZFaceSharedOut2 + 2 * PmlZFaceSharedHalo) *
    (size_t)(PmlZFaceSharedOut3 + 2 * PmlZFaceSharedHalo) * sizeof(float)
#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
    + ((size_t)PmlZFaceSharedOut1 * (size_t)(PmlZFaceSharedOut2 + 2 * radius - 1) * (size_t)PmlZFaceSharedOut3 +
       (size_t)PmlZFaceSharedOut1 * (size_t)PmlZFaceSharedOut2 * (size_t)(PmlZFaceSharedOut3 + 2 * radius - 1)) * sizeof(float)
#endif
    ;
  dimb_pml_zface_shared.x = PmlZFaceSharedThreads;
  dimb_pml_zface_shared.y = 1;
  dimb_pml_zface_shared.z = 1;
  dimg_pml_zface_shared.x = ceil_div_int(core_nx, PmlZFaceSharedOut2);
  dimg_pml_zface_shared.y = ceil_div_int(core_ny, PmlZFaceSharedOut3);
  dimg_pml_zface_shared.z = 2;
#endif

#if defined(CUDA3D_PML_TILE_LIST_V) || defined(CUDA3D_PML_TILE_LIST_P)
#ifdef CUDA3D_PML_TILE_LIST_V
  n_v_pml_tiles = build_pml_tile_list(&h_v_pml_tiles, nby, nbx, nbz, nbd, 1);
#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
  split_v_pml_len16_tiles(&h_v_pml_tiles, &n_v_pml_tiles,
			  &h_v_len16_tiles, &n_v_len16_tiles,
			  nby, nbx, nbz, nbd);
  if (n_v_len16_tiles > 0) {
    cudaMalloc((void**)&d_v_len16_tiles, (size_t)n_v_len16_tiles * sizeof(PmlTile));
    cudaMemcpy(d_v_len16_tiles, h_v_len16_tiles, (size_t)n_v_len16_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
  }
  if (n_v_pml_tiles > 0) {
    cudaMalloc((void**)&d_v_pml_tiles, (size_t)n_v_pml_tiles * sizeof(PmlTile));
    cudaMemcpy(d_v_pml_tiles, h_v_pml_tiles, (size_t)n_v_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
  }
#else
  cudaMalloc((void**)&d_v_pml_tiles, (size_t)n_v_pml_tiles * sizeof(PmlTile));
  cudaMemcpy(d_v_pml_tiles, h_v_pml_tiles, (size_t)n_v_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
#endif
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
  n_p_pml_tiles = build_pml_tile_list(&h_p_pml_tiles, nby, nbx, nbz, nbd, 0);
#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
  split_pml_len16_tiles(&h_p_pml_tiles, &n_p_pml_tiles,
			&h_p_len16_tiles, &n_p_len16_tiles,
			nby, nbx, nbz, nbd);
  if (n_p_len16_tiles > 0) {
    cudaMalloc((void**)&d_p_len16_tiles, (size_t)n_p_len16_tiles * sizeof(PmlTile));
    cudaMemcpy(d_p_len16_tiles, h_p_len16_tiles, (size_t)n_p_len16_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
#ifdef CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
    const size_t compact_lines = (size_t)n_p_len16_tiles * PmlTileBlockSize2 * PmlTileBlockSize3;
    cudaMalloc((void**)&d_p_len16_compact_dzz16, compact_lines * 16u * sizeof(float));
    cudaMalloc((void**)&d_p_len16_compact_dz_old23, compact_lines * 23u * sizeof(float));
    cudaMalloc((void**)&d_p_len16_compact_dz_next23, compact_lines * 23u * sizeof(float));
    cudaMalloc((void**)&d_p_len16_compact_err_sum, sizeof(float));
    cudaMalloc((void**)&d_p_len16_compact_ref_sum, sizeof(float));
    cudaMalloc((void**)&d_p_len16_compact_max_abs, sizeof(float));
    cudaMalloc((void**)&d_p_len16_compact_bad_count, sizeof(int));
#endif
  }
#endif
#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
  if (n_p_pml_tiles > 0) {
    cudaMalloc((void**)&d_p_pml_tiles, (size_t)n_p_pml_tiles * sizeof(PmlTile));
    cudaMemcpy(d_p_pml_tiles, h_p_pml_tiles, (size_t)n_p_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
  }
#else
  cudaMalloc((void**)&d_p_pml_tiles, (size_t)n_p_pml_tiles * sizeof(PmlTile));
  cudaMemcpy(d_p_pml_tiles, h_p_pml_tiles, (size_t)n_p_pml_tiles * sizeof(PmlTile), cudaMemcpyHostToDevice);
#endif
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
#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
  dimb_v_len16.x=32;
  dimb_v_len16.y=4;
  dimb_v_len16.z=1;
  dimg_v_len16.x=n_v_len16_tiles;
  dimg_v_len16.y=1;
  dimg_v_len16.z=1;
#endif
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
  dimb_pml.x=PmlTileBlockSize1;
  dimb_pml.y=PmlTileBlockSize2;
  dimb_pml.z=PmlTileBlockSize3;
  dimg_pml.x=n_p_pml_tiles;
  dimg_pml.y=1;
  dimg_pml.z=1;
#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
  dimb_pml_len16.x=32;
  dimb_pml_len16.y=4;
  dimb_pml_len16.z=1;
  dimg_pml_len16.x=n_p_len16_tiles;
  dimg_pml_len16.y=1;
  dimg_pml_len16.z=1;
#endif
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
#if defined(CUDA3D_PML_TILE_LIST_V) && defined(CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK)
  if(mytid==0)
    printf("PML velocity len16 halfwarp enabled: len16_tiles=%d residual_v_tiles=%d block=32x4x1\n",
	   n_v_len16_tiles, n_v_pml_tiles);
#endif
#if defined(CUDA3D_PML_TILE_LIST_P) && defined(CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK)
  if(mytid==0)
    printf("PML pressure len16 halfwarp enabled: len16_tiles=%d residual_p_tiles=%d block=32x4x1\n",
	   n_p_len16_tiles, n_p_pml_tiles);
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  if(mytid==0)
    printf("PML zface pressure specialize enabled: zface_p_tiles=%d, block=%dx%dx%d\n",
	   n_zface_p_pml_tiles,
	   PmlZFaceBlockSize1, PmlZFaceBlockSize2, PmlZFaceBlockSize3);
#endif
#ifdef CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
  if(mytid==0)
    printf("PML zface shared VP debug enabled: grid=%dx%dx%d out=%dx%dx%d threads=%d smem=%zu stage_v=%d\n",
	   (int)dimg_pml_zface_shared.x, (int)dimg_pml_zface_shared.y, (int)dimg_pml_zface_shared.z,
	   PmlZFaceSharedOut1, PmlZFaceSharedOut2, PmlZFaceSharedOut3,
	   PmlZFaceSharedThreads, pml_zface_shared_smem,
#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
	   1
#else
	   0
#endif
	   );
#endif
#endif

  cudaFuncSetCacheConfig(cuda_fd3d_v_pml_ns, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(cuda_fd3d_p_core_ns, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_ns, cudaFuncCachePreferL1);
#ifdef CUDA3D_PML_TILE_LIST_V
  cudaFuncSetCacheConfig(cuda_fd3d_v_pml_tile_ns, cudaFuncCachePreferL1);
#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
  cudaFuncSetCacheConfig(cuda_fd3d_v_pml_len16_halfwarp_ns, cudaFuncCachePreferL1);
#endif
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_tile_ns, cudaFuncCachePreferL1);
#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_len16_halfwarp_ns, cudaFuncCachePreferL1);
#endif
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_zface_ns, cudaFuncCachePreferL1);
#endif
#ifdef CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
  cudaFuncSetAttribute(cuda_fd3d_pml_zface_shared_vp_debug_ns,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       (int)pml_zface_shared_smem);
  cudaFuncSetCacheConfig(cuda_fd3d_pml_zface_shared_vp_debug_ns, cudaFuncCachePreferShared);
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
#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && defined(CUDA3D_CPML_VMEM_DEBUG_FILL)
    cudaMemset(d_memory_dy_next, 0xff, mem_y_bytes);
    cudaMemset(d_memory_dx_next, 0xff, mem_x_bytes);
    cudaMemset(d_memory_dz_next, 0xff, mem_z_bytes);
    check_gpu_error_loop("fill CPML VMEM next");
#endif
#ifdef CUDA3D_PML_TILE_LIST_V
#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
    if (n_v_len16_tiles > 0) {
      cuda_fd3d_v_pml_len16_halfwarp_ns<<<dimg_v_len16, dimb_v_len16>>>(d_p1, d_vy, d_vx, d_vz,
				    tdy, tdx, tdz,
				    nby, nbx, nbz, nbd, dt,
				    d_ay_h, d_by_h, d_ax_h, d_bx_h, d_az_h, d_bz_h,
				    d_memory_dy, d_memory_dx, d_memory_dz,
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
				    d_memory_dy_next, d_memory_dx_next, d_memory_dz_next,
#endif
				    d_v_len16_tiles, n_v_len16_tiles);
      check_gpu_error_loop("compute V pml len16 halfwarp");
    }
    if (n_v_pml_tiles > 0) {
#endif
      cuda_fd3d_v_pml_tile_ns<<<dimg_v, dimb_v>>>(d_p1, d_vy, d_vx, d_vz,
				    tdy, tdx, tdz,
				    nby, nbx, nbz, nbd, dt,
				    d_ay_h, d_by_h, d_ax_h, d_bx_h, d_az_h, d_bz_h,
				    d_memory_dy, d_memory_dx, d_memory_dz,
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
				    d_memory_dy_next, d_memory_dx_next, d_memory_dz_next,
#endif
				    d_v_pml_tiles, n_v_pml_tiles);
#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
    }
#endif
#else
    cuda_fd3d_v_pml_ns<<<dimg_v, dimb_v>>>(d_p1, d_vy, d_vx, d_vz,
				    tdy, tdx, tdz,
				    nby, nbx, nbz, nbd, dt,
				    d_ay_h, d_by_h, d_ax_h, d_bx_h, d_az_h, d_bz_h,
				    d_memory_dy, d_memory_dx, d_memory_dz
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
				    , d_memory_dy_next, d_memory_dx_next, d_memory_dz_next
#endif
				    );
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
#ifdef CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
    cuda_fd3d_pml_zface_shared_vp_debug_ns<<<dimg_pml_zface_shared, dimb_pml_zface_shared,
					     pml_zface_shared_smem >>>(d_p0, d_p1,
								       d_cw2, tdy, tdx, tdz,
								       nby, nbx, nbz, nbd, dt2,
								       d_memory_dzz,
								       d_memory_dz,
								       d_memory_dz_next);
    check_gpu_error_loop("compute P pml zface shared VP debug");
#endif
#ifdef CUDA3D_PML_TILE_LIST_P
#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
    if (n_p_len16_tiles > 0) {
      cuda_fd3d_p_pml_len16_halfwarp_ns<<<dimg_pml_len16, dimb_pml_len16 >>>(d_p0, d_p1, d_vy, d_vx,
					     d_cw2, tdy, tdx, tdz,
					     nby, nbx, nbz, nbd, dt2,
					     d_memory_dzz,
					     d_memory_dz,
					     d_memory_dz_next,
					     d_p_len16_tiles, n_p_len16_tiles);
      check_gpu_error_loop("compute P pml len16 halfwarp");
#ifdef CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
      if (it < 3 || it == nt - 1) {
	const size_t mirror_items = (size_t)n_p_len16_tiles * PmlTileBlockSize2 * PmlTileBlockSize3 * 23u;
	const int mirror_threads = 256;
	int mirror_blocks = (int)((mirror_items + mirror_threads - 1u) / mirror_threads);
	if (mirror_blocks > 65535) mirror_blocks = 65535;
	cudaMemset(d_p_len16_compact_err_sum, 0, sizeof(float));
	cudaMemset(d_p_len16_compact_ref_sum, 0, sizeof(float));
	cudaMemset(d_p_len16_compact_max_abs, 0, sizeof(float));
	cudaMemset(d_p_len16_compact_bad_count, 0, sizeof(int));
	cuda3d_pml_len16_compact_state_gather_ns<<<mirror_blocks, mirror_threads>>>(
	  d_memory_dzz, d_memory_dz, d_memory_dz_next,
	  d_p_len16_compact_dzz16, d_p_len16_compact_dz_old23, d_p_len16_compact_dz_next23,
	  d_p_len16_tiles, n_p_len16_tiles, nby, nbx, nbz, nbd);
	check_gpu_error_loop("gather P len16 compact mirror");
	cuda3d_pml_len16_compact_state_compare_ns<<<mirror_blocks, mirror_threads>>>(
	  d_memory_dzz, d_memory_dz, d_memory_dz_next,
	  d_p_len16_compact_dzz16, d_p_len16_compact_dz_old23, d_p_len16_compact_dz_next23,
	  d_p_len16_tiles, n_p_len16_tiles, nby, nbx, nbz, nbd,
	  d_p_len16_compact_err_sum, d_p_len16_compact_ref_sum,
	  d_p_len16_compact_max_abs, d_p_len16_compact_bad_count);
	check_gpu_error_loop("compare P len16 compact mirror");
	float h_compact_err = 0.0f, h_compact_ref = 0.0f, h_compact_max = 0.0f;
	int h_compact_bad = 0;
	cudaMemcpy(&h_compact_err, d_p_len16_compact_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_compact_ref, d_p_len16_compact_ref_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_compact_max, d_p_len16_compact_max_abs, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_compact_bad, d_p_len16_compact_bad_count, sizeof(int), cudaMemcpyDeviceToHost);
	check_gpu_error_loop("copy P len16 compact mirror stats");
	const double rel_l2 = sqrt((double)h_compact_err / ((double)h_compact_ref + 1.0e-30));
	if (mytid == 0)
	  printf("PML len16 compact-state mirror it=%d rel_l2=%e max_abs=%e bad=%d\n",
		 it, rel_l2, (double)h_compact_max, h_compact_bad);
	if (!isfinite(rel_l2) || rel_l2 > 1.0e-6 || h_compact_bad != 0) {
	  printf("ERROR PML len16 compact-state mirror mismatch it=%d rel_l2=%e max_abs=%e bad=%d\n",
		 it, rel_l2, (double)h_compact_max, h_compact_bad);
	  exit(1);
	}
      }
#endif
    }
#endif
#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
    if (n_p_pml_tiles > 0) {
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
      check_gpu_error_loop("compute P pml");
    }
#else
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
    check_gpu_error_loop("compute P pml");
#endif
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
    check_gpu_error_loop("compute P pml");
#endif
#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && defined(CUDA3D_CPML_VMEM_DEBUG_FILL)
    cudaDeviceSynchronize();
    check_gpu_error_2("sync before CPML VMEM coverage check");
    check_cpml_vmem_next_written("memory_dy_next", d_memory_dy_next, mem_y_count, mytid, snum, it);
    check_cpml_vmem_next_written("memory_dx_next", d_memory_dx_next, mem_x_count, mytid, snum, it);
    check_cpml_vmem_next_written("memory_dz_next", d_memory_dz_next, mem_z_count, mytid, snum, it);
#endif
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
    {
      float *tmp_dy = d_memory_dy;
      d_memory_dy = d_memory_dy_next;
      d_memory_dy_next = tmp_dy;

      float *tmp_dx = d_memory_dx;
      d_memory_dx = d_memory_dx_next;
      d_memory_dx_next = tmp_dx;

      float *tmp_dz = d_memory_dz;
      d_memory_dz = d_memory_dz_next;
      d_memory_dz_next = tmp_dz;
    }
#else
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
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
  cudaFree(d_memory_dy_next); cudaFree(d_memory_dx_next);
#endif
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
#if defined(CUDA3D_PML_TILE_LIST_V) && defined(CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK)
  if (d_v_len16_tiles) cudaFree(d_v_len16_tiles);
  if (h_v_len16_tiles) free(h_v_len16_tiles);
#endif
#if defined(CUDA3D_PML_TILE_LIST_P) && defined(CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK)
  if (d_p_len16_tiles) cudaFree(d_p_len16_tiles);
  if (h_p_len16_tiles) free(h_p_len16_tiles);
#endif
#ifdef CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
  if (d_p_len16_compact_dzz16) cudaFree(d_p_len16_compact_dzz16);
  if (d_p_len16_compact_dz_old23) cudaFree(d_p_len16_compact_dz_old23);
  if (d_p_len16_compact_dz_next23) cudaFree(d_p_len16_compact_dz_next23);
  if (d_p_len16_compact_err_sum) cudaFree(d_p_len16_compact_err_sum);
  if (d_p_len16_compact_ref_sum) cudaFree(d_p_len16_compact_ref_sum);
  if (d_p_len16_compact_max_abs) cudaFree(d_p_len16_compact_max_abs);
  if (d_p_len16_compact_bad_count) cudaFree(d_p_len16_compact_bad_count);
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

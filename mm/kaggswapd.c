// SPDX-License-Identifier: GPL-2.0
/*
 * kaggswapd has two scan classes:
 *
 * (1) Sampling scan: stamp window ids into page_ext_agg_data.
 *     - LRU sampling: scan pages and call kagg_try_mark_window().
 *     - VMA sampling: pick a small cold seed batch from inactive anon tail,
 *       then fan out via RMAP + walk_page_vma() and stamp related anon pages.
 *
 * (2) Mining scan: isolate pages in batches and build hash buckets for
 *     similarity clustering (current tree: batch + bucket + putback skeleton).
 *
 * page_ext_agg_data::last_kagg_win dedups repeated stamps in the same window.
 */

#include <linux/freezer.h>
#include <linux/hugetlb.h>
#include <linux/kaggswapd.h>
#include <linux/kthread.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/nodemask.h>
#include <linux/jhash.h>
#include <linux/page_ext.h>
#include <linux/pagewalk.h>
#include <linux/atomic.h>
#include <linux/slab.h>
#include <linux/rmap.h>
#include <linux/sched.h>
#include <linux/sort.h>
#include <linux/spinlock.h>
#include <linux/swap.h>
#include <linux/printk.h>

#ifdef CONFIG_TRANSPARENT_HUGEPAGE
#include <linux/huge_mm.h>
#endif

#ifdef CONFIG_KAGGSWAPD_DEBUG
#include <linux/debugfs.h>
#include <linux/module.h>
#include <linux/seq_file.h>
#endif

#include "internal.h"

int sysctl_dual_path_scan_mode __read_mostly =
	(int)KAGG_SCAN_INACTIVE_PCT;
int sysctl_dual_path_scan_pct __read_mostly = 30;
int sysctl_dual_path_scan_interval_ms __read_mostly = 100;

int sysctl_dual_path_miner_enable __read_mostly;
int sysctl_dual_path_miner_interval_ms __read_mostly = 6000;
int sysctl_dual_path_miner_similarity_pct __read_mostly = 75;
int sysctl_dual_path_miner_min_cluster __read_mostly = 33;

#define KAGG_MINER_HASH_BITS	10
#define KAGG_MINER_HASH_SIZE	(1U << KAGG_MINER_HASH_BITS)
#define KAGG_MINER_PREFIX	5
#define KAGG_VMA_SEED_MAX	1024U
#define KAGG_MINER_BATCH_MAX	128U
#define KAGG_MINER_CLUSTER_MAX	256U
#define KAGG_MINER_RESCHED_QUOTA	64U
#define KAGG_MINER_VIEW_MAX	8192U
/*
 * Max inactive-anon LRU slots scanned in one isolate_chunk() call when filling
 * chunk_target. Allows wrapping past tail (list head) mid-chunk when the
 * round-robin cursor falls in a sparse region, without unbounded lru_lock hold.
 */
#define KAGG_MINER_ISO_SCAN_ABS_MAX	32768U
/*
 * Bound greedy seed rounds per hash bucket: with a high min_cluster and low
 * match rate, seeds are repeatedly peeled and tiny clusters rebucket'd back,
 * which can require an enormous number of iterations before the bucket drains.
 */
#define KAGG_MINER_BUCKET_SEED_MAX	8192U

static atomic_t kagg_win_next_atomic = ATOMIC_INIT(1);
/*
 * Scratch buffers for kaggswapd thread to avoid large on-stack arrays.
 * kaggswapd is single-threaded in this tree.
 */
static struct page *kagg_vma_seed_buf[KAGG_VMA_SEED_MAX];
static struct hlist_head kagg_miner_bucket_buf[KAGG_MINER_HASH_SIZE];
static struct page *kagg_miner_emit_pages_buf[KAGG_MINER_CLUSTER_MAX];
static struct page_ext_agg_data *kagg_miner_cluster_memd_buf[KAGG_MINER_CLUSTER_MAX];

/*
 * Sorted (by page_ext_agg_data *) map for isolated pages: rebuilt once per
 * kagg_miner_cluster_and_putback to avoid O(nmem * isolated) scans in emit.
 */
struct kagg_miner_iso_pair {
	struct page_ext_agg_data	*d;
	struct page			*page;
};

static struct kagg_miner_iso_pair kagg_miner_iso_map[KAGG_MINER_VIEW_MAX];
static unsigned int kagg_miner_iso_nmap;

/*
 * Per-memcg miner metadata: cursor and key counters.
 * - cursor_lru_steps[]: inactive anon LRU resume cursor per node.
 * - counters: best-effort cumulative stats (debug visibility).
 */
struct kagg_memcg_meta {
	struct hlist_node		node;
	unsigned short			memcg_id;
	unsigned int			cursor_lru_steps[MAX_NUMNODES];
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_t			miner_rounds;
	atomic64_t			view_target_pages;
	atomic64_t			isolated_pages;
	atomic64_t			clusters_emitted;
	atomic64_t			cluster_pages;
	atomic64_t			putback_pages;
	atomic64_t			bucket_trunc;
#endif
};

#ifdef CONFIG_MEMCG
#define KAGG_MEMCG_META_HASH_BITS	8
#define KAGG_MEMCG_META_HASH_SIZE	(1U << KAGG_MEMCG_META_HASH_BITS)
static struct hlist_head kagg_memcg_meta_hash[KAGG_MEMCG_META_HASH_SIZE];
static DEFINE_SPINLOCK(kagg_memcg_meta_lock);
static inline unsigned int kagg_memcg_meta_hash_idx(unsigned short memcg_id)
{
	return (unsigned int)memcg_id & (KAGG_MEMCG_META_HASH_SIZE - 1U);
}
#endif
static struct kagg_memcg_meta kagg_root_meta;

static struct kagg_memcg_meta *kagg_memcg_meta_get(struct mem_cgroup *memcg,
						    bool create)
{
#ifdef CONFIG_MEMCG
	unsigned short id;
	struct kagg_memcg_meta *meta;
	struct kagg_memcg_meta *new_meta;
	unsigned int idx;

	if (mem_cgroup_disabled() || !memcg)
		return &kagg_root_meta;

	id = mem_cgroup_id(memcg);
	if (!id)
		return &kagg_root_meta;

	idx = kagg_memcg_meta_hash_idx(id);
	spin_lock(&kagg_memcg_meta_lock);
	hlist_for_each_entry(meta, &kagg_memcg_meta_hash[idx], node) {
		if (meta->memcg_id == id) {
			spin_unlock(&kagg_memcg_meta_lock);
			return meta;
		}
	}
	spin_unlock(&kagg_memcg_meta_lock);

	if (!create)
		return NULL;

	new_meta = kzalloc(sizeof(*new_meta), GFP_ATOMIC);
	if (!new_meta)
		return NULL;
	new_meta->memcg_id = id;
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_set(&new_meta->miner_rounds, 0);
	atomic64_set(&new_meta->view_target_pages, 0);
	atomic64_set(&new_meta->isolated_pages, 0);
	atomic64_set(&new_meta->clusters_emitted, 0);
	atomic64_set(&new_meta->cluster_pages, 0);
	atomic64_set(&new_meta->putback_pages, 0);
	atomic64_set(&new_meta->bucket_trunc, 0);
#endif

	spin_lock(&kagg_memcg_meta_lock);
	hlist_for_each_entry(meta, &kagg_memcg_meta_hash[idx], node) {
		if (meta->memcg_id == id) {
			spin_unlock(&kagg_memcg_meta_lock);
			kfree(new_meta);
			return meta;
		}
	}
	hlist_add_head(&new_meta->node, &kagg_memcg_meta_hash[idx]);
	spin_unlock(&kagg_memcg_meta_lock);
	return new_meta;
#else
	(void)memcg;
	(void)create;
	return &kagg_root_meta;
#endif
}

#if defined(CONFIG_MEMCG)
/*
 * mem_cgroup_css_offline() calls this before mem_cgroup_id_put(): drop per-id
 * miner cursor/stats so recycled memcg_id cannot alias stale meta (slab leak).
 */
void kagg_memcg_meta_free(struct mem_cgroup *memcg)
{
	unsigned short id;
	unsigned int idx;
	struct kagg_memcg_meta *meta;

	if (!memcg || mem_cgroup_disabled())
		return;
	id = mem_cgroup_id(memcg);
	if (!id)
		return;

	idx = kagg_memcg_meta_hash_idx(id);
	spin_lock(&kagg_memcg_meta_lock);
	hlist_for_each_entry(meta, &kagg_memcg_meta_hash[idx], node) {
		if (meta->memcg_id == id) {
			hlist_del_init(&meta->node);
			spin_unlock(&kagg_memcg_meta_lock);
			kfree(meta);
			return;
		}
	}
	spin_unlock(&kagg_memcg_meta_lock);
}
#endif

static unsigned int kagg_percent_target(unsigned long total, unsigned int pct,
					unsigned int cap)
{
	unsigned long target;

	if (!total)
		return 0;
	target = (total * (unsigned long)pct + 99UL) / 100UL;
	if (!target)
		target = 1;
	if (target > cap)
		target = cap;
	return (unsigned int)target;
}

/**
 * kagg_next_window_id - next u16 window id for per-page aggregation ring.
 *
 * Lock-free compare-and-exchange on 1..65535 (0 reserved / never returned).
 * Miner must tolerate rare alias vs last_kagg_win after long uptime.
 */
u16 kagg_next_window_id(void)
{
	int expected;
	int seen;
	unsigned u;
	unsigned n;

	expected = atomic_read(&kagg_win_next_atomic);

	for (;;) {
		u = (unsigned)expected & 0xFFFFU;
		n = (u + 1U) & 0xFFFFU;
		if (!n)
			n = 1U;
		seen = atomic_cmpxchg_relaxed(&kagg_win_next_atomic,
					      expected, (int)n);
		if (seen == expected)
			return (u16)u;
		expected = seen;
	}
}

#ifdef CONFIG_KAGGSWAPD_DEBUG

static atomic64_t kagg_dbg_scan_rounds;
static atomic64_t kagg_dbg_pages_stamped;
static atomic64_t kagg_dbg_dedup_same_win;
static atomic64_t kagg_dbg_miner_rounds;
static atomic64_t kagg_dbg_miner_isolated_pages;
static atomic64_t kagg_dbg_miner_clusters_emitted;
static atomic64_t kagg_dbg_miner_cluster_pages;
static atomic64_t kagg_dbg_miner_putback_pages;
static atomic64_t kagg_dbg_miner_groups_cleared;
static atomic64_t kagg_dbg_miner_members_linked;
static atomic64_t kagg_dbg_miner_bucket_trunc;
static atomic64_t kagg_dbg_miner_block_start;
static atomic64_t kagg_dbg_miner_view_target_pages;
static atomic64_t kagg_dbg_miner_view_target_cap_hits;
static atomic64_t kagg_dbg_miner_iso_chunks;
static atomic64_t kagg_dbg_miner_iso_target_slots;
static atomic64_t kagg_dbg_miner_iso_walked_slots;
static atomic64_t kagg_dbg_miner_iso_chunk_underfill;
static atomic64_t kagg_dbg_miner_iso_chunk_maxscan_hit;
/* LRU walk: skipped before elevate refcount */
static atomic64_t kagg_dbg_miner_iso_skip_non_anon;
static atomic64_t kagg_dbg_miner_iso_skip_huge;
static atomic64_t kagg_dbg_miner_iso_skip_not_lru_walk;
static atomic64_t kagg_dbg_miner_iso_skip_not_evictable;
static atomic64_t kagg_dbg_miner_iso_skip_no_ref;
/* isolate_lru_page: attempts == sum(nr_cands); classify -EBUSY-ish paths */
static atomic64_t kagg_dbg_miner_iso_attempts;
static atomic64_t kagg_dbg_miner_iso_fail_not_lru;
static atomic64_t kagg_dbg_miner_iso_fail_race;

#ifdef CONFIG_MEMCG
#define KAGG_MEMCG_TOP_N	10

struct kagg_memcg_top_row {
	unsigned int	memcg_id;
	unsigned long	anon_act;
	unsigned long	anon_inact;
	long long	miner_rounds;
	long long	group_size;
	long long	clustered_pages;
};

/* Descending by group_size (clusters_emitted); tie-break clustered_pages, memcg_id. */
static int kagg_memcg_top_row_cmp(const void *a, const void *b)
{
	const struct kagg_memcg_top_row *ra = a;
	const struct kagg_memcg_top_row *rb = b;

	if (ra->group_size > rb->group_size)
		return -1;
	if (ra->group_size < rb->group_size)
		return 1;
	if (ra->clustered_pages > rb->clustered_pages)
		return -1;
	if (ra->clustered_pages < rb->clustered_pages)
		return 1;
	if (ra->memcg_id < rb->memcg_id)
		return -1;
	if (ra->memcg_id > rb->memcg_id)
		return 1;
	return 0;
}

static void kagg_memcg_top_row_fill(struct mem_cgroup *memcg,
				    struct kagg_memcg_top_row *row)
{
	struct kagg_memcg_meta *meta;

	row->memcg_id = (unsigned int)mem_cgroup_id(memcg);
	row->anon_act = memcg_page_state(memcg, NR_ACTIVE_ANON);
	row->anon_inact = memcg_page_state(memcg, NR_INACTIVE_ANON);
	meta = kagg_memcg_meta_get(memcg, false);
	row->miner_rounds = meta ? (long long)atomic64_read(&meta->miner_rounds) : 0LL;
	row->group_size = meta ? (long long)atomic64_read(&meta->clusters_emitted) : 0LL;
	row->clustered_pages = meta ? (long long)atomic64_read(&meta->cluster_pages) : 0LL;
}
#endif /* CONFIG_MEMCG */

static int kagg_summary_show(struct seq_file *m, void *v)
{
	u64 miner_rounds;
	unsigned long anon_active, anon_inactive, file_active, file_inactive;
	u64 anon_total, file_total;
	u64 memcg_total = 0, memcg_online_nr = 0;
	u64 memcg_max_depth = 0;
	u64 total_groups = 0, total_group_pages = 0;

	(void)v;

	anon_active = global_node_page_state(NR_ACTIVE_ANON);
	anon_inactive = global_node_page_state(NR_INACTIVE_ANON);
	file_active = global_node_page_state(NR_ACTIVE_FILE);
	file_inactive = global_node_page_state(NR_INACTIVE_FILE);
	anon_total = (u64)anon_active + (u64)anon_inactive;
	file_total = (u64)file_active + (u64)file_inactive;

	seq_puts(m, "=== 系统概况 (System) ===\n");
	seq_printf(m, "numa_online_nodes(NUMA节点数) %u\n", num_online_nodes());
	seq_printf(m, "mem_total_pages(系统总页) %lu\n", totalram_pages());
	seq_printf(m, "mem_anon_pages(匿名页总数) %llu [active=%lu inactive=%lu]\n",
		   (unsigned long long)anon_total, anon_active, anon_inactive);
	seq_printf(m, "mem_file_pages(文件页总数) %llu [active=%lu inactive=%lu]\n",
		   (unsigned long long)file_total, file_active, file_inactive);
#ifdef CONFIG_MEMCG
	if (!mem_cgroup_disabled()) {
		struct mem_cgroup *memcg = NULL;
		struct kagg_memcg_top_row *rows = NULL;
		size_t row_n = 0, row_cap = 0;

		while ((memcg = mem_cgroup_iter(NULL, memcg, NULL))) {
			struct mem_cgroup *p = memcg;
			u64 depth = 0;

			memcg_total++;
			if (mem_cgroup_online(memcg))
				memcg_online_nr++;
			while (p) {
				depth++;
				p = parent_mem_cgroup(p);
			}
			if (depth > memcg_max_depth)
				memcg_max_depth = depth;

			if (row_n >= row_cap) {
				struct kagg_memcg_top_row *nr;
				size_t ncap;

				if (!row_cap)
					ncap = 64;
				else if (row_cap > (SIZE_MAX / 2 / sizeof(*rows)))
					ncap = SIZE_MAX / sizeof(*rows);
				else
					ncap = row_cap * 2;
				nr = kvmalloc_array(ncap, sizeof(*rows), GFP_KERNEL);
				if (!nr) {
					mem_cgroup_iter_break(NULL, memcg);
					break;
				}
				if (rows) {
					memcpy(nr, rows, row_n * sizeof(*rows));
					kvfree(rows);
				}
				rows = nr;
				row_cap = ncap;
			}
			kagg_memcg_top_row_fill(memcg, &rows[row_n]);
			row_n++;
		}
		seq_printf(m,
			   "memcg_summary(total/online/hierarchy/max_depth) %llu/%llu/%u/%llu\n",
			   (unsigned long long)memcg_total,
			   (unsigned long long)memcg_online_nr,
			   root_mem_cgroup->use_hierarchy ? 1U : 0U,
			   (unsigned long long)memcg_max_depth);
		seq_puts(m,
			 "memcg_top10(by group_size desc, top 10; group_size=clusters_emitted; clustered_pages=cluster_pages; no meta => last three are 0)\n");
		if (rows && row_n) {
			size_t i;
			unsigned int lines;

			sort(rows, row_n, sizeof(*rows), kagg_memcg_top_row_cmp, NULL);
			lines = (unsigned int)min_t(size_t, row_n, (size_t)KAGG_MEMCG_TOP_N);
			for (i = 0; i < lines; i++) {
				struct kagg_memcg_top_row *r = &rows[i];

				seq_printf(m,
					   "memcg_id=%u active_anon_pages=%lu inactive_anon_pages=%lu miner_rounds=%lld group_size=%lld clustered_pages=%lld\n",
					   r->memcg_id, r->anon_act, r->anon_inact,
					   r->miner_rounds, r->group_size,
					   r->clustered_pages);
			}
		} else if (memcg_total) {
			seq_puts(m,
				 "memcg_top10(alloc_failed or incomplete; try smaller hierarchy or raise memory)\n");
		}
		kvfree(rows);
	} else {
		seq_puts(m, "memcg_status(状态) disabled\n");
	}
#else
	seq_puts(m, "memcg_status(状态) not_configured\n");
#endif

	seq_puts(m, "\n=== 采样统计 (Sampling) ===\n");
	seq_printf(m, "scan_rounds(采样轮次) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_scan_rounds));
	seq_printf(m, "pages_stamped(打标页数) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_pages_stamped));
	seq_printf(m, "dedup_same_window(同窗去重次数) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_dedup_same_win));

	seq_puts(m, "\n=== 挖掘核心统计 (Mining Core) ===\n");
	miner_rounds = (u64)atomic64_read(&kagg_dbg_miner_rounds);
	seq_printf(m, "miner_rounds(挖掘轮次) %lld\n",
		   (long long)miner_rounds);
	seq_printf(m, "miner_isolated_pages(隔离页累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_isolated_pages));
	seq_printf(m, "miner_view_target_pages(挖掘目标页累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_view_target_pages));

	seq_puts(m, "\n--- 聚类结果 (Clustering) ---\n");
	seq_printf(m, "miner_clusters_emitted(产出组累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_clusters_emitted));
	seq_printf(m, "miner_cluster_pages(聚合页累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_cluster_pages));
	seq_printf(m, "miner_putback_pages(LRU回填页累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_putback_pages));
	seq_printf(m, "miner_groups_cleared(清理组累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_groups_cleared));
	seq_printf(m, "miner_members_linked(入组成员页累计) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_members_linked));
	seq_printf(m, "miner_bucket_trunc(分桶截断次数) %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_bucket_trunc));
	seq_puts(m, "\n=== 聚合组概览 (best-effort) ===\n");
#ifdef CONFIG_DUAL_PATH_SWAP
	{
		struct pglist_data *pgdat;

		for_each_online_pgdat(pgdat) {
			struct page_group *grp;
			unsigned long flags;
#ifdef CONFIG_MEMCG
			if (!mem_cgroup_disabled()) {
				struct mem_cgroup *memcg = NULL;

				while ((memcg = mem_cgroup_iter(NULL, memcg, NULL))) {
					struct lruvec *lruvec;

					if (!mem_cgroup_online(memcg))
						continue;
					lruvec = mem_cgroup_lruvec(memcg, pgdat);
					spin_lock_irqsave(&pgdat->lru_lock, flags);
					list_for_each_entry(grp, &lruvec->agg_list, lru) {
						total_groups++;
						total_group_pages += grp->nr_pages;
					}
					spin_unlock_irqrestore(&pgdat->lru_lock, flags);
				}
				continue;
			}
#endif
			{
				struct lruvec *lruvec = mem_cgroup_lruvec(NULL, pgdat);

			spin_lock_irqsave(&pgdat->lru_lock, flags);
			list_for_each_entry(grp, &lruvec->agg_list, lru) {
				total_groups++;
				total_group_pages += grp->nr_pages;
			}
			spin_unlock_irqrestore(&pgdat->lru_lock, flags);
			}
		}
		if (total_groups) {
			u64 avg_int = div64_u64(total_group_pages, total_groups);
			u64 avg_frac = div64_u64((total_group_pages % total_groups) * 100,
						 total_groups);

			seq_printf(m,
				   "all_nodes groups(组数) %llu pages(页数) %llu avg_group_pages(平均组大小) %llu.%02llu\n",
				   (unsigned long long)total_groups,
				   (unsigned long long)total_group_pages,
				   (unsigned long long)avg_int,
				   (unsigned long long)avg_frac);
		} else {
			seq_puts(m,
				 "all_nodes groups(组数) 0 pages(页数) 0 avg_group_pages(平均组大小) 0.00\n");
		}
	}
#endif
	seq_printf(m, "next_win_id(下一个窗口ID, 竞态只读) %u\n",
		   (unsigned int)atomic_read(&kagg_win_next_atomic) & 0xFFFFu);
	return 0;
}

static int kagg_summary_open(struct inode *inode, struct file *file)
{
	return single_open(file, kagg_summary_show, NULL);
}

static const struct file_operations kagg_summary_fops = {
	.owner		= THIS_MODULE,
	.open		= kagg_summary_open,
	.read		= seq_read,
	.llseek		= seq_lseek,
	.release	= single_release,
};

static void __init kagg_debugfs_register(void)
{
	struct dentry *d;

	if (!debugfs_initialized())
		return;
	d = debugfs_create_dir("kaggswapd", NULL);
	if (IS_ERR_OR_NULL(d))
		return;
	debugfs_create_file("summary", 0444, d, NULL, &kagg_summary_fops);
}
#endif /* CONFIG_KAGGSWAPD_DEBUG */

/**
 * kagg_try_mark_window - append @win_id unless this page was already stamped
 * in the same window (dedup across LRU seeds and VMA expansion).
 */
static void kagg_try_mark_window(struct page *page, u16 win_id, bool require_lru)
{
	struct page_ext_agg *agg;
	struct page_ext_agg_data *d;

	page = compound_head(page);

	if (unlikely(!pfn_valid(page_to_pfn(page))))
		return;
	if (require_lru && !PageLRU(page))
		return;
	if (PageHuge(page) || unlikely(!page_evictable(page)))
		return;
	if (!PageAnon(page))
		return;

	agg = lookup_page_ext_agg(page);
	if (!agg)
		return;
	d = page_ext_agg_ensure_data(agg);
	if (!d)
		return;
	/*
	 * LRU scan drops lru_lock between get_page and here; free can clear
	 * agg->data under dual_path_page_ext_prepare_free. Re-check linkage.
	 */
	if (unlikely(READ_ONCE(agg->data) != d))
		return;
	if (READ_ONCE(d->last_kagg_win) == win_id) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
		atomic64_inc(&kagg_dbg_dedup_same_win);
#endif
		return;
	}
	if (unlikely(READ_ONCE(agg->data) != d))
		return;

	page_ext_agg_push_window_into(d, win_id);
	WRITE_ONCE(d->last_kagg_win, win_id);
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_inc(&kagg_dbg_pages_stamped);
#endif
}

/*
 * Default sampler: LRU-visible anon pages only.
 */
void kagg_default_sample_mark(struct page *page, u16 win_id, void *priv)
{
	(void)priv;
	kagg_try_mark_window(page, win_id, true);
}

static unsigned long kagg_bounded_max_scan(unsigned long total,
					  unsigned int scan_pct)
{
	unsigned long capped;

	WARN_ON_ONCE(!scan_pct || scan_pct > 100);

	if (!total)
		return 0;
	capped = (total * (unsigned long)scan_pct + 99UL) / 100UL;
	if (!capped)
		capped = 1;
	return min(total, capped);
}

static int kagg_vma_pmd_entry(pmd_t *pmd, unsigned long addr,
			      unsigned long end, struct mm_walk *walk)
{
	struct vm_area_struct *vma = walk->vma;
	spinlock_t *ptl;
	pte_t *ptep;
	struct page *page;
	u16 win_id = *(u16 *)walk->private;

#ifdef CONFIG_TRANSPARENT_HUGEPAGE
	ptl = pmd_trans_huge_lock(pmd, vma);
	if (ptl) {
		if (pmd_present(*pmd)) {
			page = pmd_page(*pmd);
			kagg_try_mark_window(page, win_id, false);
		}
		spin_unlock(ptl);
		return 0;
	}
#endif
	if (pmd_trans_unstable(pmd))
		return 0;

	ptep = pte_offset_map_lock(walk->mm, pmd, addr, &ptl);
	for (; addr < end; addr += PAGE_SIZE, ptep++) {
		pte_t pte = *ptep;

		if (!pte_present(pte))
			continue;
		page = vm_normal_page(vma, addr, pte);
		if (!page)
			continue;
		kagg_try_mark_window(page, win_id, false);
	}
	pte_unmap_unlock(ptep - 1, ptl);
	cond_resched();
	return 0;
}

static const struct mm_walk_ops kagg_vma_walk_ops = {
	.pmd_entry	= kagg_vma_pmd_entry,
};

/*
 * From one inactive-LRU seed, expand every anonymous VMA in its anon_interval
 * tree slice and mark all present anon pages in those VMAs (dedup by win_id).
 */
static void kagg_expand_mappings_for_seed(struct page *page, u16 win_id)
{
	struct anon_vma *anon_vma;
	struct anon_vma_chain *avc;
	unsigned long pgoff_start, pgoff_end;

	page = compound_head(page);
	if (!PageAnon(page))
		return;

	anon_vma = page_get_anon_vma(page);
	if (!anon_vma)
		return;

	pgoff_start = page_to_pgoff(page);
	pgoff_end = pgoff_start + thp_nr_pages(page) - 1;

	anon_vma_lock_read(anon_vma);
	anon_vma_interval_tree_foreach(avc, &anon_vma->rb_root, pgoff_start,
				       pgoff_end) {
		struct vm_area_struct *vma = avc->vma;
		struct mm_struct *mm = vma->vm_mm;

		if (!vma_is_anonymous(vma))
			continue;
		if (!mm)
			continue;

		if (!mmap_read_trylock(mm))
			continue;

		walk_page_vma(vma, &kagg_vma_walk_ops, &win_id);
		mmap_read_unlock(mm);
	}
	anon_vma_unlock_read(anon_vma);
	put_anon_vma(anon_vma);
}

/*
 * VMA expand mode:
 * take a tail seed batch sized by inactive anon percentage, then fan out via
 * RMAP/VMA walk.
 */
static void kagg_scan_inactive_vma_expand(struct lruvec *lruvec,
					  unsigned long inactive_sz,
					  unsigned int scan_pct,
					  u16 win_id)
{
	pg_data_t *pgdat = lruvec_pgdat(lruvec);
	struct list_head *head = &lruvec->lists[LRU_INACTIVE_ANON];
	struct list_head *pos;
	unsigned int nr_seeds = 0, i;
	unsigned int seed_target;

	seed_target = kagg_percent_target(inactive_sz, scan_pct,
					  KAGG_VMA_SEED_MAX);
	if (!seed_target)
		return;

	spin_lock_irq(&pgdat->lru_lock);

	pos = head->prev;
	while (nr_seeds < seed_target && pos != head) {
		struct page *page = lru_to_page(pos);
		struct list_head *prev_pos = pos->prev;

		page = compound_head(page);
		if (!PageAnon(page) || PageHuge(page) || !PageLRU(page)) {
			pos = prev_pos;
			continue;
		}
		if (unlikely(!page_evictable(page))) {
			pos = prev_pos;
			continue;
		}
		if (!page_mapped(page)) {
			pos = prev_pos;
			continue;
		}
		if (!get_page_unless_zero(page)) {
			pos = prev_pos;
			continue;
		}
		kagg_vma_seed_buf[nr_seeds++] = page;
		pos = prev_pos;
	}

	spin_unlock_irq(&pgdat->lru_lock);

	for (i = 0; i < nr_seeds; i++) {
		kagg_expand_mappings_for_seed(kagg_vma_seed_buf[i], win_id);
		put_page(kagg_vma_seed_buf[i]);
	}
}

static void kagg_scan_lru_bounded(struct lruvec *lruvec, enum lru_list lruid,
				  unsigned long max_scan,
				  u16 win_id,
				  kagg_page_action_fn action_fn,
				  void *priv)
{
	struct list_head *head = &lruvec->lists[lruid];
	struct list_head *pos;
	unsigned long scanned = 0;
	pg_data_t *pgdat = lruvec_pgdat(lruvec);

	if (!max_scan || !action_fn)
		return;

	spin_lock_irq(&pgdat->lru_lock);
	pos = head->next;
	while (scanned < max_scan && pos != head) {
		struct page *page = lru_to_page(pos);
		struct list_head *next_pos = pos->next;

		scanned++;
		page = compound_head(page);
		if (get_page_unless_zero(page)) {
			/* Refcount must be visible before we drop lru_lock vs. free path. */
			smp_mb();
			spin_unlock_irq(&pgdat->lru_lock);
			action_fn(page, win_id, priv);
			put_page(page);
			cond_resched();
			spin_lock_irq(&pgdat->lru_lock);
		}
		pos = next_pos;
	}
	spin_unlock_irq(&pgdat->lru_lock);
}

static void kagg_sample_scan_lruvec(struct lruvec *lruvec,
				    enum kagg_scan_mode mode,
				    unsigned int scan_pct, u16 win_id,
				    kagg_page_action_fn action_fn,
				    void *priv)
{
	unsigned long inactive_sz, active_sz, max_scan;

	inactive_sz = lruvec_lru_size(lruvec, LRU_INACTIVE_ANON, MAX_NR_ZONES - 1);
	active_sz = lruvec_lru_size(lruvec, LRU_ACTIVE_ANON, MAX_NR_ZONES - 1);

	switch (mode) {
	case KAGG_SCAN_INACTIVE_PCT:
		max_scan = kagg_bounded_max_scan(inactive_sz, scan_pct);
		kagg_scan_lru_bounded(lruvec, LRU_INACTIVE_ANON, max_scan,
				      win_id, action_fn, priv);
		break;

	case KAGG_SCAN_ACTIVE_PCT:
		max_scan = kagg_bounded_max_scan(active_sz, scan_pct);
		kagg_scan_lru_bounded(lruvec, LRU_ACTIVE_ANON, max_scan,
				      win_id, action_fn, priv);
		break;

	case KAGG_SCAN_ALL_ANON: {
		kagg_scan_lru_bounded(lruvec, LRU_INACTIVE_ANON, ULONG_MAX,
				      win_id, action_fn, priv);
		kagg_scan_lru_bounded(lruvec, LRU_ACTIVE_ANON, ULONG_MAX,
				      win_id, action_fn, priv);
		break;
	}

	case KAGG_SCAN_INACTIVE_VMA_EXPAND_PCT:
		kagg_scan_inactive_vma_expand(lruvec, inactive_sz, scan_pct,
					      win_id);
		break;

	default:
		/*
		 * KAGG_SCAN_MODE_NR and clamp mistakes: mode is pinned to
		 * [0, KAGG_SCAN_MODE_NR - 1] above; keep default for -Wswitch.
		 */
		break;
	}
}

/*
 * Sampling plane entrypoint:
 * mode 0/1/2 are LRU sampling variants; mode 3 is seed-based VMA expansion.
 * Per pgdat, walk all memcg lruvecs to avoid only sampling root memcg pages.
 */
void kagg_sample_scan_pgdat(struct pglist_data *pgdat,
			    enum kagg_scan_mode scan_mode,
			    unsigned int scan_pct, u16 win_id,
			    kagg_page_action_fn action_fn,
			    void *priv)
{
	enum kagg_scan_mode mode;

	if (!dual_path_swap_enabled())
		return;

	if (!pgdat || unlikely(!num_online_nodes()))
		return;

	if (!action_fn)
		action_fn = kagg_default_sample_mark;

	mode = (enum kagg_scan_mode)
		clamp_val((int)scan_mode, 0, (int)KAGG_SCAN_MODE_NR - 1);

#ifdef CONFIG_MEMCG
	if (!mem_cgroup_disabled()) {
		struct mem_cgroup *memcg = NULL;

		while ((memcg = mem_cgroup_iter(NULL, memcg, NULL))) {
			struct lruvec *lruvec;

			if (!mem_cgroup_online(memcg))
				continue;
			lruvec = mem_cgroup_lruvec(memcg, pgdat);
			kagg_sample_scan_lruvec(lruvec, mode, scan_pct, win_id,
						action_fn, priv);
		}
		return;
	}
#endif
	kagg_sample_scan_lruvec(mem_cgroup_lruvec(NULL, pgdat), mode, scan_pct,
				win_id, action_fn, priv);
}

static void kagg_miner_init_buckets(struct hlist_head *buckets)
{
	unsigned int i;

	for (i = 0; i < KAGG_MINER_HASH_SIZE; i++)
		INIT_HLIST_HEAD(&buckets[i]);
}

static unsigned int kagg_miner_hash_prefix(const struct page_ext_agg_data *d)
{
	u32 keys[KAGG_MINER_PREFIX];
	unsigned int cnt = 0, seen, step;
	int head;

	head = READ_ONCE(d->head);
	for (step = 0; step < MAX_PAGE_EXT_AGG_WINDOW && cnt < KAGG_MINER_PREFIX;
	     step++) {
		u16 id;
		unsigned int k;
		bool dup = false;

		head--;
		if (head < 0)
			head = MAX_PAGE_EXT_AGG_WINDOW - 1;
		id = READ_ONCE(d->window_ids[head]);
		if (!id)
			continue;
		for (k = 0; k < cnt; k++) {
			if (keys[k] == (u32)id) {
				dup = true;
				break;
			}
		}
		if (!dup)
			keys[cnt++] = (u32)id;
	}
	if (!cnt)
		return 0;

	for (seen = 1; seen < cnt; seen++) {
		u32 v = keys[seen];
		int j = (int)seen - 1;

		while (j >= 0 && keys[j] > v) {
			keys[j + 1] = keys[j];
			j--;
		}
		keys[j + 1] = v;
	}

	return jhash2(keys, cnt, 0) & (KAGG_MINER_HASH_SIZE - 1);
}

static unsigned int kagg_windows_unique(const struct page_ext_agg_data *d,
					u16 *uniq)
{
	unsigned int i, cnt = 0;

	for (i = 0; i < MAX_PAGE_EXT_AGG_WINDOW; i++) {
		u16 id = READ_ONCE(d->window_ids[i]);
		unsigned int j;
		bool dup = false;

		if (!id)
			continue;
		for (j = 0; j < cnt; j++) {
			if (uniq[j] == id) {
				dup = true;
				break;
			}
		}
		if (!dup)
			uniq[cnt++] = id;
	}
	if (cnt > 1U) {
		unsigned int seen, j;
		u16 v;

		for (seen = 1U; seen < cnt; seen++) {
			v = uniq[seen];
			j = seen;
			while (j > 0U && uniq[j - 1U] > v) {
				uniq[j] = uniq[j - 1U];
				j--;
			}
			uniq[j] = v;
		}
	}
	return cnt;
}

static bool kagg_miner_similar(const struct page_ext_agg_data *seed,
			       const struct page_ext_agg_data *cand,
			       unsigned int threshold_pct)
{
	u16 s[MAX_PAGE_EXT_AGG_WINDOW], c[MAX_PAGE_EXT_AGG_WINDOW];
	unsigned int ns, nc, i, j, inter = 0, denom;

	ns = kagg_windows_unique(seed, s);
	nc = kagg_windows_unique(cand, c);
	if (!ns || !nc)
		return false;

	i = 0;
	j = 0;
	while (i < ns && j < nc) {
		if (s[i] == c[j]) {
			inter++;
			i++;
			j++;
		} else if (s[i] < c[j]) {
			i++;
		} else {
			j++;
		}
	}
	denom = max(ns, nc);
	return inter * 100U >= threshold_pct * denom;
}

static void __maybe_unused kagg_miner_clear_old_groups(struct lruvec *lruvec)
{
	struct page_group *grp, *tmp;
	pg_data_t *pgdat = lruvec_pgdat(lruvec);
	LIST_HEAD(local_free);
	unsigned int cleared = 0;
	unsigned int freed = 0;

	spin_lock_irq(&pgdat->lru_lock);
	list_for_each_entry_safe(grp, tmp, &lruvec->agg_list, lru) {
		lruvec_page_group_del_init(grp);
		list_add(&grp->lru, &local_free);
		cleared++;
	}
	spin_unlock_irq(&pgdat->lru_lock);
	list_for_each_entry_safe(grp, tmp, &local_free, lru) {
		list_del_init(&grp->lru);
		kfree(grp);
		freed++;
		if (!(freed % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
	}
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_add(cleared, &kagg_dbg_miner_groups_cleared);
#endif
}

static int kagg_miner_iso_pair_cmp(const void *a, const void *b)
{
	const struct kagg_miner_iso_pair *pa = a;
	const struct kagg_miner_iso_pair *pb = b;

	if (pa->d < pb->d)
		return -1;
	if (pa->d > pb->d)
		return 1;
	return 0;
}

/*
 * Fill kagg_miner_iso_map[], sorted by page_ext_agg_data * for lookup in emit.
 */
static unsigned int kagg_miner_build_iso_sorted_map(struct list_head *all_isolated)
{
	struct page *page;
	unsigned int n = 0;

	list_for_each_entry(page, all_isolated, lru) {
		struct page *head = compound_head(page);
		struct page_ext_agg *agg = lookup_page_ext_agg(head);
		struct page_ext_agg_data *d = page_ext_agg_get_data_maybe(agg);

		if (!d)
			continue;
		if (unlikely(n >= KAGG_MINER_VIEW_MAX)) {
			WARN_ON_ONCE(1);
			break;
		}
		kagg_miner_iso_map[n].d = d;
		kagg_miner_iso_map[n].page = head;
		n++;
		if (!(n % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
	}
	if (n > 1U)
		sort(kagg_miner_iso_map, (size_t)n,
		     sizeof(kagg_miner_iso_map[0]), kagg_miner_iso_pair_cmp, NULL);
	return n;
}

/* Strip miner hash nodes only; isolated pages remain on @all_isolated for putback. */
static void kagg_miner_bucket_strip_nodes(struct hlist_head *bucket)
{
	struct page_ext_agg_data *d;
	struct hlist_node *tmp;

	hlist_for_each_entry_safe(d, tmp, bucket, miner_node)
		hlist_del_init(&d->miner_node);
}

static struct page *kagg_miner_iso_map_lookup(struct page_ext_agg_data *want)
{
	unsigned int lo = 0, hi = kagg_miner_iso_nmap;

	while (lo < hi) {
		unsigned int mid = lo + ((hi - lo) >> 1);
		struct page_ext_agg_data *md = kagg_miner_iso_map[mid].d;

		if (md == want)
			return kagg_miner_iso_map[mid].page;
		if ((uintptr_t)md < (uintptr_t)want)
			lo = mid + 1U;
		else
			hi = mid;
	}
	return NULL;
}

static void kagg_miner_rebucket_agg_data(struct hlist_head *buckets,
					 struct page_ext_agg_data **memd,
					 unsigned int nmem)
{
	unsigned int j;

	for (j = 0; j < nmem; j++) {
		struct page_ext_agg_data *d = memd[j];
		unsigned int idx = kagg_miner_hash_prefix(d);

		if (!hlist_unhashed(&d->miner_node))
			hlist_del_init(&d->miner_node);
		hlist_add_head(&d->miner_node, &buckets[idx]);
	}
}

static bool kagg_miner_emit_cluster(struct lruvec *lruvec,
				    struct page_ext_agg_data **memd,
				    unsigned int nmem,
				    struct kagg_memcg_meta *meta)
{
	struct page_group *grp;
	pg_data_t *pgdat = lruvec_pgdat(lruvec);
	unsigned int i;
	struct page **pages_mem = kagg_miner_emit_pages_buf;

	if (!nmem)
		return false;

	for (i = 0; i < nmem; i++) {
		struct page_ext_agg_data *d = memd[i];
		struct page *page;

		if (unlikely(!d || READ_ONCE(d->group)))
			return false;
		page = kagg_miner_iso_map_lookup(d);
		if (unlikely(!page))
			return false;
		pages_mem[i] = page;
	}

	grp = kzalloc(sizeof(*grp), GFP_KERNEL);
	if (!grp)
		return false;
	page_group_init(grp);
	grp->nr_pages = nmem;
	grp->creation_time = (u64)jiffies;
	grp->hotness = (u16)min_t(unsigned int, nmem, 0xFFFFU);

	spin_lock_irq(&pgdat->lru_lock);
	lruvec_page_group_add_tail(lruvec, grp);
	spin_unlock_irq(&pgdat->lru_lock);

	/*
	 * Moving up to nmem pages under lru_lock + local IRQ off was long enough
	 * to provoke rcu_sched stalls. The aggregate list head only needs a short
	 * critical section; page_list links are private until d->group publishes.
	 */
	for (i = 0; i < nmem; i++) {
		struct page *page = pages_mem[i];
		struct page_ext_agg_data *d = memd[i];

		list_del_init(&page->lru);
		list_add_tail(&page->lru, &grp->page_list);
		smp_wmb();
		WRITE_ONCE(d->group, grp);
		if (!((i + 1U) % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
	}
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_inc(&kagg_dbg_miner_clusters_emitted);
	atomic64_add(nmem, &kagg_dbg_miner_cluster_pages);
	atomic64_add(nmem, &kagg_dbg_miner_members_linked);
	if (meta) {
		atomic64_inc(&meta->clusters_emitted);
		atomic64_add(nmem, &meta->cluster_pages);
	}
	pr_info_ratelimited("kaggswapd: page_group %p: linked %u member pages (node %d)\n",
			      grp, nmem, pgdat->node_id);
#endif
	return true;
}

static unsigned int kagg_miner_isolate_chunk(struct lruvec *lruvec,
					     unsigned int chunk_target,
					     struct list_head *local_list,
					     struct kagg_memcg_meta *meta)
{
	pg_data_t *pgdat = lruvec_pgdat(lruvec);
	unsigned int nid = pgdat->node_id;
	struct list_head *head = &lruvec->lists[LRU_INACTIVE_ANON];
	struct list_head *pos, *next;
	struct page *cands[KAGG_MINER_BATCH_MAX];
	unsigned int nr_cands = 0, i;
	unsigned int isolated_ok = 0;
	unsigned int start_steps;
	unsigned int skip;
	unsigned int consumed;
	unsigned int total_walked = 0;
	unsigned long inactive_nr;
	unsigned long max_scan;
	unsigned int *cursor_lru_steps;

	if (!chunk_target)
		return 0;
	if (chunk_target > KAGG_MINER_BATCH_MAX)
		chunk_target = KAGG_MINER_BATCH_MAX;
	cursor_lru_steps = meta ? meta->cursor_lru_steps : kagg_root_meta.cursor_lru_steps;

	spin_lock_irq(&pgdat->lru_lock);

	inactive_nr = lruvec_lru_size(lruvec, LRU_INACTIVE_ANON, MAX_NR_ZONES - 1);
	max_scan = (unsigned long)chunk_target * 64UL;
	if (max_scan < inactive_nr * 2UL)
		max_scan = inactive_nr * 2UL;
	if (max_scan > (unsigned long)KAGG_MINER_ISO_SCAN_ABS_MAX)
		max_scan = (unsigned long)KAGG_MINER_ISO_SCAN_ABS_MAX;
	if (max_scan < (unsigned long)chunk_target * 8UL)
		max_scan = (unsigned long)chunk_target * 8UL;
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_inc(&kagg_dbg_miner_iso_chunks);
	atomic64_add(chunk_target, &kagg_dbg_miner_iso_target_slots);
#endif

	start_steps = READ_ONCE(cursor_lru_steps[nid]);
	skip = start_steps;
	pos = head->next;
	while (skip && pos != head) {
		pos = pos->next;
		skip--;
	}
	if (skip) {
		/* Cursor offset past list length; restart from head. */
		WRITE_ONCE(cursor_lru_steps[nid], 0);
		consumed = 0;
		pos = head->next;
	} else {
		consumed = start_steps;
	}

	/*
	 * Keep scanning (wrap at tail) until chunk_target references are held or
	 * we hit max_scan, so one chunk is not starved by an unlucky cursor.
	 */
	while (nr_cands < chunk_target && total_walked < max_scan) {
		if (pos == head)
			pos = head->next;
		if (pos == head)
			break;
		for (; pos != head && nr_cands < chunk_target &&
		       total_walked < max_scan; pos = next) {
			struct page *page = compound_head(lru_to_page(pos));
			next = pos->next;

			total_walked++;

			if (!PageAnon(page)) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
				atomic64_inc(&kagg_dbg_miner_iso_skip_non_anon);
#endif
				continue;
			}
			if (PageHuge(page)) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
				atomic64_inc(&kagg_dbg_miner_iso_skip_huge);
#endif
				continue;
			}
			if (!PageLRU(page)) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
				atomic64_inc(&kagg_dbg_miner_iso_skip_not_lru_walk);
#endif
				continue;
			}
			if (unlikely(!page_evictable(page))) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
				atomic64_inc(&kagg_dbg_miner_iso_skip_not_evictable);
#endif
				continue;
			}
			if (!get_page_unless_zero(page)) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
				atomic64_inc(&kagg_dbg_miner_iso_skip_no_ref);
#endif
				continue;
			}
			cands[nr_cands++] = page;
		}
	}
	if (pos == head) {
		WRITE_ONCE(cursor_lru_steps[nid], 0);
	} else {
		WRITE_ONCE(cursor_lru_steps[nid], consumed + total_walked);
	}
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_add(total_walked, &kagg_dbg_miner_iso_walked_slots);
	if (nr_cands < chunk_target) {
		atomic64_inc(&kagg_dbg_miner_iso_chunk_underfill);
		if (total_walked >= max_scan)
			atomic64_inc(&kagg_dbg_miner_iso_chunk_maxscan_hit);
	}
#endif
	spin_unlock_irq(&pgdat->lru_lock);

	for (i = 0; i < nr_cands; i++) {
		struct page *page = cands[i];
		bool lrutest;
		int err;

#ifdef CONFIG_KAGGSWAPD_DEBUG
		atomic64_inc(&kagg_dbg_miner_iso_attempts);
#endif
		/*
		 * isolate_lru_page() only distinguishes success vs -EBUSY; split
		 * whether PageLRU was already clear before calling (race after LRU
		 * walk unlocked) vs still LRU (lost under pgdat->lru_lock inside).
		 */
		lrutest = !!PageLRU(page);
		err = isolate_lru_page(page);
		if (!err) {
			list_add_tail(&page->lru, local_list);
			isolated_ok++;
		} else {
#ifdef CONFIG_KAGGSWAPD_DEBUG
			if (!lrutest)
				atomic64_inc(&kagg_dbg_miner_iso_fail_not_lru);
			else
				atomic64_inc(&kagg_dbg_miner_iso_fail_race);
#endif
			put_page(page);
		}
		if (!((i + 1) % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
	}
	if (nr_cands % KAGG_MINER_RESCHED_QUOTA)
		cond_resched();
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_add(isolated_ok, &kagg_dbg_miner_isolated_pages);
	if (meta)
		atomic64_add(isolated_ok, &meta->isolated_pages);
#endif
	return isolated_ok;
}

/*
 * Bucketize one isolated batch (no locks; pages are off-LRU).
 */
static void kagg_miner_bucketize_batch(struct list_head *batch_list,
				       struct hlist_head *buckets,
				       unsigned int *work)
{
	struct page *page;

	list_for_each_entry(page, batch_list, lru) {
		struct page_ext_agg *agg;
		struct page_ext_agg_data *d;
		unsigned int idx;

		agg = lookup_page_ext_agg(page);
		if (!agg)
			continue;
		d = page_ext_agg_ensure_data(agg);
		if (!d)
			continue;
		if (!hlist_unhashed(&d->miner_node))
			hlist_del_init(&d->miner_node);
		idx = kagg_miner_hash_prefix(d);
		hlist_add_head(&d->miner_node, &buckets[idx]);
		(*work)++;
		if (!((*work) % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
	}
}

static void kagg_miner_cluster_and_putback(struct lruvec *lruvec,
					   struct hlist_head *buckets,
					   struct list_head *all_isolated,
					   struct kagg_memcg_meta *meta)
{
	struct page *page, *next;
	unsigned int i, work = 0;
	unsigned int sim_pct = (unsigned int)clamp_val(
		READ_ONCE(sysctl_dual_path_miner_similarity_pct), 1, 100);
	unsigned int min_cluster = (unsigned int)clamp_val(
		READ_ONCE(sysctl_dual_path_miner_min_cluster), 1,
		KAGG_MINER_CLUSTER_MAX);
	struct page_ext_agg_data **memd = kagg_miner_cluster_memd_buf;

	kagg_miner_iso_nmap = kagg_miner_build_iso_sorted_map(all_isolated);

	for (i = 0; i < KAGG_MINER_HASH_SIZE; i++) {
		struct hlist_head *bucket = &buckets[i];
		unsigned int bucket_seeds = 0;

		while (!hlist_empty(bucket)) {
			struct page_ext_agg_data *seed;
			struct page_ext_agg_data *cand;
			struct hlist_node *tmp;
			unsigned int nmem = 0;
			unsigned int j;

			if (++bucket_seeds > KAGG_MINER_BUCKET_SEED_MAX) {
				kagg_miner_bucket_strip_nodes(bucket);
#ifdef CONFIG_KAGGSWAPD_DEBUG
				atomic64_inc(&kagg_dbg_miner_bucket_trunc);
				if (meta)
					atomic64_inc(&meta->bucket_trunc);
#endif
				pr_warn_ratelimited(
					"kaggswapd: miner hash bucket seed cap (%u); stripping remaining nodes\n",
					(unsigned int)KAGG_MINER_BUCKET_SEED_MAX);
				break;
			}

			work++;
			if (!(work % KAGG_MINER_RESCHED_QUOTA))
				cond_resched();

			seed = hlist_entry(bucket->first,
					struct page_ext_agg_data, miner_node);
			hlist_del_init(&seed->miner_node);
			memd[nmem++] = seed;

			hlist_for_each_entry_safe(cand, tmp, bucket, miner_node) {
				work++;
				if (!(work % KAGG_MINER_RESCHED_QUOTA))
					cond_resched();
				if (nmem >= KAGG_MINER_CLUSTER_MAX)
					break;
				if (!kagg_miner_similar(seed, cand, sim_pct))
					continue;
				hlist_del_init(&cand->miner_node);
				memd[nmem++] = cand;
			}

			if (nmem < min_cluster) {
				for (j = 0; j < nmem; j++) {
					struct page_ext_agg_data *d = memd[j];
					unsigned int idx = kagg_miner_hash_prefix(d);

					hlist_add_head(&d->miner_node, &buckets[idx]);
				}
				work++;
				if (!(work % KAGG_MINER_RESCHED_QUOTA))
					cond_resched();
				continue;
			}

			if (!kagg_miner_emit_cluster(lruvec, memd, nmem, meta))
				kagg_miner_rebucket_agg_data(buckets, memd, nmem);
			work++;
			if (!(work % KAGG_MINER_RESCHED_QUOTA))
				cond_resched();
		}
		cond_resched();
	}

	list_for_each_entry_safe(page, next, all_isolated, lru) {
		struct page_ext_agg *agg = lookup_page_ext_agg(page);
		struct page_ext_agg_data *d = page_ext_agg_get_data_maybe(agg);

		if (d && READ_ONCE(d->group))
			continue;
		if (d && !hlist_unhashed(&d->miner_node))
			hlist_del_init(&d->miner_node);
		list_del_init(&page->lru);
		putback_lru_page(page);
		put_page(page);
		work++;
		if (!(work % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
#ifdef CONFIG_KAGGSWAPD_DEBUG
		atomic64_inc(&kagg_dbg_miner_putback_pages);
		if (meta)
			atomic64_inc(&meta->putback_pages);
#endif
	}
}

static void kagg_miner_scan_lruvec(struct lruvec *lruvec, unsigned int seed_pct)
{
	pg_data_t *pgdat = lruvec_pgdat(lruvec);
	struct hlist_head *buckets = kagg_miner_bucket_buf;
	LIST_HEAD(all_isolated);
	unsigned long inactive_sz;
	unsigned int view_target;
	unsigned int attempt_budget = 0;
	unsigned int work = 0;
	struct kagg_memcg_meta *meta = NULL;

	if (!pgdat)
		return;
	inactive_sz = lruvec_lru_size(lruvec, LRU_INACTIVE_ANON, MAX_NR_ZONES - 1);
	view_target = kagg_percent_target(inactive_sz, seed_pct, KAGG_MINER_VIEW_MAX);
	if (!view_target)
		return;
	meta = kagg_memcg_meta_get(lruvec_memcg(lruvec), true);
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_add(view_target, &kagg_dbg_miner_view_target_pages);
	if (view_target == KAGG_MINER_VIEW_MAX)
		atomic64_inc(&kagg_dbg_miner_view_target_cap_hits);
	if (meta) {
		atomic64_inc(&meta->miner_rounds);
		atomic64_add(view_target, &meta->view_target_pages);
	}
#endif

	kagg_miner_init_buckets(buckets);
	if (meta)
		WRITE_ONCE(meta->cursor_lru_steps[pgdat->node_id], 0);
	/*
	 * One miner round consumes a fixed "view_target" worth of small-batch
	 * attempts (chunk-sized), even if some batches transiently isolate 0
	 * pages. Do not end the round early on an empty batch alone.
	 */
	while (attempt_budget < view_target) {
		LIST_HEAD(batch_list);
		unsigned int chunk = min_t(unsigned int, KAGG_MINER_BATCH_MAX,
					   view_target - attempt_budget);
		unsigned int got;

		attempt_budget += chunk;
		got = kagg_miner_isolate_chunk(lruvec, chunk, &batch_list, meta);
		if (got) {
			kagg_miner_bucketize_batch(&batch_list, buckets, &work);
			list_splice_tail_init(&batch_list, &all_isolated);
		}
		cond_resched();
	}
	if (!list_empty(&all_isolated))
		kagg_miner_cluster_and_putback(lruvec, buckets, &all_isolated, meta);
}

/* Mining plane entrypoint for one pgdat: walk all memcg lruvecs. */
static void kagg_miner_scan_pgdat(struct pglist_data *pgdat,
				  unsigned int seed_pct)
{
#ifdef CONFIG_MEMCG
	if (!mem_cgroup_disabled()) {
		struct mem_cgroup *memcg = NULL;

		while ((memcg = mem_cgroup_iter(NULL, memcg, NULL))) {
			struct lruvec *lruvec;

			if (!mem_cgroup_online(memcg))
				continue;
			lruvec = mem_cgroup_lruvec(memcg, pgdat);
			kagg_miner_scan_lruvec(lruvec, seed_pct);
		}
		return;
	}
#endif
	kagg_miner_scan_lruvec(mem_cgroup_lruvec(NULL, pgdat), seed_pct);
}

static int kaggswapd_fn(void *unused)
{
	enum kagg_scan_mode mode;
	unsigned int pct;
	unsigned int interval;
	unsigned int miner_interval;
	unsigned long next_miner_deadline = jiffies;
	u16 win_id;
	struct pglist_data *pgdat;

	(void)unused;

	set_freezable();
	for (;;) {
		if (kthread_should_stop())
			break;

		if (!dual_path_swap_enabled()) {
			set_current_state(TASK_INTERRUPTIBLE);
			if (HZ)
				schedule_timeout(msecs_to_jiffies(1000UL));
			else
				schedule_timeout(round_jiffies_up_relative(1));
			try_to_freeze();
			continue;
		}

		mode = (enum kagg_scan_mode)
			clamp_val(sysctl_dual_path_scan_mode, 0,
				  (int)KAGG_SCAN_MODE_NR - 1);

		pct = (unsigned int)clamp_val(sysctl_dual_path_scan_pct, 1, 100);
		interval = (unsigned int)clamp_val(sysctl_dual_path_scan_interval_ms,
						     10, 600000);
		miner_interval = (unsigned int)clamp_val(
			sysctl_dual_path_miner_interval_ms, 1000, 36000);

		win_id = kagg_next_window_id();

		for_each_online_pgdat(pgdat) {
			kagg_sample_scan_pgdat(pgdat, mode, pct, win_id,
					      kagg_default_sample_mark, NULL);
		}

		if (READ_ONCE(sysctl_dual_path_miner_enable) &&
		    time_after_eq(jiffies, next_miner_deadline)) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
			atomic64_inc(&kagg_dbg_miner_block_start);
#endif
			for_each_online_pgdat(pgdat)
				kagg_miner_scan_pgdat(pgdat, pct);
			next_miner_deadline = jiffies +
				msecs_to_jiffies((unsigned long)miner_interval);
#ifdef CONFIG_KAGGSWAPD_DEBUG
			atomic64_inc(&kagg_dbg_miner_rounds);
#endif
		}

#ifdef CONFIG_KAGGSWAPD_DEBUG
		atomic64_inc(&kagg_dbg_scan_rounds);
#endif

		set_current_state(TASK_INTERRUPTIBLE);
		schedule_timeout(msecs_to_jiffies(interval));
		try_to_freeze();
	}

	return 0;
}

static int __init kagg_scan_init(void)
{
	struct task_struct *tsk;
#ifdef CONFIG_MEMCG
	unsigned int i;
#endif

#ifdef CONFIG_MEMCG
	for (i = 0; i < KAGG_MEMCG_META_HASH_SIZE; i++)
		INIT_HLIST_HEAD(&kagg_memcg_meta_hash[i]);
#endif

	tsk = kthread_run(kaggswapd_fn, NULL, "kaggswapd");
	if (IS_ERR(tsk)) {
		pr_warn("kaggswapd: start failed (%ld)\n", PTR_ERR(tsk));
		return PTR_ERR(tsk);
	}

#ifdef CONFIG_KAGGSWAPD_DEBUG
	kagg_debugfs_register();
#endif
	return 0;
}
late_initcall(kagg_scan_init);

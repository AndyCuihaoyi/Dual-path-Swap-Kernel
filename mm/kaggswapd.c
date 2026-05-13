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
#include <linux/spinlock.h>
#include <linux/swap.h>

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
#define KAGG_MINER_VIEW_MAX	4096U

static atomic_t kagg_win_next_atomic = ATOMIC_INIT(1);
static unsigned long kagg_miner_cursor_pfn[MAX_NUMNODES];
static bool kagg_miner_cursor_valid[MAX_NUMNODES];
/*
 * Scratch buffers for kaggswapd thread to avoid large on-stack arrays.
 * kaggswapd is single-threaded in this tree.
 */
static struct page *kagg_vma_seed_buf[KAGG_VMA_SEED_MAX];
static struct hlist_head kagg_miner_bucket_buf[KAGG_MINER_HASH_SIZE];

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

#define KAGG_DBG_RING	32

static atomic64_t kagg_dbg_scan_rounds;
static atomic64_t kagg_dbg_pages_stamped;
static atomic64_t kagg_dbg_dedup_same_win;
static atomic64_t kagg_dbg_miner_rounds;
static atomic64_t kagg_dbg_miner_isolated_pages;
static atomic64_t kagg_dbg_miner_clusters_emitted;
static atomic64_t kagg_dbg_miner_cluster_pages;
static atomic64_t kagg_dbg_miner_putback_pages;
static atomic64_t kagg_dbg_miner_groups_cleared;

static atomic_t kagg_dbg_wr_seq;
static unsigned long kagg_dbg_recent_pfn[KAGG_DBG_RING];

static void kagg_dbg_note_stamp(unsigned long pfn)
{
	unsigned s = (unsigned)atomic_inc_return(&kagg_dbg_wr_seq);

	WRITE_ONCE(kagg_dbg_recent_pfn[(s - 1U) % KAGG_DBG_RING], pfn);
}

static int kagg_summary_show(struct seq_file *m, void *v)
{
	unsigned w, count, k;

	(void)v;

	seq_printf(m, "scan_rounds %lld\n",
		   (long long)atomic64_read(&kagg_dbg_scan_rounds));
	seq_printf(m, "pages_stamped %lld\n",
		   (long long)atomic64_read(&kagg_dbg_pages_stamped));
	seq_printf(m, "dedup_same_window %lld\n",
		   (long long)atomic64_read(&kagg_dbg_dedup_same_win));
	seq_printf(m, "miner_rounds %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_rounds));
	seq_printf(m, "miner_isolated_pages %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_isolated_pages));
	seq_printf(m, "miner_clusters_emitted %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_clusters_emitted));
	seq_printf(m, "miner_cluster_pages %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_cluster_pages));
	seq_printf(m, "miner_putback_pages %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_putback_pages));
	seq_printf(m, "miner_groups_cleared %lld\n",
		   (long long)atomic64_read(&kagg_dbg_miner_groups_cleared));
	seq_printf(m, "mem_total_pages %lu\n", totalram_pages());
	seq_printf(m, "mem_anon_active_pages %lu\n",
		   global_node_page_state(NR_ACTIVE_ANON));
	seq_printf(m, "mem_anon_inactive_pages %lu\n",
		   global_node_page_state(NR_INACTIVE_ANON));
	seq_printf(m, "next_win_id %u (best-effort racy read)\n",
		   (unsigned int)atomic_read(&kagg_win_next_atomic) & 0xFFFFu);
	seq_puts(m, "\nagg_groups_per_node (best-effort):\n");
#ifdef CONFIG_DUAL_PATH_SWAP
	{
		struct pglist_data *pgdat;

		for_each_online_pgdat(pgdat) {
			struct lruvec *lruvec = mem_cgroup_lruvec(NULL, pgdat);
			struct page_group *grp;
			unsigned long flags;
			unsigned int groups = 0;
			u64 pages = 0;

			spin_lock_irqsave(&pgdat->lru_lock, flags);
			list_for_each_entry(grp, &lruvec->agg_list, lru) {
				groups++;
				pages += grp->nr_pages;
			}
			spin_unlock_irqrestore(&pgdat->lru_lock, flags);
			seq_printf(m, "node%d groups %u pages %llu\n",
				   pgdat->node_id, groups,
				   (unsigned long long)pages);
		}
	}
#endif
	seq_puts(m, "\nrecent_stamped_pfns (newest first, best-effort):\n");

	w = (unsigned)atomic_read(&kagg_dbg_wr_seq);
	count = min_t(unsigned, w, KAGG_DBG_RING);
	for (k = 0; k < count; k++) {
		unsigned idx = (w - 1U - k) % KAGG_DBG_RING;
		unsigned long pfn = READ_ONCE(kagg_dbg_recent_pfn[idx]);
		struct page *page;
		struct page_ext_agg *agg;
		struct page_ext_agg_data *d;
		int j;

		seq_printf(m, "pfn %#lx ", pfn);
		if (!pfn_valid(pfn)) {
			seq_puts(m, "invalid\n");
			continue;
		}
		page = pfn_to_online_page(pfn);
		if (!page) {
			seq_puts(m, "offline\n");
			continue;
		}
		page = compound_head(page);
		agg = lookup_page_ext_agg(page);
		if (!agg) {
			seq_puts(m, "no_agg\n");
			continue;
		}
		d = page_ext_agg_get_data_maybe(agg);
		if (!d) {
			seq_puts(m, "no_agg_data\n");
			continue;
		}
		seq_printf(m, "head %d last_win %u group %p win[",
			   d->head, (unsigned int)d->last_kagg_win, d->group);
		for (j = 0; j < MAX_PAGE_EXT_AGG_WINDOW; j++) {
			seq_printf(m, "%s%u", j ? "," : "",
				   (unsigned int)d->window_ids[j]);
		}
		seq_puts(m, "]\n");
	}
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
	if (READ_ONCE(d->last_kagg_win) == win_id) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
		atomic64_inc(&kagg_dbg_dedup_same_win);
#endif
		return;
	}

	page_ext_agg_push_window_into(d, win_id);
	WRITE_ONCE(d->last_kagg_win, win_id);
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_inc(&kagg_dbg_pages_stamped);
	kagg_dbg_note_stamp(page_to_pfn(page));
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

/*
 * Sampling plane entrypoint:
 * mode 0/1/2 are LRU sampling variants; mode 3 is seed-based VMA expansion.
 */
void kagg_sample_scan_pgdat(struct pglist_data *pgdat,
			    enum kagg_scan_mode scan_mode,
			    unsigned int scan_pct, u16 win_id,
			    kagg_page_action_fn action_fn,
			    void *priv)
{
	struct lruvec *lruvec;
	unsigned long inactive_sz, active_sz, max_scan;
	enum kagg_scan_mode mode;

	if (!dual_path_swap_enabled())
		return;

	if (!pgdat || unlikely(!num_online_nodes()))
		return;

	lruvec = mem_cgroup_lruvec(NULL, pgdat);

	if (!action_fn)
		action_fn = kagg_default_sample_mark;

	mode = (enum kagg_scan_mode)
		clamp_val((int)scan_mode, 0, (int)KAGG_SCAN_MODE_NR - 1);

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

	for (i = 0; i < ns; i++) {
		for (j = 0; j < nc; j++) {
			if (s[i] == c[j]) {
				inter++;
				break;
			}
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

static void kagg_miner_emit_cluster(struct lruvec *lruvec, unsigned int cnt)
{
	struct page_group *grp;
	pg_data_t *pgdat = lruvec_pgdat(lruvec);

	grp = kzalloc(sizeof(*grp), GFP_NOWAIT | __GFP_NOWARN);
	if (!grp)
		return;
	page_group_init(grp);
	grp->nr_pages = cnt;
	grp->creation_time = (u64)jiffies;
	grp->hotness = (u16)min_t(unsigned int, cnt, 0xFFFFU);

	spin_lock_irq(&pgdat->lru_lock);
	lruvec_page_group_add_tail(lruvec, grp);
	spin_unlock_irq(&pgdat->lru_lock);
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_inc(&kagg_dbg_miner_clusters_emitted);
	atomic64_add(cnt, &kagg_dbg_miner_cluster_pages);
#endif
}

static unsigned int kagg_miner_isolate_chunk(struct pglist_data *pgdat,
					     unsigned int chunk_target,
					     struct list_head *local_list)
{
	unsigned int nid = pgdat->node_id;
	struct lruvec *lruvec = mem_cgroup_lruvec(NULL, pgdat);
	struct list_head *head = &lruvec->lists[LRU_INACTIVE_ANON];
	struct list_head *pos, *next;
	struct page *cands[KAGG_MINER_BATCH_MAX];
	unsigned int nr_cands = 0, i;
	unsigned int isolated_ok = 0;
	unsigned long cursor_pfn = READ_ONCE(kagg_miner_cursor_pfn[nid]);
	bool cursor_valid = READ_ONCE(kagg_miner_cursor_valid[nid]);

	if (!chunk_target)
		return 0;
	if (chunk_target > KAGG_MINER_BATCH_MAX)
		chunk_target = KAGG_MINER_BATCH_MAX;

	spin_lock_irq(&pgdat->lru_lock);
	pos = head->next;
	if (cursor_valid) {
		struct list_head *it;

		list_for_each(it, head) {
			struct page *p = compound_head(lru_to_page(it));

			if (page_to_pfn(p) == cursor_pfn) {
				pos = it;
				break;
			}
		}
	}
	for (; pos != head && nr_cands < chunk_target; pos = next) {
		struct page *page = compound_head(lru_to_page(pos));
		next = pos->next;

		if (!PageAnon(page) || PageHuge(page) || !PageLRU(page))
			continue;
		if (unlikely(!page_evictable(page)))
			continue;
		if (!get_page_unless_zero(page))
			continue;
		cands[nr_cands++] = page;
	}
	if (pos != head) {
		struct page *p = compound_head(lru_to_page(pos));

		WRITE_ONCE(kagg_miner_cursor_pfn[nid], page_to_pfn(p));
		WRITE_ONCE(kagg_miner_cursor_valid[nid], true);
	} else {
		WRITE_ONCE(kagg_miner_cursor_valid[nid], false);
	}
	spin_unlock_irq(&pgdat->lru_lock);

	for (i = 0; i < nr_cands; i++) {
		struct page *page = cands[i];

		if (!isolate_lru_page(page)) {
			list_add_tail(&page->lru, local_list);
			isolated_ok++;
		} else {
			put_page(page);
		}
		if (!((i + 1) % KAGG_MINER_RESCHED_QUOTA))
			cond_resched();
	}
	if (nr_cands % KAGG_MINER_RESCHED_QUOTA)
		cond_resched();
#ifdef CONFIG_KAGGSWAPD_DEBUG
	atomic64_add(isolated_ok, &kagg_dbg_miner_isolated_pages);
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
					   struct list_head *all_isolated)
{
	struct page *page, *next;
	unsigned int i, work = 0;
	unsigned int sim_pct = (unsigned int)clamp_val(
		READ_ONCE(sysctl_dual_path_miner_similarity_pct), 1, 100);
	unsigned int min_cluster = (unsigned int)clamp_val(
		READ_ONCE(sysctl_dual_path_miner_min_cluster), 1,
		KAGG_MINER_CLUSTER_MAX);

	for (i = 0; i < KAGG_MINER_HASH_SIZE; i++) {
		struct hlist_head *bucket = &buckets[i];

		while (!hlist_empty(bucket)) {
			struct page_ext_agg_data *seed;
			struct page_ext_agg_data *cand;
			struct hlist_node *tmp;
			unsigned int cnt = 0;

			seed = hlist_entry(bucket->first,
					struct page_ext_agg_data, miner_node);
			hlist_del_init(&seed->miner_node);
			cnt++;

			hlist_for_each_entry_safe(cand, tmp, bucket, miner_node) {
				if (cnt >= KAGG_MINER_CLUSTER_MAX)
					break;
				if (!kagg_miner_similar(seed, cand, sim_pct))
					continue;
				hlist_del_init(&cand->miner_node);
				cnt++;
				work++;
				if (!(work % KAGG_MINER_RESCHED_QUOTA))
					cond_resched();
			}

			if (cnt >= min_cluster)
				kagg_miner_emit_cluster(lruvec, cnt);
		}
		cond_resched();
	}

	list_for_each_entry_safe(page, next, all_isolated, lru) {
		struct page_ext_agg *agg = lookup_page_ext_agg(page);
		struct page_ext_agg_data *d = page_ext_agg_get_data_maybe(agg);

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
#endif
	}
}

/* Mining plane entrypoint for one pgdat. */
static void kagg_miner_scan_pgdat(struct pglist_data *pgdat,
				  unsigned int seed_pct)
{
	struct lruvec *lruvec = mem_cgroup_lruvec(NULL, pgdat);
	struct hlist_head *buckets = kagg_miner_bucket_buf;
	LIST_HEAD(all_isolated);
	unsigned long inactive_sz;
	unsigned int view_target;
	unsigned int total_isolated = 0;
	unsigned int work = 0;

	if (!pgdat)
		return;
	inactive_sz = lruvec_lru_size(lruvec, LRU_INACTIVE_ANON, MAX_NR_ZONES - 1);
	view_target = kagg_percent_target(inactive_sz, seed_pct, KAGG_MINER_VIEW_MAX);
	if (!view_target)
		return;

	kagg_miner_init_buckets(buckets);
	while (total_isolated < view_target) {
		LIST_HEAD(batch_list);
		unsigned int chunk = min_t(unsigned int, KAGG_MINER_BATCH_MAX,
					   view_target - total_isolated);
		unsigned int got;

		got = kagg_miner_isolate_chunk(pgdat, chunk, &batch_list);
		if (!got)
			break;
		total_isolated += got;
		kagg_miner_bucketize_batch(&batch_list, buckets, &work);
		list_splice_tail_init(&batch_list, &all_isolated);
		cond_resched();
	}
	if (!list_empty(&all_isolated))
		kagg_miner_cluster_and_putback(lruvec, buckets, &all_isolated);
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

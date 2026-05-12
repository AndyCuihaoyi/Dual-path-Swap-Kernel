// SPDX-License-Identifier: GPL-2.0
/*
 * kaggswapd: aggregated anonymous LRU path — sampling today, pattern miner
 * and page_group linkage later (all guarded by CONFIG_KAGGSWAPD).
 *
 * LRU scanners invoke a caller-supplied action per compound head — no interim
 * allocation (callback / iterator pattern).  VMA-expand mode walks page tables
 * under mmap_lock; duplicate physical pages in one window are suppressed via
 * page_ext_agg_data::last_kagg_win (via lazy kmalloc inside page_ext_agg).
 */

#include <linux/freezer.h>
#include <linux/hugetlb.h>
#include <linux/kaggswapd.h>
#include <linux/kthread.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/nodemask.h>
#include <linux/page_ext.h>
#include <linux/pagewalk.h>
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

static DEFINE_SPINLOCK(kagg_win_id_lock);
static u16 kagg_win_next_id = 1;

/**
 * kagg_next_window_id - next u16 window id for per-page aggregation ring.
 *
 * Only bookkeeping is this counter + tiny lock — no widening of page_ext_agg.
 * Ids reuse 1..65535 wrapping at u16 overflow; miner must tolerate rare
 * ambiguity vs last_kagg_win after tens of thousands of scans.
 */
u16 kagg_next_window_id(void)
{
	unsigned long flags;
	u16 cur;

	spin_lock_irqsave(&kagg_win_id_lock, flags);
	cur = kagg_win_next_id;
	kagg_win_next_id++;
	if (unlikely(!kagg_win_next_id))
		kagg_win_next_id = 1;
	spin_unlock_irqrestore(&kagg_win_id_lock, flags);
	return cur;
}

#ifdef CONFIG_KAGGSWAPD_DEBUG

#define KAGG_DBG_RING	32

static atomic64_t kagg_dbg_scan_rounds;
static atomic64_t kagg_dbg_pages_stamped;
static atomic64_t kagg_dbg_dedup_same_win;

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
	seq_printf(m, "next_win_id %u (best-effort racy read)\n",
		   (unsigned int)READ_ONCE(kagg_win_next_id));
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
	if (d->last_kagg_win == win_id) {
#ifdef CONFIG_KAGGSWAPD_DEBUG
		atomic64_inc(&kagg_dbg_dedup_same_win);
#endif
		return;
	}

	page_ext_agg_push_window(agg, win_id);
	d->last_kagg_win = win_id;
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

static void kagg_scan_inactive_vma_expand(struct lruvec *lruvec,
					  unsigned long max_scan,
					  u16 win_id)
{
	pg_data_t *pgdat = lruvec_pgdat(lruvec);
	struct list_head *pos;
	unsigned long scanned = 0;

	if (!max_scan)
		return;

	spin_lock_irq(&pgdat->lru_lock);

	pos = lruvec->lists[LRU_INACTIVE_ANON].next;
	while (scanned < max_scan &&
	       pos != &lruvec->lists[LRU_INACTIVE_ANON]) {
		struct page *page = lru_to_page(pos);
		struct list_head *next_pos = pos->next;

		scanned++;

		page = compound_head(page);
		if (!PageAnon(page) || PageHuge(page)) {
			pos = next_pos;
			continue;
		}

		if (!get_page_unless_zero(page)) {
			pos = next_pos;
			continue;
		}

		spin_unlock_irq(&pgdat->lru_lock);

		kagg_expand_mappings_for_seed(page, win_id);
		put_page(page);

		spin_lock_irq(&pgdat->lru_lock);
		pos = next_pos;
	}

	spin_unlock_irq(&pgdat->lru_lock);
}

static void kagg_scan_lru_bounded(struct lruvec *lruvec, enum lru_list lruid,
				  unsigned long max_scan,
				  u16 win_id,
				  kagg_page_action_fn action_fn,
				  void *priv)
{
	struct page *page;
	unsigned long scanned = 0;
	pg_data_t *pgdat = lruvec_pgdat(lruvec);

	if (!max_scan || !action_fn)
		return;

	spin_lock_irq(&pgdat->lru_lock);
	list_for_each_entry(page, &lruvec->lists[lruid], lru) {
		page = compound_head(page);
		scanned++;

		action_fn(page, win_id, priv);

		if (scanned >= max_scan)
			break;
	}
	spin_unlock_irq(&pgdat->lru_lock);
}

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
		struct page *page;

		spin_lock_irq(&pgdat->lru_lock);

		list_for_each_entry(page, &lruvec->lists[LRU_INACTIVE_ANON], lru) {
			page = compound_head(page);
			action_fn(page, win_id, priv);
		}
		list_for_each_entry(page, &lruvec->lists[LRU_ACTIVE_ANON], lru) {
			page = compound_head(page);
			action_fn(page, win_id, priv);
		}
		spin_unlock_irq(&pgdat->lru_lock);
		break;
	}

	case KAGG_SCAN_INACTIVE_VMA_EXPAND_PCT:
		max_scan = kagg_bounded_max_scan(inactive_sz, scan_pct);
		kagg_scan_inactive_vma_expand(lruvec, max_scan, win_id);
		break;

	default:
		/*
		 * KAGG_SCAN_MODE_NR and clamp mistakes: mode is pinned to
		 * [0, KAGG_SCAN_MODE_NR - 1] above; keep default for -Wswitch.
		 */
		break;
	}
}

static int kaggswapd_fn(void *unused)
{
	enum kagg_scan_mode mode;
	unsigned int pct;
	unsigned int interval;
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

		win_id = kagg_next_window_id();

		for_each_online_pgdat(pgdat) {
			kagg_sample_scan_pgdat(pgdat, mode, pct, win_id,
					      kagg_default_sample_mark, NULL);
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

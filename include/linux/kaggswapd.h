/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_KAGGSWAPD_H
#define _LINUX_KAGGSWAPD_H

#ifdef CONFIG_KAGGSWAPD

#include <linux/types.h>

struct page;
struct pglist_data;

/*
 * Sampling scan strategies (sysctl dual_path_scan_mode, 0..3):
 *
 *	INACTIVE_PCT — inactive anon LRU, list head order; first scan_pct %% slots.
 *	ACTIVE_PCT   — active anon LRU, same rule.
 *	ALL_ANON     — full inactive + active anon (pct ignored).
 *	INACTIVE_VMA_EXPAND_PCT — inactive anon prefix as seeds; rmap each seed,
 *		then walk each covering anonymous VMA (walk_page_vma); physical
 *		pages deduped per window via page_ext_agg_data::last_kagg_win.
 *
 * Window ids are u16 from kagg_next_window_id(): a global/in-RAM allocator only
 * (no extra per-page fields). Ids recycle 1..65535 with rare alias versus
 * last_kagg_win after long uptime — acceptable for heuristic sampling.
 */
enum kagg_scan_mode {
	KAGG_SCAN_INACTIVE_PCT = 0,
	KAGG_SCAN_ACTIVE_PCT,
	KAGG_SCAN_ALL_ANON,
	KAGG_SCAN_INACTIVE_VMA_EXPAND_PCT,
	KAGG_SCAN_MODE_NR,
};

typedef void (*kagg_page_action_fn)(struct page *page, u16 win_id,
					void *priv);

extern int sysctl_dual_path_scan_mode;
extern int sysctl_dual_path_scan_pct;
extern int sysctl_dual_path_scan_interval_ms;

extern int sysctl_dual_path_miner_enable;
extern int sysctl_dual_path_miner_interval_ms;
extern int sysctl_dual_path_miner_similarity_pct;
extern int sysctl_dual_path_miner_min_cluster;

void kagg_default_sample_mark(struct page *page, u16 win_id, void *priv);

u16 kagg_next_window_id(void);
void kagg_sample_scan_pgdat(struct pglist_data *pgdat,
				enum kagg_scan_mode scan_mode,
				unsigned int scan_pct, u16 win_id,
				kagg_page_action_fn action_fn,
				void *priv);

#if defined(CONFIG_MEMCG)
struct mem_cgroup;
void kagg_memcg_meta_free(struct mem_cgroup *memcg);
#endif

#endif /* CONFIG_KAGGSWAPD */

#endif /* _LINUX_KAGGSWAPD_H */

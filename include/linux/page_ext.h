/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __LINUX_PAGE_EXT_H
#define __LINUX_PAGE_EXT_H

#include <linux/compiler.h>
#include <linux/types.h>
#include <linux/list.h>
#include <linux/stacktrace.h>
#include <linux/stackdepot.h>

struct pglist_data;
struct page_ext_operations {
	size_t offset;
	size_t size;
	bool (*need)(void);
	void (*init)(void);
};

#ifdef CONFIG_PAGE_EXTENSION

enum page_ext_flags {
	PAGE_EXT_OWNER,
	PAGE_EXT_OWNER_ALLOCATED,
#if defined(CONFIG_IDLE_PAGE_TRACKING) && !defined(CONFIG_64BIT)
	PAGE_EXT_YOUNG,
	PAGE_EXT_IDLE,
#endif
};

/*
 * Page Extension can be considered as an extended mem_map.
 * A page_ext page is associated with every page descriptor. The
 * page_ext helps us add more information about the page.
 * All page_ext are allocated at boot or memory hotplug event,
 * then the page_ext for pfn always exists.
 */
struct page_ext {
	unsigned long flags;
};

extern unsigned long page_ext_size;
extern void pgdat_page_ext_init(struct pglist_data *pgdat);

#ifdef CONFIG_SPARSEMEM
static inline void page_ext_init_flatmem(void)
{
}
extern void page_ext_init(void);
#else
extern void page_ext_init_flatmem(void);
static inline void page_ext_init(void)
{
}
#endif

struct page_ext *lookup_page_ext(const struct page *page);

#if defined(CONFIG_DUAL_PATH_SWAP)
struct page_group;

#define MAX_PAGE_EXT_AGG_WINDOW		16

struct page_ext_agg_data;

/*
 * Tail slot per PFN from struct page_ext: only a pointer.  Anonymous sampling
 * lazily allocates page_ext_agg_data from a kmem_cache; untouched pages keep
 * data == NULL (no slab object).  PFN recycle must pair with
 * dual_path_page_ext_prepare_free().
 */
struct page_ext_agg {
	struct page_ext_agg_data	*data;
};

/**
 * struct page_ext_agg_data - ring + group link + intrusive miner hash node.
 *
 * The miner uses miner_node for in-place bucket chaining (no extra allocation).
 * Member struct page * for a cluster is resolved after bucketing by scanning
 * the miner-isolated page list (see kagg_miner_emit_cluster()).
 */
struct page_ext_agg_data {
	struct page_group		*group;
	struct hlist_node		miner_node;
	u16				window_ids[MAX_PAGE_EXT_AGG_WINDOW];
	int				head;
	u16				last_kagg_win;
};

extern struct page_ext_operations page_ext_agg_ops;

void page_ext_agg_reset(struct page_ext_agg *agg);

void page_ext_agg_push_window(struct page_ext_agg *agg, u16 win_id);

void dual_path_page_ext_prepare_free(struct page *page, unsigned int order);

/**
 * Lock-free slab install after page_ext_agg_ensure_data() inline READ_ONCE misses.
 *
 * Calling context may hold pgdat->lru_lock or page-table PTL — must remain
 * non-blocking: kmem_cache allocator flags must stay GFP_NOWAIT (or equivalently
 * GFP_ATOMIC-type); never GFP_KERNEL or direct reclaim can deadlock.
 * Concurrent installers on the same @agg are serialized by cmpxchg on &agg->data.
 *
 * Implemented in mm/page_ext.c (only declaration lives here).
 */
struct page_ext_agg_data *__page_ext_agg_ensure_slow(struct page_ext_agg *agg);

/* Fast path inlined here; misses call __page_ext_agg_ensure_slow(). */
static inline struct page_ext_agg_data *page_ext_agg_ensure_data(
	struct page_ext_agg *agg)
{
	struct page_ext_agg_data *d;

	if (!agg)
		return NULL;
	d = READ_ONCE(agg->data);
	if (likely(d))
		return d;
	return __page_ext_agg_ensure_slow(agg);
}

/**
 * Append @win_id to ring — @d must be non-NULL (callers gate via ensure_data).
 */
static inline void page_ext_agg_push_window_into(struct page_ext_agg_data *d,
						 u16 win_id)
{
	int i;

	if (!d)
		return;
	i = READ_ONCE(d->head);
	if (unlikely((unsigned int)i >= MAX_PAGE_EXT_AGG_WINDOW))
		i = 0;
	d->window_ids[i] = win_id;
	i++;
	if (unlikely(i >= MAX_PAGE_EXT_AGG_WINDOW))
		i = 0;
	WRITE_ONCE(d->head, i);
}

static inline struct page_ext_agg_data *page_ext_agg_get_data_maybe(
	const struct page_ext_agg *agg)
{
	return agg ? READ_ONCE(((struct page_ext_agg *)agg)->data) : NULL;
}

static inline struct page_ext_agg *page_ext_get_agg(struct page_ext *page_ext)
{
	return (void *)page_ext + page_ext_agg_ops.offset;
}

static inline struct page_ext_agg *lookup_page_ext_agg(const struct page *page)
{
	struct page_ext *ext = lookup_page_ext(page);

	if (!ext)
		return NULL;
	return page_ext_get_agg(ext);
}

#else /* !CONFIG_DUAL_PATH_SWAP */

static inline void dual_path_page_ext_prepare_free(struct page *p, unsigned int o)
{
}
#endif /* CONFIG_DUAL_PATH_SWAP */

static inline struct page_ext *page_ext_next(struct page_ext *curr)
{
	void *next = curr;
	next += page_ext_size;
	return next;
}

#else /* !CONFIG_PAGE_EXTENSION */
struct page_ext;

static inline void pgdat_page_ext_init(struct pglist_data *pgdat)
{
}

static inline struct page_ext *lookup_page_ext(const struct page *page)
{
	return NULL;
}

static inline void page_ext_init(void)
{
}

static inline void page_ext_init_flatmem(void)
{
}
#endif /* CONFIG_PAGE_EXTENSION */
#endif /* __LINUX_PAGE_EXT_H */
